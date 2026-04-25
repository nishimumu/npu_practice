from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


def get_file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def get_onnx_model_size_mb(path: Path) -> float:
    total_size = get_file_size_mb(path)

    external_data_path = path.with_name(path.name + ".data")
    if external_data_path.exists():
        total_size += get_file_size_mb(external_data_path)

    return total_size


def build_mnist_loaders(data_dir: Path) -> tuple[DataLoader, DataLoader, datasets.MNIST]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    calibration_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    return calibration_loader, test_loader, test_dataset


class MnistCalibrationDataReader(CalibrationDataReader):
    def __init__(
        self,
        data_loader: DataLoader,
        input_name: str,
        num_batches: int,
    ):
        self.data_loader = data_loader
        self.input_name = input_name
        self.num_batches = num_batches
        self.data_iter = iter(self._iter_batches())

    def _iter_batches(self):
        for batch_idx, (images, _) in enumerate(self.data_loader, start=1):
            yield {
                self.input_name: images.numpy().astype(np.float32),
            }

            if batch_idx >= self.num_batches:
                break

    def get_next(self):
        return next(self.data_iter, None)


def create_onnx_session(model_path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )


def evaluate_onnx(session: ort.InferenceSession, test_loader: DataLoader) -> tuple[float, float]:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in test_loader:
        outputs = session.run(
            [output_name],
            {
                input_name: images.numpy().astype(np.float32),
            },
        )[0]

        output_tensor = torch.from_numpy(outputs)
        loss = F.cross_entropy(output_tensor, labels)

        total_loss += loss.item() * images.size(0)

        predictions = output_tensor.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total * 100

    return avg_loss, accuracy


def main():
    torch.manual_seed(42)

    fp32_onnx_path = Path("mnist_small_cnn_fp32.onnx")
    int8_onnx_path = Path("mnist_small_cnn_static_int8.onnx")

    if not fp32_onnx_path.exists():
        raise FileNotFoundError(
            f"{fp32_onnx_path} was not found. "
            "Run export_to_onnx.py first to create the FP32 ONNX model."
        )

    calibration_loader, test_loader, test_dataset = build_mnist_loaders(Path("data"))

    fp32_session = create_onnx_session(fp32_onnx_path)
    input_name = fp32_session.get_inputs()[0].name

    calibration_batches = 20
    calibration_reader = MnistCalibrationDataReader(
        data_loader=calibration_loader,
        input_name=input_name,
        num_batches=calibration_batches,
    )

    print("=" * 60)
    print("Static ONNX INT8 quantization")
    print("=" * 60)
    print(f"Input model        : {fp32_onnx_path}")
    print(f"Output model       : {int8_onnx_path}")
    print(f"Calibration batches: {calibration_batches}")
    print()

    quantize_static(
        model_input=fp32_onnx_path,
        model_output=int8_onnx_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        use_external_data_format=False,
    )

    onnx_model = onnx.load(int8_onnx_path)
    onnx.checker.check_model(onnx_model)
    print("Quantized ONNX model check: OK")
    print()

    int8_session = create_onnx_session(int8_onnx_path)

    fp32_loss, fp32_acc = evaluate_onnx(fp32_session, test_loader)
    int8_loss, int8_acc = evaluate_onnx(int8_session, test_loader)

    print("=" * 60)
    print("Accuracy comparison")
    print("=" * 60)
    print(f"ONNX FP32        | Loss: {fp32_loss:.4f} | Accuracy: {fp32_acc:.2f}%")
    print(f"ONNX Static INT8 | Loss: {int8_loss:.4f} | Accuracy: {int8_acc:.2f}%")
    print()

    print("=" * 60)
    print("File size comparison")
    print("=" * 60)
    print(f"ONNX FP32        : {get_onnx_model_size_mb(fp32_onnx_path):.3f} MB")
    print(f"ONNX Static INT8 : {get_onnx_model_size_mb(int8_onnx_path):.3f} MB")
    print()

    sample_image, sample_label = test_dataset[0]
    sample_input = sample_image.unsqueeze(0).numpy().astype(np.float32)

    fp32_output = fp32_session.run(
        [fp32_session.get_outputs()[0].name],
        {
            fp32_session.get_inputs()[0].name: sample_input,
        },
    )[0]
    int8_output = int8_session.run(
        [int8_session.get_outputs()[0].name],
        {
            int8_session.get_inputs()[0].name: sample_input,
        },
    )[0]

    fp32_pred = int(np.argmax(fp32_output, axis=1)[0])
    int8_pred = int(np.argmax(int8_output, axis=1)[0])

    print("=" * 60)
    print("Single image inference")
    print("=" * 60)
    print(f"Label          : {sample_label}")
    print(f"FP32 prediction: {fp32_pred}")
    print(f"INT8 prediction: {int8_pred}")

    print()
    print("Static ONNX quantization practice finished.")


if __name__ == "__main__":
    main()
