from pathlib import Path
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SmallCNN
from quantize_dynamic import get_file_size_mb, print_model_summary
from test import evaluate


class StaticQuantWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        # INT8に変換
        self.quant = torch.ao.quantization.QuantStub()
        self.model = model
        # FLOAT32に戻す
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


def select_quantization_backend() -> tuple[str, torch.ao.quantization.QConfig]:
    supported_engines = torch.backends.quantized.supported_engines

    # 使えるバックエンドを探してpytorchに設定する
    for backend in ("fbgemm", "x86", "onednn", "qnnpack"):
        if backend not in supported_engines:
            continue

        try:
            qconfig = torch.ao.quantization.get_default_qconfig(backend)
        except Exception:
            continue

        torch.backends.quantized.engine = backend
        return backend, qconfig

    raise RuntimeError(
        "No supported static quantization backend was found. "
        f"Supported engines: {supported_engines}"
    )


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


def calibrate(model: nn.Module, device: torch.device, data_loader: DataLoader, num_batches: int) -> None:
    model.eval()

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(data_loader, start=1):
            model(images.to(device))

            if batch_idx >= num_batches:
                break


def main():
    torch.manual_seed(42)

    device = torch.device("cpu")
    print("Device:", device)

    original_weight_path = Path("mnist_small_cnn_cpu.pth")

    if not original_weight_path.exists():
        raise FileNotFoundError(
            f"{original_weight_path} was not found. "
            "Run train.py first to create the trained FP32 weights."
        )

    calibration_loader, test_loader, test_dataset = build_mnist_loaders(Path("data"))

    fp32_model = SmallCNN().to(device)
    fp32_model.load_state_dict(
        torch.load(original_weight_path, map_location=device)
    )
    fp32_model.eval()

    backend, qconfig = select_quantization_backend()
    print("Quantization backend:", backend)

    prepared_model = StaticQuantWrapper(copy.deepcopy(fp32_model)).to(device)
    prepared_model.eval()
    prepared_model.qconfig = qconfig

    # Observerを起動させてcalibration観測準備を行う
    torch.ao.quantization.prepare(prepared_model, inplace=True)

    calibration_batches = 20
    print(f"Calibrating with {calibration_batches} batches...")
    calibrate(
        model=prepared_model,
        device=device,
        data_loader=calibration_loader,
        num_batches=calibration_batches,
    )

    # calibrationで観測された値に基づきスケールを設定する
    int8_static_model = torch.ao.quantization.convert(prepared_model, inplace=False)
    int8_static_model.eval()

    print_model_summary(fp32_model, "Original FP32 model")
    print_model_summary(int8_static_model, "Static quantized INT8 model")

    fp32_loss, fp32_acc = evaluate(fp32_model, device, test_loader)
    int8_loss, int8_acc = evaluate(int8_static_model, device, test_loader)

    print("=" * 60)
    print("Accuracy comparison")
    print("=" * 60)
    print(f"FP32         | Loss: {fp32_loss:.4f} | Accuracy: {fp32_acc:.2f}%")
    print(f"Static INT8  | Loss: {int8_loss:.4f} | Accuracy: {int8_acc:.2f}%")
    print()

    fp32_save_path = Path("mnist_small_cnn_fp32_state_dict.pth")
    int8_save_path = Path("mnist_small_cnn_static_int8_state_dict.pth")

    torch.save(fp32_model.state_dict(), fp32_save_path)
    torch.save(int8_static_model.state_dict(), int8_save_path)

    print("=" * 60)
    print("File size comparison")
    print("=" * 60)
    print(f"FP32 state_dict        : {get_file_size_mb(fp32_save_path):.3f} MB")
    print(f"Static INT8 state_dict : {get_file_size_mb(int8_save_path):.3f} MB")
    print()

    sample_image, sample_label = test_dataset[0]
    sample_input = sample_image.unsqueeze(0).to(device)

    with torch.no_grad():
        fp32_output = fp32_model(sample_input)
        int8_output = int8_static_model(sample_input)

    fp32_pred = fp32_output.argmax(dim=1).item()
    int8_pred = int8_output.argmax(dim=1).item()

    print("=" * 60)
    print("Single image inference")
    print("=" * 60)
    print(f"Label          : {sample_label}")
    print(f"FP32 prediction: {fp32_pred}")
    print(f"INT8 prediction: {int8_pred}")

    print()
    print("Static quantization practice finished.")


if __name__ == "__main__":
    main()
