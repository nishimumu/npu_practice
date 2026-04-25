from pathlib import Path
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SmallCNN
from quantize_dynamic import get_file_size_mb, print_model_summary
from quantize_static import StaticQuantWrapper, select_quantization_backend
from test import evaluate
from train import train_one_epoch


def build_qat_model(fp32_model: nn.Module, qconfig: torch.ao.quantization.QConfig) -> nn.Module:
    qat_model = StaticQuantWrapper(copy.deepcopy(fp32_model))
    qat_model.train()
    qat_model.qconfig = qconfig
    torch.ao.quantization.prepare_qat(qat_model, inplace=True)
    return qat_model


def main():
    torch.manual_seed(42)

    device = torch.device("cpu")
    print("Device:", device)

    data_dir = Path("data")
    original_weight_path = Path("mnist_small_cnn_cpu.pth")

    if not original_weight_path.exists():
        raise FileNotFoundError(
            f"{original_weight_path} was not found. "
            "Run train.py first to create the trained FP32 weights."
        )

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    fp32_model = SmallCNN().to(device)
    fp32_model.load_state_dict(
        torch.load(original_weight_path, map_location=device)
    )
    fp32_model.eval()

    backend, qat_qconfig = select_quantization_backend(qat=True)
    print("Quantization backend:", backend)

    qat_model = build_qat_model(fp32_model, qat_qconfig).to(device)

    optimizer = torch.optim.Adam(
        qat_model.parameters(),
        lr=0.0001,
    )

    epochs = 3

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=qat_model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
        )

        test_loss, test_acc = evaluate(
            model=qat_model,
            device=device,
            test_loader=test_loader,
        )

        print("-" * 60)
        print(f"Epoch {epoch} finished")
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
        print(f"Test  Loss: {test_loss:.4f} | Test  Accuracy: {test_acc:.2f}%")
        print("-" * 60)

        if epoch == 2:
            qat_model.apply(torch.ao.quantization.disable_observer)
            print("Fake-quant observers disabled.")

    qat_model.eval()
    int8_qat_model = torch.ao.quantization.convert(qat_model, inplace=False)
    int8_qat_model.eval()

    print_model_summary(fp32_model, "Original FP32 model")
    print_model_summary(int8_qat_model, "QAT converted INT8 model")

    fp32_loss, fp32_acc = evaluate(fp32_model, device, test_loader)
    qat_loss, qat_acc = evaluate(qat_model, device, test_loader)
    int8_loss, int8_acc = evaluate(int8_qat_model, device, test_loader)

    print("=" * 60)
    print("Accuracy comparison")
    print("=" * 60)
    print(f"FP32              | Loss: {fp32_loss:.4f} | Accuracy: {fp32_acc:.2f}%")
    print(f"QAT fake-quant    | Loss: {qat_loss:.4f} | Accuracy: {qat_acc:.2f}%")
    print(f"QAT converted INT8| Loss: {int8_loss:.4f} | Accuracy: {int8_acc:.2f}%")
    print()

    fp32_save_path = Path("mnist_small_cnn_fp32_state_dict.pth")
    qat_save_path = Path("mnist_small_cnn_qat_fake_quant_state_dict.pth")
    int8_save_path = Path("mnist_small_cnn_qat_int8_state_dict.pth")

    torch.save(fp32_model.state_dict(), fp32_save_path)
    torch.save(qat_model.state_dict(), qat_save_path)
    torch.save(int8_qat_model.state_dict(), int8_save_path)

    print("=" * 60)
    print("File size comparison")
    print("=" * 60)
    print(f"FP32 state_dict              : {get_file_size_mb(fp32_save_path):.3f} MB")
    print(f"QAT fake-quant state_dict    : {get_file_size_mb(qat_save_path):.3f} MB")
    print(f"QAT converted INT8 state_dict: {get_file_size_mb(int8_save_path):.3f} MB")
    print()

    sample_image, sample_label = test_dataset[0]
    sample_input = sample_image.unsqueeze(0).to(device)

    with torch.no_grad():
        fp32_output = fp32_model(sample_input)
        qat_output = qat_model(sample_input)
        int8_output = int8_qat_model(sample_input)

    fp32_pred = fp32_output.argmax(dim=1).item()
    qat_pred = qat_output.argmax(dim=1).item()
    int8_pred = int8_output.argmax(dim=1).item()

    print("=" * 60)
    print("Single image inference")
    print("=" * 60)
    print(f"Label               : {sample_label}")
    print(f"FP32 prediction     : {fp32_pred}")
    print(f"QAT prediction      : {qat_pred}")
    print(f"INT8 prediction     : {int8_pred}")

    print()
    print("QAT practice finished.")


if __name__ == "__main__":
    main()
