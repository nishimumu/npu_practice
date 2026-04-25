from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SmallCNN
from test import evaluate

def get_file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024

def print_model_summary(model: nn.Module, title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(model)
    print()

def main():
    torch.manual_seed(42)

    device = torch.device("cpu")
    print("Device:", device)

    original_weight_path = Path("mnist_small_cnn_cpu.pth")

    if not original_weight_path.exists():
        raise FileNotFoundError(
            f"{original_weight_path} が見つかりません。"
            "先にMNIST学習プログラムを実行して重みを作成してください。"
        )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    test_dataset = datasets.MNIST(
        root=Path("data"),
        train=False,
        download=True,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    # 1. 元のfloat32モデルを作成して重みを読み込む
    fp32_model = SmallCNN().to(device)
    fp32_model.load_state_dict(
        torch.load(original_weight_path, map_location=device)
    )
    fp32_model.eval()

    # 2. 動的量子化
    # 今回は nn.Linear のみ量子化する
    int8_dynamic_model = torch.ao.quantization.quantize_dynamic(
        model=copy.deepcopy(fp32_model),
        qconfig_spec={nn.Linear},
        dtype=torch.qint8,
        inplace=False,
    )
    int8_dynamic_model.eval()

    # 3. モデル構造を表示
    print_model_summary(fp32_model, "Original FP32 model")
    print_model_summary(int8_dynamic_model, "Dynamic quantized INT8 model")

    # 4. 精度を比較
    fp32_loss, fp32_acc = evaluate(fp32_model, device, test_loader)
    int8_loss, int8_acc = evaluate(int8_dynamic_model, device, test_loader)

    print("=" * 60)
    print("Accuracy comparison")
    print("=" * 60)
    print(f"FP32          | Loss: {fp32_loss:.4f} | Accuracy: {fp32_acc:.2f}%")
    print(f"Dynamic INT8  | Loss: {int8_loss:.4f} | Accuracy: {int8_acc:.2f}%")
    print()

    # 5. state_dictを保存してファイルサイズ比較
    fp32_save_path = Path("mnist_small_cnn_fp32_state_dict.pth")
    int8_save_path = Path("mnist_small_cnn_dynamic_int8_state_dict.pth")

    torch.save(fp32_model.state_dict(), fp32_save_path)
    torch.save(int8_dynamic_model.state_dict(), int8_save_path)

    print("=" * 60)
    print("File size comparison")
    print("=" * 60)
    print(f"FP32 state_dict         : {get_file_size_mb(fp32_save_path):.3f} MB")
    print(f"Dynamic INT8 state_dict : {get_file_size_mb(int8_save_path):.3f} MB")
    print()

    # 6. 1枚だけ推論して結果を見る
    sample_image, sample_label = test_dataset[0]
    sample_input = sample_image.unsqueeze(0).to(device)

    with torch.no_grad():
        fp32_output = fp32_model(sample_input)
        int8_output = int8_dynamic_model(sample_input)

    fp32_pred = fp32_output.argmax(dim=1).item()
    int8_pred = int8_output.argmax(dim=1).item()

    print("=" * 60)
    print("Single image inference")
    print("=" * 60)
    print(f"Label          : {sample_label}")
    print(f"FP32 prediction: {fp32_pred}")
    print(f"INT8 prediction: {int8_pred}")

    print()
    print("Dynamic quantization practice finished.")


if __name__ == "__main__":
    main()