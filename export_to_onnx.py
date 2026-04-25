from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import SmallCNN

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    weight_path = Path("mnist_small_cnn_cpu.pth")
    onnx_path = Path("mnist_small_cnn_fp32.onnx")

    if not weight_path.exists():
        raise FileNotFoundError(
            f"{weight_path} が見つかりません。先に学習済み重みを作成してください。"
        )

    model = SmallCNN()
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()

    # ONNX export時に使うダミー入力。
    # MNISTなので shape は [batch, channel, height, width] = [1, 1, 28, 28]
    dummy_input = torch.randn(1, 1, 28, 28)

    # PyTorchモデルをONNX形式へ変換
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    print(f"Exported ONNX model: {onnx_path}")

    # ONNXモデルとして正しいかチェック
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check: OK")

    # PyTorch出力とONNX Runtime出力を比較
    test_input = torch.randn(1, 1, 28, 28)

    with torch.no_grad():
        torch_output = model(test_input).numpy()

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    onnx_inputs = {
        "input": test_input.numpy().astype(np.float32)
    }

    onnx_output = session.run(
        ["logits"],
        onnx_inputs,
    )[0]

    max_abs_diff = np.max(np.abs(torch_output - onnx_output))

    print("PyTorch output:")
    print(torch_output)

    print("ONNX Runtime output:")
    print(onnx_output)

    print(f"Max absolute difference: {max_abs_diff:.8f}")

    if max_abs_diff < 1e-4:
        print("PyTorch and ONNX Runtime outputs are close.")
    else:
        print("Warning: outputs differ more than expected.")


if __name__ == "__main__":
    main()
