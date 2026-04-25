# Name
NPU_PRACTICE
GPUではなくNPUにAIモデルを実装するための練習サンプル
 
# Requirement
pyproject.toml参照
 
# Usage
- train.py: MNISTを題材として小型CNNの学習を実行する
- quantize_dynamic.py: DYNAMICな量子化を行い、モデルサイズが小さくなることを確認するサンプル
- quantize_static.py: STATICな量子化を行い、モデルサイズが小さくなることを確認するサンプル
- train_qat.py: QAT(Quantization Aware Training)を用いて学習を実行するサンプル
- quantize_static_onnx.py: ONNXでFP32モデルをINT8量子化するサンプル
- export_to_onnx.py: PyTorchのモデルをonnxに変換するサンプル

# Note
量子化の手法
- PTQ: 学習済みモデルを後から量子化する
  - STATIC: 事前にactivationのスケールを決める
  - DYNAMIC: 実行時にactivationのスケールを決める
- QAT: 量子化される前提で学習・微調整

# 量子化学習ロードマップ
<input type="checkbox" checked> PyTorchモデル作成<br>
↓<br>
<input type="checkbox" checked> 学習<br>
↓<br>
<input type="checkbox" checked> dynamic quantization<br>
  torch.ao.quantization.quantize_dynamicの理解<br>
↓<br>
<input type="checkbox" checked> static quantization<br>
  prepare -> calibration -> convertの流れの理解<br>
↓<br>
<input type="checkbox" checked> QAT (Quantization Aware Training)<br>
  prepare -> 追加学習(前半: Observer有効、後半: Observer無効) -> convertの流れの理解<br>

# NPU前提量子化学習ロードマップ
<input type="checkbox" checked> PyTorchモデル作成 (tarin.py)<br>
↓<br>
<input type="checkbox" checked> 学習 (train.py)<br>
↓<br>
<input type="checkbox" checked> ONNX export (export_to_onnx.py)<br>
↓<br>
<input type="checkbox" checked> ONNX Runtimeで推論確認 (export_to_onnx.py)<br>
↓<br>
<input type="checkbox" checked> ONNX RuntimeでINT8量子化 (quantize_static_onnx.py)<br>
↓<br>
<input type="checkbox" checked> ONNX Runtimeで推論確認 (quantize_static_onnx.py)<br>
↓<br>
<input type="checkbox" unchecked> NetronでQDQ / QOperator構造を見る<br>
