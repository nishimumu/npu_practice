# Name
NPU_PRACTICE
GPUではなくNPUにAIモデルを実装するための練習サンプル
 
# Requirement
pyproject.toml参照
 
# Usage
train.py: MNISTを題材として小型CNNの学習を実行する
quantize_dynamic.py: DYNAMICな量子化を行い、モデルサイズが小さくなることを確認するサンプル
quantize_static.py: STATICな量子化を行い、モデルサイズが小さくなることを確認するサンプル
 
# Note
量子化の手法
- PTQ: 学習済みモデルを後から量子化する
-- STATIC: 事前にactivationのスケールを決める
-- DYNAMIC: 実行時にactivationのスケールを決める
- QAT: 量子化される前提で学習・微調整

# 学習ロードマップ
[x]PyTorchモデル作成
↓
[x]学習
↓
[x]dynamic quantization
- torch.ao.quantization.quantize_dynamicの理解
↓
[x]static quantization
- prepare -> calibration -> convertの流れの理解
↓
QAT
↓
ONNX export
↓
ONNX Runtimeで推論確認
↓
FP32 / INT8 / ONNX出力比較
↓
モデル構造・op確認
