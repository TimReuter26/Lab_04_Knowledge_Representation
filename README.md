# Lab 04: 命名实体识别实验

本仓库包含知识表示与推理实验四的命名实体识别代码、数据处理结果、训练脚本和 Colab 训练入口。

## 内容

- `Lab_04_content.pdf`：实验手册。
- `实验四_命名实体识别/process_todo.py`：DuIE 数据转 BIO 格式。
- `实验四_命名实体识别/data_loader_todo.py`：NER 数据集加载与 padding。
- `实验四_命名实体识别/model_todo.py`：BERT-BiLSTM-CRF 模型。
- `实验四_命名实体识别/main_todo.py`：训练、保存与测试逻辑。
- `实验四_命名实体识别/train_controlled.py`：WSL CPU 可控规模训练脚本。
- `实验四_命名实体识别/ltp_ner_demo.py`：LTP NER 示例脚本。
- `实验四_命名实体识别/data/duie/ner_data/`：已生成的 BIO 训练/验证数据。
- `notebooks/lab04_bert_ner_colab.ipynb`：Colab T4 GPU 训练 notebook。
- `*_results.md`：实验过程和报告素材记录。

## 大文件说明

以下文件不提交到 GitHub：

- `实验四_命名实体识别/model_hub/`
- `实验四_命名实体识别/checkpoint/`
- `实验四_命名实体识别/ltp_small/pytorch_model.bin`

原因是模型权重和 checkpoint 超过 GitHub 普通仓库限制。Colab notebook 会自动下载 `hfl/chinese-bert-wwm-ext`，训练后会重新生成 checkpoint。

## WSL CPU 快速训练

```bash
cd /home/mmama/workspace/AI_Projects/Lab_04_content/实验四_命名实体识别
/home/mmama/miniconda3/envs/mnist_exp/bin/python train_controlled.py
```

## Colab GPU 训练

1. 打开 `notebooks/lab04_bert_ner_colab.ipynb`。
2. 在 Colab 中选择 `Runtime -> Change runtime type -> T4 GPU`。
3. 设置 notebook 里的 `REPO_URL` 为本仓库 URL。
4. 依次运行所有单元格。

Colab 默认使用全量训练数据：

- train：10000 条
- dev：1000 条
- epoch：1
- batch size：12
- max sequence length：256

