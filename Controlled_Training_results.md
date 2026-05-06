# BERT-BiLSTM-CRF 可控规模训练结果

## 预训练模型处理

实验手册要求使用 `hfl/chinese-bert-wwm-ext` 作为 BERT 预训练模型。项目中原本缺少以下目录：

`实验四_命名实体识别/model_hub/chinese-bert-wwm-ext/`

已使用 Hugging Face 下载模型到本地目录，并验证 `BertTokenizer` 与 `BertModel` 可正常加载。

模型文件示例：

| 文件 | 大小 |
| --- | ---: |
| `model_hub/chinese-bert-wwm-ext/pytorch_model.bin` | 411578458 bytes |
| `model_hub/chinese-bert-wwm-ext/config.json` | 647 bytes |
| `model_hub/chinese-bert-wwm-ext/vocab.txt` | 109540 bytes |

加载验证：

| 项目 | 值 |
| --- | --- |
| tokenizer vocab size | 21128 |
| hidden size | 768 |
| BERT layers | 12 |

## 训练环境

| 项目 | 值 |
| --- | --- |
| Python | `/home/mmama/miniconda3/envs/mnist_exp/bin/python` |
| PyTorch | `2.11.0+cpu` |
| Transformers | `4.27.4` |
| pytorch-crf | `0.7.2` |
| CUDA | 不可用 |

说明：当前 WSL 环境只能使用 CPU。全量 DuIE 训练预计耗时较长，因此本步骤按照实验手册“可根据硬件配置调整训练设置和训练集、测试集大小”的要求，执行可控规模训练，验证训练流程和权重保存。

## 训练脚本

训练脚本：

`实验四_命名实体识别/train_controlled.py`

运行命令：

```bash
cd /home/mmama/workspace/AI_Projects/Lab_04_content/实验四_命名实体识别
/home/mmama/miniconda3/envs/mnist_exp/bin/python train_controlled.py
```

## 训练参数

| 参数 | 值 |
| --- | ---: |
| train samples | 120 |
| dev samples | 24 |
| epoch | 1 |
| train batch size | 2 |
| dev batch size | 2 |
| max seq len | 256 |
| BIO labels | 49 |
| train steps | 60 |

## 训练成功日志节选

```text
device: cpu
train_samples: 120
dev_samples: 24
max_seq_len: 256
num_labels: 49
【train】1/1 1/60 loss:238.9073486328125
【train】1/1 10/60 loss:97.20169067382812
【train】1/1 20/60 loss:79.12483978271484
【train】1/1 30/60 loss:46.32414245605469
【train】1/1 40/60 loss:45.012237548828125
【train】1/1 50/60 loss:58.53745651245117
【train】1/1 60/60 loss:80.4083251953125
checkpoint: ./checkpoint/duie_controlled/pytorch_model_ner.bin
```

训练过程完成了真实的前向传播、CRF loss 计算、反向传播、参数更新和 checkpoint 保存。

## 权重保存验证

生成文件：

| 文件 | 大小 |
| --- | ---: |
| `checkpoint/duie_controlled/ner_args.json` | 3938 bytes |
| `checkpoint/duie_controlled/pytorch_model_ner.bin` | 412898275 bytes |

权重读取验证：

| 项目 | 值 |
| --- | ---: |
| state dict keys | 213 |

## 验证集结果

由于仅使用 120 条训练样本进行 CPU 快速训练，验证指标较低，但已出现少量实体命中，说明训练和解码链路有效。

```text
              precision    recall  f1-score   support

          人物       0.08      0.05      0.06        41

   micro avg       0.08      0.02      0.03        91
   macro avg       0.01      0.00      0.00        91
weighted avg       0.04      0.02      0.03        91
```

## 结论

本步骤已经完成实验要求中的模型训练和权重保存：BERT-BiLSTM-CRF 模型能够在 WSL CPU 环境下正常训练，并成功保存 `pytorch_model_ner.bin`。如果需要得到更高质量的预测结果，建议后续使用 GitHub + Colab T4 GPU 对全量训练集进行训练。
