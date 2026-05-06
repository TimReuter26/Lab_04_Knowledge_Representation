# BERT-BiLSTM-CRF 训练代码补全与环境确认

## 补全文件

本步骤补全了 BERT 命名实体识别训练所需代码：

| 文件 | 补全内容 |
| --- | --- |
| `实验四_命名实体识别/data_loader_todo.py` | 补全 `input_ids` 和 `attention_mask` 到 `max_seq_len` |
| `实验四_命名实体识别/model_todo.py` | 获取 batch size，并使用线性层输出每个 token 的标签发射分数 |
| `实验四_命名实体识别/main_todo.py` | 补全梯度清零、反向传播、参数更新、学习率更新和 `NerDataset` 加载 |
| `实验四_命名实体识别/config.py` | 自动创建 checkpoint 输出目录，并按实验手册默认值设置 `max_seq_len=256`、`epochs=1` |

同时新增常规入口文件：

| 文件 | 说明 |
| --- | --- |
| `实验四_命名实体识别/data_loader.py` | 导出 `NerDataset` |
| `实验四_命名实体识别/model.py` | 导出 `BertNer` 和 `ModelOutput` |
| `实验四_命名实体识别/main.py` | 训练入口，调用 `main("duie")` |

## 模型训练代码逻辑

模型结构为 BERT + BiLSTM + CRF：

1. `BertModel` 读取输入 token 序列，生成上下文相关的 token 表示。
2. BiLSTM 在 BERT 表示基础上继续建模序列的双向上下文信息。
3. 线性层将 BiLSTM 输出映射到 BIO 标签空间，得到每个 token 对各标签的发射分数。
4. CRF 层建模标签之间的转移约束，训练时最大化真实标签序列概率，预测时使用解码得到最优标签序列。

训练循环逻辑：

1. 从 `DataLoader` 取出 batch，并移动到 CPU/GPU 设备。
2. 前向传播得到 CRF loss。
3. `optimizer.zero_grad()` 清空上一轮梯度。
4. `loss.backward()` 计算梯度。
5. `optimizer.step()` 更新模型参数。
6. `schedule.step()` 更新学习率。
7. 按 `save_step` 保存 checkpoint，训练结束后保存最终权重。

## WSL 环境确认

使用的 Python 环境：

`/home/mmama/miniconda3/envs/mnist_exp/bin/python`

依赖确认结果：

| 依赖 | 状态 |
| --- | --- |
| PyTorch | `2.11.0+cpu` |
| Transformers | `4.27.4` |
| pytorch-crf | `0.7.2` |
| seqeval | 可导入 |
| scikit-learn | 可导入 |
| scipy | 可导入 |

CUDA 状态：

`torch.cuda.is_available() == False`

说明：当前 WSL 环境只能使用 CPU 训练。正式训练如果耗时过长，可以改用 GitHub + Colab T4 GPU 方案。

## 冒烟测试结果

语法编译检查通过：

```bash
/home/mmama/miniconda3/envs/mnist_exp/bin/python -m py_compile config.py data_loader_todo.py model_todo.py main_todo.py data_loader.py model.py main.py
```

配置实例化结果：

| 配置项 | 值 |
| --- | --- |
| `data_path` | `./data/duie/ner_data` |
| `num_labels` | 49 |
| `max_seq_len` | 256 |
| `epochs` | 1 |
| `train_batch_size` | 12 |

DataLoader 样本测试结果：

| 张量 | 形状 |
| --- | --- |
| `input_ids` | `(256,)` |
| `attention_mask` | `(256,)` |
| `labels` | `(256,)` |

当前仍需在正式训练前准备 BERT 预训练模型目录：

`实验四_命名实体识别/model_hub/chinese-bert-wwm-ext/`

该目录目前不存在，下一步训练前需要下载或放置 `hfl/chinese-bert-wwm-ext` 的模型文件。
