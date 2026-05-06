# DuIE 数据 BIO 转换结果

## 处理目标

根据实验手册要求，将 `duie_data/train.json` 和 `duie_data/dev.json` 中的实体标注转换为 NER 常用的 BIO 序列标注格式，并生成实体类别标签文件。

## 补全代码

已补全脚本：

`实验四_命名实体识别/process_todo.py`

主要补全部分：

- 定义原始数据路径：`duie_data/train.json`、`duie_data/dev.json`、`duie_data/test.json`、`duie_data/duie_schema.json`
- 定义输出路径：`data/duie/ner_data/` 和 `data/duie/re_data/`
- 从 schema 中读取 `subject_type` 和 `object_type["@value"]`
- 使用 `re.finditer(re.escape(...), text)` 在原文中定位 subject 和 object
- 将实体首字符标注为 `B-实体类型`，后续字符标注为 `I-实体类型`
- 生成 `labels.txt`、`train.txt`、`dev.txt`

## 生成文件

| 文件 | 说明 |
| --- | --- |
| `实验四_命名实体识别/data/duie/ner_data/labels.txt` | NER 实体类别标签 |
| `实验四_命名实体识别/data/duie/ner_data/train.txt` | 训练集 BIO 数据 |
| `实验四_命名实体识别/data/duie/ner_data/dev.txt` | 验证集 BIO 数据 |
| `实验四_命名实体识别/data/duie/re_data/rels.txt` | 关系类型辅助文件 |

## 校验结果

运行命令：

```bash
cd /home/mmama/workspace/AI_Projects/Lab_04_content/实验四_命名实体识别
/home/mmama/miniconda3/envs/mnist_exp/bin/python process_todo.py
```

数据转换结果：

| 数据 | 记录数 | 长度不匹配记录数 | 非 O 实体标签数 |
| --- | ---: | ---: | ---: |
| train.txt | 10000 | 0 | 127512 |
| dev.txt | 1000 | 0 | 12332 |

实体类别数量：24 类。

说明：所有样本均满足 `len(text) == len(labels)`，即每个字符都有对应 BIO 标签，满足后续 BERT 序列标注训练的数据格式要求。

## 样例

原文：

`《邪少兵王》是冰火未央写的网络小说连载于旗峰天下`

部分 BIO 结果：

| 字符 | 标签 |
| --- | --- |
| 邪 | B-图书作品 |
| 少 | I-图书作品 |
| 兵 | I-图书作品 |
| 王 | I-图书作品 |
| 冰 | B-人物 |
| 火 | I-人物 |
| 未 | I-人物 |
| 央 | I-人物 |

该样例说明，脚本能够从 DuIE 的 SPO 标注中抽取 subject 和 object，并转换为字符级 BIO 标注。
