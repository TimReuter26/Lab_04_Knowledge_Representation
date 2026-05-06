# -*- coding: utf-8 -*-
"""使用训练好的 BERT-BiLSTM-CRF 权重预测实验手册中的 NER 例句。"""

import json
import os

import torch
from transformers import BertTokenizer

from config import NerConfig
from model import BertNer


EXAMPLES = [
    "《民航客运服务会话》是1995年中国民航出版社出版的图书，作者是周石田",
    "再有之后的《半生缘》，蒋勤勤饰演的顾曼璐完全把林心如的曼桢衬得像是涉世未深的小姑娘，毫无半点风情",
    "裴友生，男，汉族，湖北蕲春人，1957年12月出生，大专学历",
    "吴君如演的周吉是电影《花田喜事》，在周吉大婚之夜，其夫林嘉声逃走失踪，后来其夫新科状元高中回来，周吉急往城楼相识，但林嘉声却言夫妻情断，覆水难收",
]


def decode_bio(chars, labels):
    entities = []
    start = None
    ent_type = None

    def close_entity(end):
        if start is not None:
            entities.append(
                {
                    "text": "".join(chars[start:end]),
                    "type": ent_type,
                    "start": start,
                    "end": end,
                }
            )

    for idx, label in enumerate(labels):
        if label == "O":
            close_entity(idx)
            start = None
            ent_type = None
            continue

        prefix, _, current_type = label.partition("-")
        if prefix == "B" or current_type != ent_type:
            close_entity(idx)
            start = idx
            ent_type = current_type
        elif prefix == "I" and start is None:
            start = idx
            ent_type = current_type

    close_entity(len(labels))
    return entities


def predict_sentence(sentence, model, tokenizer, args, device):
    chars = list(sentence)
    max_chars = args.max_seq_len - 2
    if len(chars) > max_chars:
        chars = chars[:max_chars]

    input_ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + chars + ["[SEP]"])
    attention_mask = [1] * len(input_ids)
    pad_len = args.max_seq_len - len(input_ids)
    input_ids = input_ids + [0] * pad_len
    attention_mask = attention_mask + [0] * pad_len

    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    length = int(attention_mask[0].sum().item())
    pred_ids = output.logits[0][1 : length - 1]
    pred_labels = [args.id2label[idx] for idx in pred_ids]
    return {
        "sentence": sentence,
        "labels": pred_labels,
        "entities": decode_bio(chars, pred_labels),
    }


def main():
    args = NerConfig("duie")
    checkpoint_dir = os.path.join("checkpoint", "duie_colab")
    checkpoint_path = os.path.join(checkpoint_dir, "pytorch_model_ner.bin")
    args.output_dir = checkpoint_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    model = BertNer(args)
    state_dict = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("missing_keys:", missing)
    if unexpected:
        print("unexpected_keys:", unexpected)
    model.to(device)
    model.eval()

    results = [predict_sentence(sentence, model, tokenizer, args, device) for sentence in EXAMPLES]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
