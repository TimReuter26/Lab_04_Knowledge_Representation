# -*- coding: utf-8 -*-
"""CPU 可控规模 BERT-BiLSTM-CRF NER 训练脚本。

该脚本用于在无 GPU 的 WSL 环境中完成一次可复现的训练成功验证：
- 使用 hfl/chinese-bert-wwm-ext 本地预训练模型；
- 使用 DuIE BIO 数据的一个小子集；
- 保存训练权重到 checkpoint/duie_controlled/pytorch_model_ner.bin；
- 输出训练 loss 和验证集 seqeval 报告。
"""

import json
import os
import random

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from config import NerConfig
from data_loader import NerDataset
from main_todo import Trainer, build_optimizer_and_scheduler
from model import BertNer


def load_jsonl(path, limit):
    rows = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def main():
    random.seed(42)
    torch.manual_seed(42)

    args = NerConfig("duie")
    args.output_dir = os.path.join("./checkpoint", "duie_controlled")
    os.makedirs(args.output_dir, exist_ok=True)

    # CPU 环境下控制训练规模，实验手册允许根据硬件配置调整训练集大小。
    args.epochs = 1
    args.train_batch_size = 2
    args.dev_batch_size = 2
    args.save_step = 4

    train_data = load_jsonl(os.path.join(args.data_path, "train.txt"), limit=120)
    dev_data = load_jsonl(os.path.join(args.data_path, "dev.txt"), limit=24)

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    train_dataset = NerDataset(train_data, args, tokenizer)
    dev_dataset = NerDataset(dev_data, args, tokenizer)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=0)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.dev_batch_size, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"train_samples: {len(train_dataset)}")
    print(f"dev_samples: {len(dev_dataset)}")
    print(f"max_seq_len: {args.max_seq_len}")
    print(f"num_labels: {args.num_labels}")

    model = BertNer(args)
    model.to(device)

    total_steps = len(train_loader) * args.epochs
    optimizer, scheduler = build_optimizer_and_scheduler(args, model, total_steps)
    trainer = Trainer(
        output_dir=args.output_dir,
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=dev_loader,
        optimizer=optimizer,
        schedule=scheduler,
        epochs=args.epochs,
        save_step=args.save_step,
        device=device,
        id2label=args.id2label,
    )

    with open(os.path.join(args.output_dir, "ner_args.json"), "w", encoding="utf-8") as fp:
        json.dump(vars(args), fp, ensure_ascii=False, indent=2)

    trainer.train()
    print("checkpoint:", os.path.join(args.output_dir, "pytorch_model_ner.bin"))
    print("classification_report:")
    print(trainer.test())


if __name__ == "__main__":
    main()
