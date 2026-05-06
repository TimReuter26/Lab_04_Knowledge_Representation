# -*- coding: utf-8 -*-
"""LTP 命名实体识别实验示例。

运行方式：
    /home/mmama/miniconda3/envs/mnist_exp/bin/python ltp_ner_demo.py
"""

import json

from ltp import LTP


def main():
    ltp = LTP("./ltp_small")
    sentences = [
        "张伟在北京大学参加了人工智能研讨会，随后前往上海参观华为公司。",
        "2025年5月，李娜受邀到浙江大学和阿里巴巴集团交流自然语言处理技术。",
        "《红楼梦》作者曹雪芹出生于南京，人民文学出版社曾多次出版该书。",
    ]

    result = ltp.pipeline(sentences, tasks=["cws", "ner"])
    records = []
    for sentence, words, entities in zip(sentences, result.cws, result.ner):
        records.append(
            {
                "sentence": sentence,
                "words": words,
                "entities": [
                    {
                        "type": entity_type,
                        "text": entity_text,
                        "start_word_index": start,
                        "end_word_index": end,
                    }
                    for entity_type, entity_text, start, end in entities
                ],
            }
        )

    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
