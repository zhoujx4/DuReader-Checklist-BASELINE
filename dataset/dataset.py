"""
@Time : 2021/4/98:43
@Auth : 周俊贤
@File ：dataset.py
@DESCRIPTION:

"""

import json
import torch
import pandas as pd

from transformers import BertTokenizerFast
from torch.utils.data import Dataset

tokenizer = BertTokenizerFast.from_pretrained("/data/zhoujx/prev_trained_model/rbt3")


class MrcDataset(Dataset):
    def __init__(self, args, json_path, tokenizer):
        examples = []

        with open(json_path, "r", encoding="utf8") as f:
            input_data = json.load(f)["data"]

        for entry in input_data:
            #     title = entry.get("title", "").strip()
            for paragraph in entry["paragraphs"]:
                context = paragraph["context"].strip()
                title = paragraph["title"].strip()
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question = qa["question"].strip()
                    answer_starts = []
                    answers = []
                    is_impossible = False

                    if "is_impossible" in qa.keys():
                        is_impossible = qa["is_impossible"]

                    answer_starts = [answer["answer_start"] for answer in qa.get("answers", [])]
                    answers = [answer["text"].strip() for answer in qa.get("answers", [])]

                    examples.append({
                        "id": qas_id,
                        "title": title,
                        "context": context,
                        "question": question,
                        "answers": answers,
                        "answer_starts": answer_starts,
                        "is_impossible": is_impossible
                    })

        # examples = examples[:10]

        # questions = [examples[i]['question'] for i in range(len(examples))]
        questions_title = [examples[i]['question']  + examples[i]['title'] for i in range(len(examples))]
        # title_contexts = [examples[i]['title'] + examples[i]['context'] for i in range(len(examples))]
        contexts = [examples[i]['context'] for i in range(len(examples))]
        tokenized_examples = tokenizer(questions_title,
                                       contexts,
                                       padding="max_length",
                                       max_length=args.max_len,
                                       truncation="only_second",
                                       stride=args.stride,
                                       return_offsets_mapping=True,
                                       return_overflowing_tokens=True)

        df_tmp = pd.DataFrame.from_dict(tokenized_examples, orient="index").T
        tokenized_examples = df_tmp.to_dict(orient="records")

        for i, tokenized_example in enumerate(tokenized_examples):
            input_ids = tokenized_example["input_ids"]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            offsets = tokenized_example['offset_mapping']
            sequence_ids = tokenized_example['token_type_ids']

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = tokenized_example['overflow_to_sample_mapping']
            answers = examples[sample_index]['answers']
            answer_starts = examples[sample_index]['answer_starts']

            # If no answers are given, set the cls_index as answer.
            if len(answer_starts) == 0 or (answer_starts[0] == -1):
                tokenized_examples[i]["start_positions"] = cls_index
                tokenized_examples[i]["end_positions"] = cls_index
                tokenized_examples[i]['answerable_label'] = 0
            else:
                # Start/end character index of the answer in the text.
                start_char = answer_starts[0]
                end_char = start_char + len(answers[0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 2
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1
                token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and
                        offsets[token_end_index][1] >= end_char):
                    tokenized_examples[i]["start_positions"] = cls_index
                    tokenized_examples[i]["end_positions"] = cls_index
                    tokenized_examples[i]['answerable_label'] = 0
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples[i]["start_positions"] = token_start_index - 1
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples[i]["end_positions"] = token_end_index + 1
                    tokenized_examples[i]['answerable_label'] = 1

            # evaluate的时候有用
            tokenized_examples[i]["example_id"] = examples[sample_index]['id']
            tokenized_examples[i]["offset_mapping"] = [
                (o if sequence_ids[k] == 1 else None)
                for k, o in enumerate(tokenized_example["offset_mapping"])
            ]

        self.examples = examples
        self.tokenized_examples = tokenized_examples

    def __len__(self):
        return len(self.tokenized_examples)

    def __getitem__(self, index):
        return self.tokenized_examples[index]


def collate_fn(batch):
    max_len = max([sum(x['attention_mask']) for x in batch])
    all_input_ids = torch.tensor([x['input_ids'][:max_len] for x in batch])
    all_token_type_ids = torch.tensor([x['token_type_ids'][:max_len] for x in batch])
    all_attention_mask = torch.tensor([x['attention_mask'][:max_len] for x in batch])
    all_answerable_label = torch.tensor([x["answerable_label"] for x in batch])
    all_start_positions = torch.tensor([x["start_positions"] for x in batch])
    all_end_positions = torch.tensor([x["end_positions"] for x in batch])

    return {
        "all_input_ids": all_input_ids,
        "all_token_type_ids": all_token_type_ids,
        "all_attention_mask": all_attention_mask,
        "all_start_positions": all_start_positions,
        "all_end_positions": all_end_positions,
        "all_answerable_label": all_answerable_label,
    }


if __name__ == '__main__':
    dataset = MrcDataset("../data/train.json")
    a = 1
