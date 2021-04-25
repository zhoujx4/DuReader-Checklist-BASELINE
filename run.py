"""
@Time : 2020/12/1110:44
@Auth : 周俊贤
@File ：run.py
@DESCRIPTION:

"""
import copy
import json
import os
import time
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from rouge import Rouge
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import BertTokenizerFast

from dataset.dataset import MrcDataset
from dataset.dataset import collate_fn
from metric.metric import compute_prediction_checklist
from models.model import MRC_model
from utils.adversarial import FGM
from utils.finetuning_argparse import get_argparse
from utils.utils import seed_everything, ProgressBar, init_logger, logger

class CrossEntropyLossForChecklist(nn.Module):
    def __init__(self):
        super(CrossEntropyLossForChecklist, self).__init__()

    def forward(self, y, label):
        start_logits, end_logits, cls_logits = y
        start_position, end_position, answerable_label = label
        #
        start_loss = F.cross_entropy(input=start_logits, target=start_position)
        end_loss = F.cross_entropy(input=end_logits, target=end_position)
        #
        cls_loss = F.cross_entropy(input=cls_logits, target=answerable_label)
        #
        mrc_loss = (start_loss + end_loss) / 2
        loss = (mrc_loss + cls_loss) / 2

        return loss, mrc_loss, cls_loss

def train(args, train_iter, model):
    logger.info("***** Running train *****")
    # 优化器
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    linear_param_optimizer.extend(list(model.classifier_cls.named_parameters()))
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': args.learning_rate},
        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         'lr': args.linear_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': args.linear_learning_rate},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)
    # 损失函数
    criterion = CrossEntropyLossForChecklist().to(args.device)
    batch_mrc_loss = 0
    batch_cls_loss = 0
    pbar = ProgressBar(n_total=len(train_iter), desc='Training')
    print("****" * 20)
    fgm = FGM(model, epsilon=1, emb_name='word_embeddings.weight')
    for step, batch in enumerate(train_iter):
        for key in batch.keys():
            batch[key] = batch[key].to(args.device)
        logits = model(
            input_ids=batch['all_input_ids'],
            attention_mask=batch['all_attention_mask'],
            token_type_ids=batch['all_token_type_ids']
        )
        # 正常训练
        loss, mrc_loss, cls_loss = criterion(logits, (
            batch["all_start_positions"],
            batch["all_end_positions"],
            batch["all_answerable_label"]
        ))
        loss.backward()
        # 对抗训练
        fgm.attack()  # 在embedding上添加对抗扰动
        logits_adv = model(
            input_ids=batch['all_input_ids'],
            attention_mask=batch['all_attention_mask'],
            token_type_ids=batch['all_token_type_ids'])
        loss_adv, mrc_loss, cls_loss = criterion(logits_adv, (
            batch["all_start_positions"],
            batch["all_end_positions"],
            batch["all_answerable_label"]
        ))
        loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        fgm.restore()  # 恢复embedding参数
        #
        batch_mrc_loss += mrc_loss.item()
        batch_cls_loss += cls_loss.item()
        pbar(step,
             {
                 'batch_mrc_loss': batch_mrc_loss / (step + 1),
                 'batch_cls_loss': batch_cls_loss / (step + 1)
             })
        optimizer.step()
        model.zero_grad()

def evaluate(args, eval_iter, model, prefix):
    logger.info("***** Running Evalation *****")

    all_start_logits = []
    all_end_logits = []
    all_cls_logits = []
    batch_mrc_loss = 0
    batch_cls_loss = 0

    pbar = ProgressBar(n_total=len(eval_iter), desc="Evaluating")
    model.eval()
    criterion = CrossEntropyLossForChecklist().to(args.device)
    with torch.no_grad():
        for step, batch in enumerate(eval_iter):
            for key in batch.keys():
                batch[key] = batch[key].to(args.device)
            start_logits_tensor, end_logits_tensor, cls_logits_tensor = model(
                input_ids=batch['all_input_ids'],
                attention_mask=batch['all_attention_mask'],
                token_type_ids=batch['all_token_type_ids']
            )
            ###########
            loss, mrc_loss, cls_loss = criterion((start_logits_tensor, end_logits_tensor, cls_logits_tensor), (
                batch["all_start_positions"],
                batch["all_end_positions"],
                batch["all_answerable_label"]
            ))
            #########
            for idx in range(start_logits_tensor.shape[0]):
                all_start_logits.append(start_logits_tensor.cpu().numpy()[idx])
                all_end_logits.append(end_logits_tensor.cpu().numpy()[idx])
                all_cls_logits.append(cls_logits_tensor.cpu().numpy()[idx])
            pbar(step)

    all_predictions, all_nbest_json, all_cls_predictions = compute_prediction_checklist(
        eval_iter.dataset.examples,
        eval_iter.dataset.tokenized_examples,
        (all_start_logits, all_end_logits, all_cls_logits),
        True,
        args.n_best_size,
        args.max_answer_length,
        args.cls_threshold
    )

    with open(os.path.join(args.output_dir, prefix+'_predictions.json'), "w", encoding='utf-8') as writer:
        writer.write(
            json.dumps(
                all_predictions, ensure_ascii=False, indent=4) + "\n")

    with open(os.path.join(args.output_dir, prefix+'_nbest_predictions.json'), "w", encoding="utf8") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + u"\n")

    if prefix == "eval":
        df = pd.DataFrame(eval_iter.dataset.examples)
        df = df.drop_duplicates(subset=["id"])
        df["answers"] = df["answers"].apply(lambda x: x[0] if len(x[0]) != 0 else "no answer")
        df_pre_answers = pd.DataFrame.from_dict(all_predictions, orient="index", columns=["answers_pre"])
        df_pre_cls = pd.DataFrame(all_cls_predictions, columns=["id", "is_impossible_pre", "pre_0", "pre_1"])
        df_pre_cls = df_pre_cls.drop_duplicates(subset=["id"])
        #
        df = df.merge(df_pre_answers, how="left", left_on="id", right_index=True)
        df = df.merge(df_pre_cls, how="left", on="id")
        df = df.set_index("id")
        predictions_details = df.to_dict("index")
        rouge = Rouge()

        with open(os.path.join(args.output_dir, prefix+'_predictions_details.json'), "w", encoding='utf-8') as writer:
            writer.write(
                json.dumps(
                    predictions_details, ensure_ascii=False, indent=4) + "\n")

        EM = accuracy_score(df["answers"], df["answers_pre"])
        # df_f1 = df[df["is_impossible"] == False].copy()
        df_f1 = df.copy()
        df_f1["answers"] = df_f1["answers"].apply(lambda x: " ".join(list(x)))
        df_f1["answers_pre"] = df_f1["answers_pre"].apply(lambda x: " ".join(list(x)))
        F1_score = rouge.get_scores(df_f1["answers_pre"], df_f1["answers"], avg=True)["rouge-1"]["f"]

        return F1_score, EM

def main():
    args = get_argparse().parse_args()
    print(json.dumps(vars(args), sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))
    init_logger(log_file="./log/{}.log".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    seed_everything(args.seed)

    # 设置保存目录
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # device
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path)

    # Dataset & Dataloader
    train_dataset = MrcDataset(args,
                               json_path="./data/train.json",
                               tokenizer=tokenizer)
    eval_dataset = MrcDataset(args,
                              json_path="./data/dev.json",
                              tokenizer=tokenizer)
    # eval_dataset, test_dataset = random_split(eval_dataset,
    #                                           [round(0.5 * len(eval_dataset)),
    #                                            len(eval_dataset) - round(0.5 * len(eval_dataset))],
    #                                           generator=torch.Generator().manual_seed(42))
    train_iter = DataLoader(train_dataset,
                            shuffle=True,
                            batch_size=args.per_gpu_train_batch_size,
                            collate_fn=collate_fn,
                            num_workers=10)
    eval_iter = DataLoader(eval_dataset,
                           shuffle=False,
                           batch_size=args.per_gpu_eval_batch_size,
                           collate_fn=collate_fn,
                           num_workers=10)
    # test_iter = DataLoader(test_dataset,
    #                        shuffle=False,
    #                        batch_size=args.per_gpu_eval_batch_size,
    #                        collate_fn=collate_fn,
    #                        num_workers=10)
    logger.info("The nums of the train_dataset examples is {}".format(len(train_dataset.examples)))
    logger.info("The nums of the train_dataset features is {}".format(len(train_dataset)))
    logger.info("The nums of the eval_dataset examples is {}".format(len(eval_dataset.examples)))
    logger.info("The nums of the eval_dataset features is {}".format(len(eval_dataset)))

    # model
    model = MRC_model(args.model_name_or_path)
    model.to(args.device)

    # 训练
    best_f1 = 0
    early_stop = 0
    for epoch, _ in enumerate(range(int(args.num_train_epochs))):
        model.train()
        train(args, train_iter, model)
        # 每轮epoch在验证集上计算分数
        eval_f1, eval_EM = evaluate(args, eval_iter, model, prefix="eval")
        logger.info(
            "The F1-score is {}, The EM-score is {}".format(eval_f1, eval_EM)
        )
        if eval_f1 > best_f1:
            early_stop = 0
            best_f1 = eval_f1
            logger.info("the best eval f1 is {:.4f}, saving model !!".format(best_f1))
            best_model = copy.deepcopy(model.module if hasattr(model, "module") else model)
            torch.save(best_model.state_dict(), os.path.join(args.output_dir, "best_model.pkl"))
        else:
            early_stop += 1
            if early_stop == args.early_stop:
                logger.info("Early stop in {} epoch!".format(epoch))
                break
    # test_f1 = evaluate(args, test_iter, best_model)
    # logger.info("Test F1 is {:.4f}!".format(test_f1))

if __name__ == "__main__":
    main()
