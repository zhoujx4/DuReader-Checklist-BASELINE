"""
@Time : 2020/12/1110:44
@Auth : 周俊贤
@File ：run.py
@DESCRIPTION:

"""
import json
import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from dataset.dataset import MrcDataset
from dataset.dataset import collate_fn
from models.model import MRC_model
from run import evaluate
from utils.finetuning_argparse import get_argparse
from utils.utils import seed_everything, init_logger, logger


def main():
    parser = get_argparse()
    parser.add_argument("--fine_tunning_model",
                        type=str,
                        required=True,
                        help="fine_tuning model path")
    args = parser.parse_args()
    print(json.dumps(vars(args), sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))
    init_logger(log_file="./log/{}.log".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    seed_everything(args.seed)

    # save path
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # device
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path)

    # Dataset & Dataloader
    test_dataset = MrcDataset(args,
                              json_path="./data/test1.json",
                              tokenizer=tokenizer)

    test_iter = DataLoader(test_dataset,
                           shuffle=False,
                           batch_size=args.per_gpu_eval_batch_size,
                           collate_fn=collate_fn,
                           num_workers=24)

    logger.info("The nums of the test_dataset examples is {}".format(len(test_dataset.examples)))
    logger.info("The nums of the test_dataset features is {}".format(len(test_dataset)))


    # model
    model = MRC_model(args.model_name_or_path)
    model.to(args.device)
    model.load_state_dict(torch.load(args.fine_tunning_model))

    # predict test
    model.eval()
    evaluate(args, test_iter, model, prefix="test")

if __name__ == "__main__":
    main()
