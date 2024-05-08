# coding: utf-8
# @Author    :陈梦淇
# @time      :2024/3/21
import os
import sys
import datetime
import argparse

import torch
from torch.utils.data import DataLoader

from train import GCNTrainer
from data_utils import PPepISDataForGCN, collate_fn_single, collate_fn

current_directory = os.path.abspath(os.path.dirname(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))

seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
date_time = datetime.datetime.now().strftime("%m-%d")


def blg_main(para_args):
    # 加载数据
    ppepis_train_data = PPepISDataForGCN("train")
    ppepis_train_dataloader = DataLoader(ppepis_train_data, batch_size=para_args.batch_size, shuffle=False,
    #                                      pin_memory=True, num_workers=5, collate_fn=collate_fn)

    # ppepis_train_data = PPepISDataForGCN("validation")
    # ppepis_train_dataloader = DataLoader(ppepis_train_data, batch_size=para_args.batch_size, shuffle=False,
                                         pin_memory=True, num_workers=5, collate_fn=collate_fn)

    ppepis_validation_data = PPepISDataForGCN("validation")
    ppepis_validation_dataloader = DataLoader(ppepis_validation_data, batch_size=para_args.batch_size, shuffle=False,
                                              pin_memory=True, num_workers=5, collate_fn=collate_fn_single)
    gcn_trainer = GCNTrainer(input_size=33,
                             learning_rate=para_args.learning_rate,
                             optimizer=para_args.optimizer,
                             epoch_nums=para_args.epoch,
                             description=para_args.description,
                             is_save=False,
                             alpha=0.25,
                             weight=2)
    gcn_trainer.train_and_validation_part(ppepis_train_dataloader, ppepis_validation_dataloader)
   


if __name__ == "__main__":
    print("运行日期为:{}".format(date_time))
    parser = argparse.ArgumentParser(description="显示程序搜索的超参数")
    parser.add_argument("--optimizer", type=str, help="优化器类型", default="RMSprop")
    parser.add_argument("--learning_rate", type=float, help="学习率", default=2e-3)
    parser.add_argument("--epoch", type=int, help="训练轮数", default=200)
    parser.add_argument("--features_num", type=int, help="使用特征的数量", default=32)
    parser.add_argument("--description", type=str, help="说明", default="")
    args = parser.parse_args()
    print(f"description: {args.description}")
    args.batch_size = 16
    blg_main(args)
