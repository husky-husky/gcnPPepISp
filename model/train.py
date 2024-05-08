# coding: utf-8
# @Author    :陈梦淇
# @time      :2024/3/19
import os
import sys
import time
import datetime

import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from Graph import deepGCN
from utils import LRScheduler, EarlyStopping, get_logger, WeightedFocalLoss
from metrics import metrics_classification

current_directory = os.path.abspath(os.path.dirname(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))

seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

DATE_TIME = datetime.datetime.now().strftime("%m-%d")


def get_label_from_mask(labels, attention_masks):
    """

    :param labels:
    :param attention_masks:
    :return: tensor(N,)
    """
    y_true = []
    for seq_num in range(len(labels)):
        seq_len = (attention_masks[seq_num] == 1).sum()
        label = labels[seq_num][0:seq_len]
        y_true.append(label)
    y_true = torch.cat(y_true, 0).view(-1, )
    return y_true


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    elapsed_rounded = int(round(elapsed))

    return str(datetime.timedelta(seconds=elapsed_rounded))


def get_label_from_mask(labels, mask):
    """

    :param labels:
    :param attention_masks:
    :return: tensor(N,)
    """
    y_true = []
    for seq_num in range(len(labels)):
        seq_len = (mask[seq_num] == 1).sum()
        label = labels[seq_num][0:seq_len]
        y_true.append(label)
    y_true = torch.cat(y_true, 0).view(-1, )
    return y_true


class GCNTrainer(nn.Module):
    def __init__(self, input_size,
                 learning_rate,
                 optimizer,
                 epoch_nums,
                 early_stop=True, description="", is_save=False, alpha=0.25, gamma=2, weight=2.0):
        super(GCNTrainer, self).__init__()
        self.input_size = input_size
        self.epoch_nums = epoch_nums
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer
        self.early_stop = early_stop
        self.description = description
        self.is_save = is_save
        self.gamma = gamma
        self.weight = weight

        # 根据日期创建目录用于保存日志文件和csv结果文件
        if not os.path.exists(f"{DATE_TIME}"):
            os.mkdir("{}".format(DATE_TIME))

        self.csv_result = pd.DataFrame(
            columns=["name", "epoch", "precision", "recall", "accuracy", "F1", "auc", "auprc",
                     "mcc", "ppv", "npv", "tpr", "tnr"])

        # cuda
        self.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        # initialize model
        self.model = deepGCN(nlayers=4,
                             nfeat=33,
                             nhidden=256,
                             nclass=1,
                             dropout=0.1,
                             lamda=1.5,
                             alpha=0.7,
                             variant=False)


        self.optimizer = getattr(optim, optimizer)(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.scheduler = CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=100,
            last_epoch=-1,
        )

        # 损失函数
        self.loss_fn = WeightedFocalLoss(weight=torch.tensor(weight).to(self.device),
                                         alpha=alpha, gamma=gamma).to(self.device)

        self.log_name = f"{DATE_TIME}/deepGCN_{DATE_TIME}_bce_weight={weight}_gamma={gamma}_{description}.log"
        self.logger = get_logger(self.log_name)
        self.logger.info(
            f"deepGCN，gamma:{self.gamma}")

        self.model = self.model.to(self.device)
        self.early_stop = True
        self.early_stopping = EarlyStopping()

    def train_part(self, train_dataloader):
        total_loss_train = 0

        t0 = time.time()
        self.model.train()
        for step, (peptide_bio_features, peptide_attention_mat, peptide_labels,
                   protein_bio_features, protein_attention_mat, protein_labels) \
                in enumerate(train_dataloader):

            if step % 200 == 0 and step != 0:
                elapsed = format_time(time.time() - t0)
                self.logger.info(
                    '  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            peptide_bio_features, peptide_attention_mat, peptide_labels = \
                peptide_bio_features.to(self.device), peptide_attention_mat.to(self.device), \
                peptide_labels.to(self.device)

            protein_bio_features, protein_attention_mat, protein_labels = \
                protein_bio_features.to(self.device), protein_attention_mat.to(self.device), \
                protein_labels.to(self.device)

            self.optimizer.zero_grad()
            peptide_y_pre = self.model(peptide_bio_features, peptide_attention_mat)
            peptide_loss = self.loss_fn(peptide_y_pre.squeeze(-1).to(torch.float), peptide_labels.to(torch.float))

            protein_y_pre = self.model(protein_bio_features, protein_attention_mat)
            protein_loss = self.loss_fn(protein_y_pre.squeeze(-1).to(torch.float), protein_labels.to(torch.float))

            loss = peptide_loss + protein_loss
            loss.backward()
            self.optimizer.step()

            total_loss_train = total_loss_train + loss.cpu().detach().numpy()

        training_time = format_time(time.time() - t0)
        self.logger.info("")
        self.logger.info("  Average training loss: {0:.6f}".format(total_loss_train / len(train_dataloader)))
        self.logger.info("  Training epcoh took: {:}".format(training_time))

        return total_loss_train / len(train_dataloader)

    def train_part_mask(self, train_dataloader):
        total_loss_train = 0
        t0 = time.time()
        self.model.train()
        for step, (peptide_bio_features, peptide_attention_mat, peptide_labels, peptide_mask,
                   protein_bio_features, protein_attention_mat, protein_labels, protein_mask) \
                in enumerate(train_dataloader):

            if step % 20 == 0 and step != 0:
                elapsed = format_time(time.time() - t0)
                self.logger.info(
                    '  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            peptide_bio_features, peptide_attention_mat, peptide_labels, peptide_mask = \
                peptide_bio_features.to(self.device), peptide_attention_mat.to(self.device), \
                peptide_labels.to(self.device), peptide_mask.to(self.device)

            protein_bio_features, protein_attention_mat, protein_labels, protein_mask = \
                protein_bio_features.to(self.device), protein_attention_mat.to(self.device), \
                protein_labels.to(self.device), protein_mask.to(self.device)

            self.optimizer.zero_grad()
            peptide_y_true = get_label_from_mask(peptide_labels, peptide_mask)
            peptide_y_pre = self.model(peptide_bio_features, peptide_attention_mat, peptide_mask) # 图卷积
            # peptide_y_pre = self.model(peptide_bio_features, peptide_mask)  # lstm
            peptide_loss = self.loss_fn(peptide_y_pre.squeeze(-1).to(torch.float), peptide_y_true.to(torch.float))

            protein_y_true = get_label_from_mask(protein_labels, protein_mask)
            protein_y_pre = self.model(protein_bio_features, protein_attention_mat, protein_mask)# 图卷积
            # protein_y_pre = self.model(protein_bio_features, protein_mask)  # lstm
            protein_loss = self.loss_fn(protein_y_pre.squeeze(-1).to(torch.float), protein_y_true.to(torch.float))

            loss = peptide_loss + protein_loss
            loss.backward()
            self.optimizer.step()

            total_loss_train = total_loss_train + loss.cpu().detach().numpy()

        training_time = format_time(time.time() - t0)
        self.logger.info("")
        self.logger.info("  Average training loss: {0:.6f}".format(total_loss_train / len(train_dataloader)))
        self.logger.info("  Training epcoh took: {:}".format(training_time))

        return total_loss_train / len(train_dataloader)

    def validation_epoch(self, validation_dataloader, epoch):
        total_loss_validation = 0

        peptide_metrics_df = pd.DataFrame(columns=["precision", "recall", "mcc", "auroc", "auprc", "accuracy"])
        protein_metrics_df = pd.DataFrame(columns=["precision", "recall", "mcc", "auroc", "auprc", "accuracy"])

        t0 = time.time()
        self.model.eval()
        for step, (peptide_bio_features, peptide_attention_mat, peptide_labels,
                   protein_bio_features, protein_attention_mat, protein_labels) \
                in enumerate(validation_dataloader):
            peptide_bio_features, peptide_attention_mat, peptide_labels = \
                peptide_bio_features.to(self.device), peptide_attention_mat.to(self.device), \
                peptide_labels.to(self.device)

            protein_bio_features, protein_attention_mat, protein_labels = \
                protein_bio_features.to(self.device), protein_attention_mat.to(self.device), \
                protein_labels.to(self.device)

            peptide_y_pre_score = self.model.predict(peptide_bio_features, peptide_attention_mat) # 图卷积
            # peptide_y_pre_score = self.model.predict(peptide_bio_features)  # lstm
            peptide_y_pre_score = peptide_y_pre_score.squeeze(-1).to(torch.float)
            peptide_y_pre_score, peptide_labels = peptide_y_pre_score.squeeze(0), peptide_labels.squeeze(0)
            peptide_loss = self.loss_fn(peptide_y_pre_score, peptide_labels.to(torch.float))
            peptide_y_pre_cls = torch.where(peptide_y_pre_score >= 0.5, torch.tensor(1).to(self.device),
                                            torch.tensor(0).to(self.device))
            peptide_y_pre_score = peptide_y_pre_score.detach().cpu().numpy().tolist()
            peptide_y_pre_cls = peptide_y_pre_cls.detach().cpu().numpy().tolist()
            peptide_labels = peptide_labels.detach().cpu().numpy().tolist()
            peptide_metrics_df.loc[len(peptide_metrics_df)] = metrics_classification(peptide_y_pre_cls,
                                                                                     peptide_y_pre_score,
                                                                                     peptide_labels)

            protein_y_pre_score = self.model(protein_bio_features, protein_attention_mat)  # 图卷积
            # protein_y_pre_score = self.model(protein_bio_features)  # lstm
            protein_y_pre_score = protein_y_pre_score.squeeze(-1).to(torch.float)
            protein_y_pre_score, protein_labels = protein_y_pre_score.squeeze(0), protein_labels.squeeze(0)
            protein_loss = self.loss_fn(protein_y_pre_score.squeeze(-1).to(torch.float), protein_labels.to(torch.float))
            protein_y_pre_cls = torch.where(protein_y_pre_score >= 0.5, torch.tensor(1).to(self.device),
                                            torch.tensor(0).to(self.device))
            protein_y_pre_score = protein_y_pre_score.detach().cpu().numpy().tolist()
            protein_y_pre_cls = protein_y_pre_cls.detach().cpu().numpy().tolist()
            protein_labels = protein_labels.detach().cpu().numpy().tolist()
            protein_metrics_df.loc[len(protein_metrics_df)] = metrics_classification(protein_y_pre_cls,
                                                                                     protein_y_pre_score,
                                                                                     protein_labels)

            loss = peptide_loss + protein_loss
            total_loss_validation = total_loss_validation + loss.cpu().detach().numpy()

        training_time = format_time(time.time() - t0)
        self.logger.info("")
        self.logger.info(
            "  Average validation loss: {0:.6f}".format(total_loss_validation / len(validation_dataloader)))
        self.logger.info("  Training epcoh took: {:}".format(training_time))

        self.logger.info("Epoch:{},performance on peptide validation".format(epoch + 1))
        self.logger.info("precision:{:.4f}, recall:{:.4f}, mcc:{:.4f}, auc:{:.4f}, auprc:{:.4f}, accuracy:{:.4f}"
                         .format(peptide_metrics_df["precision"].mean(), peptide_metrics_df["recall"].mean(),
                                 peptide_metrics_df["mcc"].mean(), peptide_metrics_df["auroc"].mean(),
                                 peptide_metrics_df["auprc"].mean(), peptide_metrics_df["accuracy"].mean()))

        self.logger.info("Epoch:{},performance on protein validation".format(epoch + 1))
        self.logger.info("precision:{:.4f}, recall:{:.4f}, mcc:{:.4f}, auc:{:.4f}, auprc:{:.4f}, accuracy:{:.4f}"
                         .format(protein_metrics_df["precision"].mean(), protein_metrics_df["recall"].mean(),
                                 protein_metrics_df["mcc"].mean(), protein_metrics_df["auroc"].mean(),
                                 protein_metrics_df["auprc"].mean(), protein_metrics_df["accuracy"].mean()))

        return total_loss_validation / len(validation_dataloader)

    def train_and_validation_part(self, train_dataloader, validation_dataloader):
        train_acc, train_loss, best_val_loss = 0, 0, 100
        best_auprc = 0
        t0 = time.time()
        self.logger.info(f"==================description:{self.description}====================")
        self.logger.info("========learning_rate:{:}=========".format(self.learning_rate))
        for epoch in range(self.epoch_nums):
            torch.cuda.empty_cache()
            lr = self.scheduler.get_last_lr()[0]
            self.logger.info("")
            self.logger.info('======== Epoch {:} / {:}, lr:{:} ========'.format(epoch + 1, self.epoch_nums, lr))
            self.logger.info('Training...')

            train_loss = self.train_part_mask(train_dataloader)
            val_loss = self.validation_epoch(validation_dataloader, epoch)
            if self.is_save:
                self.save_model(epoch + 1)

            self.scheduler.step()
            if self.early_stop:
                self.early_stopping(val_loss)
                if self.early_stopping.early_stop:
                    break

        self.logger.info("Training complete!")
        self.logger.info("Total training took {:} (h:mm:ss)".format(format_time(time.time() - t0)))

        self.logger.info("==========train loss: {}==========".format(train_loss))
        self.logger.info("==========validation loss: {}==========".format(best_val_loss))

        # self.save_model(self.epoch_nums)
        self.csv_result.to_csv(f"{self.log_name}.csv")
        return best_auprc

    def save_model(self, epoch):
        if not os.path.exists("{}_model".format(DATE_TIME)):
            os.mkdir("{}_model".format(DATE_TIME))
        torch.save(self.model.state_dict(),
                   "{}_model/deepGCN_{}_{}_{}_{}_w={}.pth".format(DATE_TIME, DATE_TIME,
                                                                  self.optimizer_type,
                                                                  epoch,
                                                                  self.gamma,
                                                                  self.description,
                                                                  self.weight))
