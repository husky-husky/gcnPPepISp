# coding: utf-8
# @Author    :陈梦淇
# @time      :2024/3/19
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset

current_directory = os.path.abspath(os.path.dirname(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))


class PPepISDataForGCN(Dataset):
    def __init__(self, file_name):
        super(PPepISDataForGCN, self).__init__()
        self.file_name = file_name
        self.features = pd.read_pickle("features_all.pkl")

        self.pdb_id = pd.read_pickle(os.path.join(parent_directory, f"preprocess/{file_name}_id.pkl"))

        self.peptide_bio_features, self.peptide_attention_mats, self.peptide_labels, self.peptide_seq_len, \
        self.protein_bio_features, self.protein_attention_mats, self.protein_labels, self.protein_seq_len = self.get_fea()

    def __getitem__(self, item):
        output = {
            "peptide_bio_features": torch.tensor(self.peptide_bio_features[item]),
            "peptide_labels": torch.tensor(self.peptide_labels[item]),
            "peptide_seq_len": self.peptide_seq_len[item],
            "peptide_attention_mat": torch.tensor(self.peptide_attention_mats[item]),
            "protein_bio_features": torch.tensor(self.protein_bio_features[item]),
            "protein_labels": torch.tensor(self.protein_labels[item]),
            "protein_seq_len": self.protein_seq_len[item],
            "protein_attention_mat": torch.tensor(self.protein_attention_mats[item]),
            "fea_num": len(self.features)
        }
        return output

    def __len__(self):
        return len(self.peptide_bio_features)

    # def get_fea(self):
    #     peptide_bio_features, peptide_attention_mats, peptide_labels, peptide_seq_len = [], [], [], []
    #     protein_bio_features, protein_attention_mats, protein_labels, protein_seq_len = [], [], [], []
    #
    #     for index, row in tqdm(self.pdb_id.iterrows(), desc=f"loading {self.file_name}"):
    #         is_correct = 0
    #         peptide_id, protein_id = row["peptide_id"], row["protein_id"]
    #
    #         peptide_fea = self.peptide_bio_feas.loc[self.peptide_bio_feas.id == peptide_id]
    #         peptide_label = peptide_fea["label"].values.tolist()
    #         peptide_fea = peptide_fea.iloc[:, :-2]
    #         peptide_fea = peptide_fea[self.features].values.tolist()
    #         peptide_matrix = pd.read_pickle(os.path.join(self.mat_path, f"{peptide_id}.pkl"))
    #         # import pdb
    #         # pdb.set_trace()
    #         if len(peptide_label) == len(peptide_matrix) == len(peptide_fea):
    #             is_correct += 1
    #
    #         protein_fea = self.protein_bio_feas.loc[self.protein_bio_feas.id == protein_id]
    #         protein_label = protein_fea["label"].values.tolist()
    #         protein_fea = protein_fea.iloc[:, :-2]
    #         protein_fea = protein_fea[self.features].values.tolist()
    #         protein_matrix = pd.read_pickle(os.path.join(self.mat_path, f"{protein_id}.pkl"))
    #
    #         if len(protein_label) == len(protein_matrix) == len(protein_fea):
    #             is_correct += 1
    #         if is_correct == 2:
    #             peptide_labels.append(peptide_label)
    #             peptide_bio_features.append(peptide_fea)
    #             peptide_attention_mats.append(peptide_matrix)
    #             peptide_seq_len.append(len(peptide_fea))
    #
    #             protein_labels.append(protein_label)
    #             protein_bio_features.append(protein_fea)
    #             protein_attention_mats.append(protein_matrix)
    #             protein_seq_len.append(len(protein_fea))
    #
    #     return peptide_bio_features, peptide_attention_mats, peptide_labels, peptide_seq_len, protein_bio_features, \
    #            protein_attention_mats, protein_labels, protein_seq_len
    def get_fea(self):
        peptide_bio_features, peptide_attention_mats, peptide_labels, peptide_seq_len = [], [], [], []
        protein_bio_features, protein_attention_mats, protein_labels, protein_seq_len = [], [], [], []

        for index, row in tqdm(self.pdb_id.iterrows(), desc=f"loading {self.file_name}"):
            peptide_protein_fea = pd.read_pickle(os.path.join(parent_directory, f"preprocess/train_data/{row['id']}.pkl"))
            peptide_protein_mat = pd.read_pickle(os.path.join(parent_directory, f"preprocess/train_mat/{row['id']}.pkl"))

            peptide_fea, protein_fea = peptide_protein_fea["peptide"], peptide_protein_fea["protein"]

            peptide_label = peptide_fea["label"].values.tolist()
            peptide_fea = peptide_fea[self.features].values.tolist()
            peptide_matrix = peptide_protein_mat["peptide"]

            protein_label = protein_fea["label"].values.tolist()
            protein_fea = protein_fea[self.features].values.tolist()
            protein_matrix = peptide_protein_mat["protein"]

            peptide_labels.append(peptide_label)
            peptide_bio_features.append(peptide_fea)
            peptide_attention_mats.append(peptide_matrix)
            peptide_seq_len.append(len(peptide_fea))

            protein_labels.append(protein_label)
            protein_bio_features.append(protein_fea)
            protein_attention_mats.append(protein_matrix)
            protein_seq_len.append(len(protein_fea))
        return peptide_bio_features, peptide_attention_mats, peptide_labels, peptide_seq_len, protein_bio_features, \
               protein_attention_mats, protein_labels, protein_seq_len


def collate_fn(batch):
    bio_features_num = batch[0]["fea_num"]

    peptide_bio_feature = [data["peptide_bio_features"] for data in batch]
    peptide_seq_len = [data["peptide_seq_len"] for data in batch]
    peptide_labels = [data["peptide_labels"] for data in batch]
    peptide_attention_mat = [data["peptide_attention_mat"] for data in batch]
    peptide_max_len = max(peptide_seq_len)

    # generate mask: attention_mask
    peptide_attention_mask = [torch.cat([torch.tensor([1] * item), torch.tensor([0] * (peptide_max_len - item))], -1)
                              for item in peptide_seq_len]
    peptide_attention_mask = torch.stack(peptide_attention_mask, dim=0)

    # padding labels
    peptide_labels = [torch.cat([item, torch.tensor([0] * (peptide_max_len - len(item)))], -1) for item in peptide_labels]
    peptide_labels = torch.stack(peptide_labels, dim=0)

    # padding bio features
    for i in range(len(peptide_bio_feature)):
        padding_zero = torch.zeros([peptide_max_len - peptide_bio_feature[i].shape[0], bio_features_num], dtype=torch.float)
        peptide_bio_feature[i] = torch.cat([peptide_bio_feature[i], padding_zero], 0)
    peptide_bio_feature = torch.stack(peptide_bio_feature)

    # padding matrix
    for i in range(len(peptide_attention_mat)):
        padding_zero1 = torch.zeros([peptide_max_len - peptide_seq_len[i], peptide_seq_len[i]], dtype=torch.float)
        padding_zero2 = torch.zeros([peptide_max_len, peptide_max_len - peptide_seq_len[i]], dtype=torch.float)
        peptide_attention_mat[i] = torch.cat([peptide_attention_mat[i], padding_zero1], 0)
        peptide_attention_mat[i] = torch.cat([peptide_attention_mat[i], padding_zero2], 1)
    peptide_attention_mat = torch.stack(peptide_attention_mat, dim=0)

    protein_bio_feature = [data["protein_bio_features"] for data in batch]
    protein_seq_len = [data["protein_seq_len"] for data in batch]
    protein_labels = [data["protein_labels"] for data in batch]
    protein_attention_mat = [data["protein_attention_mat"] for data in batch]
    protein_max_len = max(protein_seq_len)

    # generate mask: attention_mask
    protein_attention_mask = [torch.cat([torch.tensor([1] * item), torch.tensor([0] * (protein_max_len - item))], -1)
                              for item in protein_seq_len]
    protein_attention_mask = torch.stack(protein_attention_mask, dim=0)

    # padding labels
    protein_labels = [torch.cat([item, torch.tensor([0] * (protein_max_len - len(item)))], -1) for item in protein_labels]
    protein_labels = torch.stack(protein_labels, dim=0)

    # padding bio features
    for i in range(len(protein_bio_feature)):
        padding_zero = torch.zeros([protein_max_len - protein_bio_feature[i].shape[0], bio_features_num], dtype=torch.float)
        protein_bio_feature[i] = torch.cat([protein_bio_feature[i], padding_zero], 0)
    protein_bio_feature = torch.stack(protein_bio_feature)

    # padding matrix
    for i in range(len(protein_attention_mat)):
        padding_zero1 = torch.zeros([protein_max_len - protein_seq_len[i], protein_seq_len[i]], dtype=torch.float)
        padding_zero2 = torch.zeros([protein_max_len, protein_max_len - protein_seq_len[i]], dtype=torch.float)
        protein_attention_mat[i] = torch.cat([protein_attention_mat[i], padding_zero1], 0)
        protein_attention_mat[i] = torch.cat([protein_attention_mat[i], padding_zero2], 1)
    protein_attention_mat = torch.stack(protein_attention_mat, dim=0)

    return peptide_bio_feature, peptide_attention_mat, peptide_labels, peptide_attention_mask, \
           protein_bio_feature, protein_attention_mat, protein_labels, protein_attention_mask


def collate_fn_single(batch):
    bio_features_num = batch[0]["fea_num"]

    peptide_bio_feature = [data["peptide_bio_features"] for data in batch]
    peptide_labels = [data["peptide_labels"] for data in batch]
    peptide_attention_mat = [data["peptide_attention_mat"] for data in batch]

    protein_bio_feature = [data["protein_bio_features"] for data in batch]
    protein_labels = [data["protein_labels"] for data in batch]
    protein_attention_mat = [data["protein_attention_mat"] for data in batch]

    return peptide_bio_feature[0].unsqueeze(0), peptide_attention_mat[0].unsqueeze(0), peptide_labels[0].unsqueeze(0), \
           protein_bio_feature[0].unsqueeze(0), protein_attention_mat[0].unsqueeze(0), protein_labels[0].unsqueeze(0)


if __name__ == "__main__":
    pass
    
