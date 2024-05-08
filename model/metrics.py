# coding: utf-8
# @Author    :陈梦淇
# @time      :2024/3/19
import math
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error, \
    average_precision_score
from sklearn.metrics import matthews_corrcoef
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")


def TP_FP_TN_FN(y_pre, y_label):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(0, len(y_pre)):
        if y_pre[i] == 1 and y_label[i] == 1:
            TP = TP + 1
        if y_pre[i] == 1 and y_label[i] == 0:
            FP = FP + 1
        if y_pre[i] == 0 and y_label[i] == 0:
            TN = TN + 1
        if y_pre[i] == 0 and y_label[i] == 1:
            FN = FN + 1
    return TP, FP, TN, FN


def metrics_classification(y_pre_class, y_pre_logit, y_label):
    """

    :param y_pre_class: 预测类别值
    :param y_pre_logit: 预测概率值
    :param y_label: 真实标签类别值
    :return:
    """
    TP, FP, TN, FN = TP_FP_TN_FN(y_pre_class, y_label)
    try:
        precision = TP / (TP + FP)
    except Exception as e:
        precision = 0

    try:
        recall = TP / (TP + FN)
    except Exception as e:
        recall = 0

    if precision == -1 or recall == -1:
        F1 = 0
    else:
        try:
            F1 = 2 * ((precision * recall) / (precision + recall))
        except Exception as e:
            F1 = 0

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    try:
        ppv = TP / (TP + FP)  # positive predictive value
    except Exception as e:
        ppv = 0

    try:
        npv = TN / (TN + FN)  # negative predictive value
    except Exception as e:
        npv = 0

    try:
        tpr = TP / (TP + FN)  # sensitivity / true positive rate
    except Exception as e:
        tpr = 0

    try:
        tnr = TN / (TN + FP)  # specificity / true negative rate
    except Exception as e:
        tnr = 0

    try:
        auc_score = roc_auc_score(y_label, y_pre_logit)
    except Exception as e:
        auc_score = 0

    try:
        auprc_score = average_precision_score(y_label, y_pre_logit)
    except Exception as e:
        auprc_score = 0.5

    try:
        mcc = matthews_corrcoef(y_true=y_label, y_pred=y_pre_class)
    except Exception as e:
        mcc = 0

    return [precision, recall, mcc, auc_score, auprc_score, accuracy]

