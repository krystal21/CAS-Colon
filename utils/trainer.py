import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import torch.nn.functional as F
import logging
from collections import defaultdict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_step(train_loader, model, optimizer, criterion, num_classes=10):
    model.train()
    losses = AverageMeter()

    all_targets = []
    all_predictions = []

    for step, (data, label) in enumerate(train_loader):
        img = data.cuda()
        targets = label.cuda()
        # outputs = model(img).squeeze(1)
        outputs = model(img).logits.squeeze(1)
        bsz = label.shape[0]
        loss = criterion(outputs, targets)
        losses.update(loss.item(), bsz)

        _, predicted = torch.max(outputs, 1)

        # 保存所有的预测值和真实值
        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 计算每个类别的 f1
    class_f1 = f1_score(all_targets, all_predictions, average=None, labels=range(num_classes))
    total_precision = precision_score(all_targets, all_predictions, average="macro")
    total_recall = recall_score(all_targets, all_predictions, average="macro")
    total_f1 = f1_score(all_targets, all_predictions, average="macro")
    # 计算总体准确率
    total_accuracy = accuracy_score(all_targets, all_predictions)

    logging.info(
        f"Overall - ACC: {total_accuracy:.4f}, Precision: {total_precision:.4f}, Recall: {total_recall:.4f}, F1: {total_f1:.4f}"
    )
    logging.info(f"Class Accuracy: {class_f1}")

    return losses.avg, total_accuracy, total_precision, total_recall, total_f1, class_f1


def val_step(val_loader, model, criterion, num_classes=10):
    model.eval()
    losses = AverageMeter()

    all_targets = []
    all_predictions = []

    for step, (data, label) in enumerate(val_loader):
        img = data.cuda()
        targets = label.cuda()
        with torch.no_grad():
            outputs = model(img)
            # outputs = model(img).logit.squeeze(1)
            # outputs = model(img).squeeze(1)
        bsz = label.shape[0]
        loss = criterion(outputs, targets)
        losses.update(loss.item(), bsz)

        _, predicted = torch.max(outputs, 1)

        # 保存所有的预测值和真实值
        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

    # 计算每个类别的 f1
    class_f1 = f1_score(all_targets, all_predictions, average=None, labels=range(num_classes))
    total_precision = precision_score(all_targets, all_predictions, average="macro")
    total_recall = recall_score(all_targets, all_predictions, average="macro")
    total_f1 = f1_score(all_targets, all_predictions, average="macro")
    # 计算总体准确率
    total_accuracy = accuracy_score(all_targets, all_predictions)

    logging.info(
        f"Overall - ACC: {total_accuracy:.4f}, Precision: {total_precision:.4f}, Recall: {total_recall:.4f}, F1: {total_f1:.4f}"
    )
    logging.info(f"Class Accuracy: {class_f1}")

    return losses.avg, total_accuracy, total_precision, total_recall, total_f1, class_f1


def test_step(val_loader, model, num_classes=10):
    all_targets = []
    all_predictions = []

    for step, (data, label) in enumerate(val_loader):
        img = data.cuda()
        targets = label.cuda()
        with torch.no_grad():
            outputs = model(img)
        _, predicted = torch.max(outputs, 1)

        # 保存所有的预测值和真实值
        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

    # 计算总体准确率
    total_accuracy = accuracy_score(all_targets, all_predictions)

    # 计算总体的精确率、召回率和 F1 分数
    total_precision = precision_score(all_targets, all_predictions, average="macro")
    total_recall = recall_score(all_targets, all_predictions, average="macro")
    total_f1 = f1_score(all_targets, all_predictions, average="macro")

    # 计算每个类别的准确率（使用 recall 作为每个类别的准确率）
    class_report = classification_report(all_targets, all_predictions, labels=range(num_classes), output_dict=True)
    class_accuracy = [class_report[str(i)]["recall"] for i in range(num_classes)]  # recall 代表每个类的准确率

    # 输出结果
    logging.info(
        f"Overall - ACC: {total_accuracy:.4f}, Precision: {total_precision:.4f}, Recall: {total_recall:.4f}, F1: {total_f1:.4f}"
    )
    logging.info(f"Class Accuracy: {class_accuracy}")

    return total_accuracy, total_precision, total_recall, total_f1, class_accuracy
