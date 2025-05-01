import pandas as pd
import torch
from utils.builder import build_dataloader, build_model_sup
from utils.trainer import train_step, val_step
import numpy as np
from setting import get_config
import os
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.backends.cudnn as cudnn
import os


if __name__ == "__main__":
    # 加载参数
    args, writer = get_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.training["gpu_id"]
    torch.manual_seed(args.training["seed"])
    model_str = args.model["model"]
    backbone = args.model["backbone"]
    num_classes = args.model["num_classes"]
    # 读取路径
    csv_path = args.path[args.model["dataset"]]["csv_path"]
    img_path = args.path[args.model["dataset"]]["img_path"]
    # 保存路径
    pth_path = args.path["pth"]
    result_path = args.path["result"]
    # 训练参数
    batch_size, num_workers, epochs = args.training["batch_size"], args.training["num_workers"], args.training["epochs"]
    lr, momentum, weight_decay = args.optimizer["lr"], args.optimizer["momentum"], args.optimizer["weight_decay"]
    T_max, eta_min = args.scheduler["T_max"], args.scheduler["eta_min"]

    for fold in range(3):
        fold = fold + 3
        logging.info(f"---------------fold:{fold}---------------")
        # dataloader
        df_train = pd.read_csv(f"{csv_path}/train_fold_{fold}.csv")
        df_val = pd.read_csv(f"{csv_path}/val_fold_{fold}.csv")
        images_train = [img_path + i for i in df_train["filename"].tolist()]
        images_val = [img_path + i for i in df_val["filename"].tolist()]
        labels_train = df_train["label"].tolist()
        labels_val = df_val["label"].tolist()

        dataloader_train = build_dataloader(images_train, labels_train, batch_size, num_workers, "sup")
        dataloader_val = build_dataloader(images_val, labels_val, 1, num_workers, "val")

        if model_str == "sup":
            model = build_model_sup(backbone, num_classes)
        else:
            raise ValueError("model_str set error")
        class_counts = np.bincount(labels_train)  # 计算每个类别的样本数

        total_samples = len(labels_train)
        # 计算权重：样本数越少，权重越大
        class_weights = [total_samples / count for count in class_counts]
        # 转换为 PyTorch 张量，并传递给 CrossEntropyLoss
        class_weights = torch.tensor(class_weights, dtype=torch.float).cuda()  # 如果使用 GPU，请加上 .cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        # criterion = torch.nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            model = model.cuda()
            criterion = criterion.cuda()
            cudnn.benchmark = True
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=7, verbose=True)
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

        # train
        list_train = []
        best_acc = 0
        for epoch in range(epochs):
            logging.info(f"epoch:{epoch}")

            epoch_loss_train, acc_train, pre_train, recall_train, f1_train, class_f1_train = train_step(
                dataloader_train, model, optimizer, criterion, num_classes
            )
            epoch_loss_val, acc_val, pre_val, recall_val, f1_val, class_f1_val = val_step(
                dataloader_val, model, criterion, num_classes
            )
            scheduler.step()

            writer.add_scalar(f"{fold}/train/loss", epoch_loss_train, epoch)
            writer.add_scalar(f"{fold}/train/acc", acc_train, epoch)
            writer.add_scalar(f"{fold}/val/loss", epoch_loss_val, epoch)
            writer.add_scalar(f"{fold}/val/acc", acc_val, epoch)
            writer.add_scalar(f"{fold}/val/recall", recall_val, epoch)
            writer.add_scalar(f"{fold}/val/pre", pre_val, epoch)

            if acc_val > best_acc:
                best_acc = acc_val
            torch.save(model, pth_path + "/" + str(fold) + "epoch" + str(epoch) + ".pth")

            result_train = [acc_train, acc_val, pre_val, recall_val, f1_val, class_f1_val]
            list_train.append(result_train)

            # 直接按列提取数据，避免转置操作
            df = pd.DataFrame(
                {
                    "acc_train": [x[0] for x in list_train],
                    "acc_val": [x[1] for x in list_train],
                    "preall": [x[2] for x in list_train],
                    "recall": [x[3] for x in list_train],
                    "f1_val": [x[4] for x in list_train],
                    "class_f1_val": [x[5] for x in list_train],
                }
            )
            df.to_csv(result_path + "/" + str(fold) + ".csv", index=False)
