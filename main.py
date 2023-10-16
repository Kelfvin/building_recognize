import os
import time

import pandas as pd
import torch
from sklearn.model_selection import KFold

from cfg import CFG
from data import build_dataloader
from logger import Logger
from utlis import (
    build_loss,
    build_model,
    build_trainsforms,
    train_one_epoch,
    try_gpu,
    valid_one_epoch,
)

if __name__ == "__main__":
    # 导入数据
    df = pd.read_csv(
        os.path.join("data", "train_mask.csv"), sep="\t", names=["name", "mask"]
    )

    # 导入测试集
    test_df = pd.read_csv(
        os.path.join("data", "test_a_samplesubmit.csv"),
        sep="\t",
        names=["name", "mask"],
    )
    test_df.dropna(inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    df["image_path"] = df["name"].apply(lambda x: os.path.join("data", "train", x))

    kf = KFold(
        n_splits=CFG.n_fold,
        shuffle=True,
    )

    df.loc[:, "fold"] = -1

    for fold, (_train_index, valid_index) in enumerate(kf.split(df)):
        df.loc[valid_index, "fold"] = fold

    if not os.path.exists(CFG.ckpt_path):
        os.makedirs(CFG.ckpt_path)

    # 训练与验证
    train_val_flag = True
    if train_val_flag:
        # 数据处理
        df = pd.read_csv(
            os.path.join(CFG.data_path, "train_mask.csv"),
            sep="\t",
            names=["name", "mask"],
        )
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["image_path"] = df["name"].apply(
            lambda x: os.path.join(CFG.data_path, "train", x)
        )

        # 交叉验证训练设置
        kf = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        df.loc[:, "fold"] = -1
        for fold, (_train_index, valid_index) in enumerate(kf.split(df)):
            df.loc[valid_index, "fold"] = fold
        log = Logger(os.path.join(CFG.ckpt_path, "log.txt"))

        log.open(
            os.path.join(
                CFG.ckpt_path,
                "log_train.txt",
            ),
            mode="a",
        )
        log.write(f"-----------{CFG.model_name} start -----------\n")
        log.write(f"-----------seed {CFG.seed} -----------\n")
        log.write(f"-----------lr {CFG.lr} -----------\n")
        log.write(f"-----------n_fold {CFG.n_fold} -----------\n")
        log.write(f"-----------epoch {CFG.epoch} -----------\n")

        start_fold = 0

        for fold in range(start_fold, CFG.n_fold):
            log.write(f"-----------fold {fold} start -----------\n")
            data_transforms = build_trainsforms()
            train_loader, valid_loader = build_dataloader(df, fold, data_transforms)
            model = build_model()
            model = model.to(try_gpu())
            optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
            losses_dict = build_loss()

            start_epoch = 0

            best_val_dice = 0
            best_epoch = 0

            for epoch in range(start_epoch, CFG.epoch):
                start_time = time.time()
                train_one_epoch(model, train_loader, optimizer, losses_dict, log, epoch)
                _, _, _, val_dice, val_iou = valid_one_epoch(
                    model, valid_loader, log, epoch
                )
                end_time = time.time()
                log.write(f"epoch {epoch} spend time {end_time-start_time}\n")
                print(f"epoch {epoch} spend time {end_time-start_time}\n")
                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    best_epoch = epoch
                    torch.save(
                        model.state_dict(),
                        os.path.join(CFG.ckpt_path, "best_model.pth"),
                    )
                    log.write(f"epoch {epoch} save best model\n")
                    print(f"epoch {epoch} save best model\n")
                if epoch % CFG.save_epoch == 0:
                    torch.save(
                        model.state_dict(),
                        os.path.join(CFG.ckpt_path, f"{epoch}_model.pth"),
                    )
                    log.write(f"epoch {epoch} save model\n")
                    print(f"epoch {epoch} save model\n")

            log.write(f"-----------fold {fold} end -----------\n")
            print(f"-----------fold {fold} end -----------\n")
            log.write(
                f"-----------best epoch {best_epoch} best val dice {best_val_dice} -----------\n"
            )
            print(
                f"-----------best epoch {best_epoch} best val dice {best_val_dice} -----------\n"
            )

        log.write(f"-----------{CFG.model_name} end -----------\n")
        print(f"-----------{CFG.model_name} end -----------\n")
