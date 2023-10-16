import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from cfg import CFG
from network import Unet


# 1.将图片编码为rle格式
def rle_encode(im):
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = im.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


# 2.将rle格式进行解码为图片
def rle_decode(mask_rle, shape=(512, 512)):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def train_one_epoch(model, train_loader, optimizer, losses_dict, log, epoch):
    model.train()
    losses_all, bce_all, dice_all = 0, 0, 0
    log.write(f"-----------epoch {epoch} start -----------\n")
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Train", ncols=0)
    for _, (images, mask) in pbar:
        images = images.to(try_gpu())
        mask = mask.to(try_gpu())
        optimizer.zero_grad()
        y_pred = model(images)
        bce_loss = losses_dict["BCELoss"](y_pred, mask)
        # dice_loss = losses_dict['dice_loss'](y_pred,mask)
        loss = bce_loss
        loss.backward()
        optimizer.step()

        losses_all += loss.item()
        bce_all += bce_loss.item()
        # dice_all += dice_loss.item()
        dice_all += 0

    current_lr = optimizer.param_groups[0]["lr"]
    losses_all = losses_all / len(train_loader)
    bce_all = bce_all / len(train_loader)
    dice_all = dice_all / len(train_loader)

    log.write(
        f"epoch {epoch} lr {current_lr} train loss {losses_all} bce loss {bce_all} dice loss {dice_all}\n"
    )
    print(
        f"epoch {epoch} lr {current_lr} train loss {losses_all} bce loss {bce_all} dice loss {dice_all}\n"
    )


def valid_one_epoch(model, valid_loader, log, epoch):
    model.eval()
    val_scores = []
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc="Valid", ncols=0)
    for _, (images, mask) in pbar:
        y_pred = model(images)
        y_pred = y_pred.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        val_dice = dice_coef(mask, y_pred)
        val_iou = iou_coef(mask, y_pred)
        val_scores.append([val_dice, val_iou])

    val_scores = np.array(val_scores)
    val_dice = np.mean(val_scores[:, 0])
    val_iou = np.mean(val_scores[:, 1])

    log.write(f"epoch {epoch} valid dice {val_dice} valid iou {val_iou}\n")
    print(f"epoch {epoch} valid dice {val_dice} valid iou {val_iou}\n")

    return images, y_pred, mask, val_dice, val_iou


def test_one_epoch(ckpt_paths, test_loader, CFG: CFG):
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Test", ncols=0)
    for _, (images) in pbar:
        size = images.shape
        masks = torch.zeros(size[0], CFG.class_nums, size[2], size[3])
        images = images.to(try_gpu())
        model = build_model()
        model.load_state_dict(torch.load(f"{CFG.ckpt_path}/best_model.pth"))
        model.eval()
        with torch.no_grad():
            y_pred = model(images)
            y_pred = y_pred.detach().cpu().numpy()
            y_pred = (y_pred > 0.5).astype(np.float32)
            masks += y_pred

            val_dice_test = dice_coef(masks, y_pred)
            val_iou_test = iou_coef(masks, y_pred)

            print(f"test dice {val_dice_test} test iou {val_iou_test}\n")
    return images, masks


# 构建评价指标
def dice_coef(
    y_true: torch.Tensor, y_pred: torch.Tensor, thr=0.5, dim=(2, 3), epsilon=0.001
) -> torch.Tensor:
    y_pred = (y_pred > thr).float()
    intersection = torch.sum(y_true * y_pred, dim=dim)
    union = torch.sum(y_true, dim=dim) + torch.sum(y_pred, dim=dim)
    coef = (2 * intersection + epsilon) / (union + epsilon)
    return torch.mean(coef)


def iou_coef(
    y_true: torch.Tensor, y_pred: torch.Tensor, thr=0.5, dim=(2, 3), epsilon=0.001
) -> torch.Tensor:
    y_pred = (y_pred > thr).float()
    intersection = torch.sum(y_true * y_pred, dim=dim)
    union = torch.sum(y_true, dim=dim) + torch.sum(y_pred, dim=dim) - intersection
    coef = (intersection + epsilon) / (union + epsilon)
    return torch.mean(coef)


def build_loss() -> dict:
    BCELoss = nn.BCEWithLogitsLoss()
    return {"BCELoss": BCELoss}


def build_model() -> nn.Module:
    model = Unet(CFG.class_nums)
    return model.to(try_gpu())


def try_gpu(i=0) -> torch.device:
    """如果存在，则返回gpu(i)，如果是有苹果的GPU则用mps，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")

    if torch.has_mps:
        return torch.device("mps")

    return torch.device("cpu")


def try_all_gpus() -> torch.device:
    """返回所有可用的GPU，如果是有苹果的GPU则用mps，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

    if devices:
        return devices

    if torch.has_mps:
        return [torch.device("mps")]

    return [torch.device("cpu")]


def build_trainsforms() -> dict:
    return {
        "train": transforms.Compose(
            [  # noqa: F821
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        ),
        "valid_test": transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        ),
    }
