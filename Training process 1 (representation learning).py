# ========
# 由于采用了加权平均，
# 本地调试时推理流程：分别加载b5、b6、seresnext101三个模型并推理测试集，得到全部测试集的三类概率
# 将三个模型的分类概率按 {b5:0.4, b6:0.4, se101:0.2}的比例集成


# 为保持代码简洁，不再按加权系数分开写测试集代码
# 不同比例的测试集权重已包含在压缩包的npy文件夹内
# 'b5_0~b5_14' 为b5模型15次TTA的分类概率，数组形状均为(2000,3)
# 'b6_0~b6_14' 为b6模型15次TTA的分类概率，数组形状均为(2000,3)
# 'se101_0~se101_14' 为se101模型15次TTA的分类概率，数组形状均为(2000,3)
# 加载方法见代码最下方的“复现结果”部分

import os
import cv2
import time
import numpy as np
import math
from collections import OrderedDict

from tqdm import tqdm, tqdm_notebook
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch
import torchvision
import torch.nn as nn
# from torch.nn import functional as F
import torch.optim as optim
# from torch.optim.optimizer import Optimizer, required
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from albumentations import (Resize, RandomCrop, VerticalFlip, HorizontalFlip, Normalize, Compose)
from albumentations.pytorch import ToTensor
import albumentations as albu
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision.models import resnext50_32x4d, resnet18
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score, accuracy_score
import pretrainedmodels
# 加入注意力1
from efficientnet_pytorch import pvt_v2_b1
#from efficientnet_pytorch import resnet18_2
# # 加注意力2
# from efficientnet_pytorch import EfficientNet2
# # 未加注意力1
# from efficientnet_pytorch import EfficientNet3
# # 未加注意力2
# from efficientnet_pytorch import EfficientNet4
# ------------------------------------------------
# ------------------------------------------------------

from PIL import Image
from glob import glob
import apex
from apex import amp
import pandas as pd
import gc
import warnings
from torchvision import models
from torch.autograd import Variable

warnings.filterwarnings('ignore')

# ============================================ 制作训练集csv文件 =============================================
# 以下路径已包含初赛训练集 + 复赛训练集
train_AD_path = 'E:/Transformer/classification/Train/AD/*'
train_NC_path = 'E:/Transformer/classification/Train/NC/*'
train_MCI_path = 'E:/Transformer/classification/Train/MCI/*'
img_AD_list = glob(os.path.join(train_AD_path))
img_NC_list = glob(os.path.join(train_NC_path))
img_MCI_list = glob(os.path.join(train_MCI_path))
print(len(img_AD_list))  # 4000
print(len(img_NC_list))  # 4000
print(len(img_MCI_list))  # 4000



df_train_NC = pd.DataFrame(columns=['img', 'cls'])
for i in tqdm(range(len(img_NC_list))):
    df_train_NC.loc[i, 'img'] = img_NC_list[i]
    df_train_NC.loc[i, 'cls'] = int(0)

df_train_AD = pd.DataFrame(columns=['img', 'cls'])
for i in tqdm(range(len(img_AD_list))):
    df_train_AD.loc[i, 'img'] = img_AD_list[i]
    df_train_AD.loc[i, 'cls'] = int(1)

df_train_MCI = pd.DataFrame(columns=['img', 'cls'])
for i in tqdm(range(len(img_MCI_list))):
    df_train_MCI.loc[i, 'img'] = img_MCI_list[i]
    df_train_MCI.loc[i, 'cls'] = int(2)

df_train = pd.merge(df_train_AD, df_train_NC, how='outer')
df_train = pd.merge(df_train, df_train_MCI, how='outer')
df_train['cls'] = df_train['cls'].astype('int')

df_train.to_csv('E:/Transformer/classification/Train/train_1w2.csv', index=False)  # 保存包含路径的训练数据csv文件


# ============================================ 训练部分 =============================================
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        print(x)
        return [self.transform(x), self.transform(x)]
# 图像增强
def get_transforms(phase, p=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    list_transforms = []
    list_transforms.extend(
        [
            Resize(image_size[0], image_size[1], interpolation=Image.BILINEAR),
            albu.HorizontalFlip(p=p),
            albu.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1,
                                  rotate_limit=360, border_mode=cv2.BORDER_CONSTANT,
                                  value=0, p=1),
        ]
    )
    if phase == "train":
        list_transforms.extend(
            [
                albu.OneOf([
                    albu.RandomGamma(gamma_limit=(60, 120), p=p),
                    albu.RandomBrightness(limit=0.2, p=p),
                    albu.RandomContrast(limit=0.2, p=p),
                    albu.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=p),
                ]),
                albu.OneOf([
                    albu.Blur(blur_limit=3, p=p),
                    albu.MedianBlur(blur_limit=3, p=p),
                ]),
                albu.Cutout(num_holes=16, max_h_size=int(image_size[0] / 18), max_w_size=int(image_size[0] / 18),
                            fill_value=0, p=p),
            ],
        )
    list_transforms.extend(
        [
            Normalize(mean=mean, std=std, p=1),
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms


# dataloader
class brain_Dataset(Dataset):
    def __init__(self, idx, df, phase="train"):
        assert phase in ("train", "val", "test")
        self.idx = idx
        self.df = df
        self.phase = phase
        self.transform = get_transforms(phase, p=0.5)

    def __getitem__(self, index):
        real_idx = self.idx[index]
        image_path = self.df.loc[real_idx]['img']
        image = cv2.imread(image_path)
        augmented = self.transform(image=image)
        #augmented = [self.transform(image=image),self.transform(image=image)]
        image1 = augmented['image']
        image2 = augmented['image']
        image = [image1,image2]
        label = self.df.loc[real_idx]['cls']
        if label == 0:
            label = torch.tensor([0.9, 0.05, 0.05])
        elif label == 1:
            label = torch.tensor([0.05, 0.9, 0.05])
        elif label == 2:
            label = torch.tensor([0.05, 0.05, 0.9])
        return image, label

    def __len__(self):
        return len(self.idx)


def provider_kf(df, train_idx, val_idx, phase, mean=None, std=None, batch_size=500, num_workers=0):
    index = train_idx if phase == "train" else val_idx
    dataset = brain_Dataset(index, df, phase=phase)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=True,
    )

    return dataloader


# 评价指标及日志
class Meter:
    def __init__(self, phase, epoch):
        self.f1_list = []
        # self.acc_list = []

    def update(self, targets, outputs):
        targets = targets.detach().cpu()
        targets = np.argmax(targets, axis=1)
        probs = torch.sigmoid(outputs)
        probs = torch.argmax(outputs, dim=1).detach().cpu()
        f1_ = f1_score(targets, probs, average='macro')
        self.f1_list.append(f1_)

    def get_metrics(self):
        f1_mean = np.nanmean(self.f1_list)
        return f1_mean


def epoch_log(phase, epoch, epoch_loss, meter, start):
    f1_mean = np.mean(meter.get_metrics())
    print("Loss: %0.7f | f1: %0.5f" % (epoch_loss, f1_mean))
    return f1_mean

#------------片状注意力机制
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)				# 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
#-------------------------------------
class SE(nn.Module):

    def __init__(self, in_chnls=3, ratio=1):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls//ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls//ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return F.sigmoid(out)
# -----------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_size, common_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 4, common_size)
        )

    def forward(self, x):
        out = self.linear(x)
        return out
# 定义模型
class eff_b7_1(nn.Module):
    def __init__(self):
        super(eff_b7_1, self).__init__()
        model1 = pvt_v2_b1()
        #model1 = resnet18_2()
        self.resnet1 = model1
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.ReLU(True),
            #nn.Linear(512, 128),
            nn.Linear(512, 128),
        )

    def forward(self, img):
        out = self.resnet1(img)
        #x = self.fc(out)
        #print(x.shape)
        #x = torch.flatten(out, 1)
        x = F.normalize(out, dim=1)
        return x


class eff_b7_2(nn.Module):
    def __init__(self):
        super(eff_b7_2, self).__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(512, 3),
        )
    def forward(self, img):
        out = self.fc(img)
        return out
# ------------------------------------------------------
#----------------------------------
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        labels = torch.max(labels, 1)[1]

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

# 主程序
class Trainer_kf(object):
    # def __init__(self, model, train_idx, val_idx, num_epochs=600):
    def __init__(self, model, train_idx, val_idx, num_epochs=800):
        # def __init__(self, model, train_idx, val_idx, num_epochs=500):
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.lr_list = []
        self.lr_list2 = []
        self.num_workers = 0
        self.batch_size = {"train": 64, "val": 64}
        self.accumulation_steps = 512 // self.batch_size['train']
        self.lr = 5e-4
        self.num_epochs = num_epochs
        self.best_loss = float("inf")
        self.best_f1 = float(0)
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        #self.net2 = model2
        # 回归损失函数
        self.criterion = SupConLoss()
        # 1
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, amsgrad=True)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[60, 100, 120], gamma=0.1)
        self.net = self.net.to(self.device)
        self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level="O1")
        # 2
        cudnn.benchmark = True
        #duibi1
        self.dataloaders = {
            phase: provider_kf(
                df=df_train,
                train_idx=self.train_idx,
                val_idx=self.val_idx,
                phase=phase,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.f1_scores = {phase: [] for phase in self.phases}

    def iterate_train(self, epoch, phase):
        self.net.train()
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f'Starting epoch: {epoch} | phase: {phase} | start time: {start}')
        batch_size = self.batch_size[phase]
        dataloader = self.dataloaders[phase]


        running_loss = 0.0
        total_batches = len(dataloader)
        for itr, batch in enumerate(dataloader):
            images, targets = batch
            # images = images[0]
            # images = images[1]
            images = torch.cat([images[0], images[1]], dim=0)

            #print(images.shape)
            # print("---")
            # print(targets.shape)
           # images =torch.cat([images1[0], images2[1]], dim=1)
            #images =images1[0]
            #images  = torch.mean(torch.stack([images1[0], images2[1]]), dim=0)
            images = images.to(self.device)
            label = targets.to(self.device)
            outputs  = self.net(images)
            #outputs2 = self.net(images)
            #outputs = torch.cat([outputs1[0], outputs2[0]], dim=0)
            # topk
            loss_all = torch.zeros((1, len(outputs)))
            # 1
            bsz = (label.shape[0])
            f1, f2 = torch.split(outputs, [bsz, bsz], dim=0)
            #features = torch.mean(torch.stack([f1, f2]), 0)
            #print(features.shape)
            #features = torch.mean(torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)]), 1)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            #features = 0.02 * ((torch.abs(features)) ** 2) + torch.abs(features)
           # print(features.shape)
            loss = self.criterion(features, label)
            # loss2 = self.criterion2(features, label)
            # loss = loss1 + loss2 * 0.02
            # loss = torch.mean(torch.stack([loss1, loss2]), 0)
            # loss = loss1
            # out = outputs
            # lab = label
            # loss = self.criterion(out, lab)
            # loss = loss.mean()
            loss = loss / self.accumulation_steps
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            if (itr + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            running_loss += loss.item()
            #meter.update(targets, features)
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        #f1_mean = epoch_log(phase, epoch, epoch_loss, meter, start)

        self.losses[phase].append(epoch_loss)
        #self.f1_scores[phase].append(f1_mean)
        #return epoch_loss, f1_mean
        print(epoch_loss)
        return epoch_loss

    def iterate_val(self, epoch, phase):
        self.net.eval()

        with torch.no_grad():
            meter = Meter(phase, epoch)
            start = time.strftime("%H:%M:%S")
            print(f'Starting epoch: {epoch} | phase: {phase} | start time: {start}')
            batch_size = self.batch_size[phase]
            dataloader = self.dataloaders[phase]
            running_loss = 0.0
            total_batches = len(dataloader)

            for itr, batch in enumerate(dataloader):
                images, targets = batch
                images = torch.cat([images[0], images[1]], dim=0)
                images = images.to(self.device)
                #label = targets.to(self.device)
                label = targets.to(self.device)
                # 1
                #outputs = self.net(images)
                outputs= self.net(images)
                bsz = (label.shape[0])
                f1, f2 = torch.split(outputs, [bsz, bsz], dim=0)
                #features = torch.mean(torch.stack([f1, f2]), 0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                # features = 0.02 * ((torch.abs(features)) ** 2) + torch.abs(features)
                loss = self.criterion(features, label)
                # loss2 = self.criterion2(features, label)
                # loss = loss1 + loss2 * 0.02
                # loss = torch.mean(torch.stack([loss1, loss2]), 0)
                # loss = loss1
                # loss = loss.mean()
                running_loss += loss.item()
                # meter.update(targets, outputs)
                #meter.update(targets, features)
                # measure accuracy and record loss

            epoch_loss = running_loss / total_batches
            #f1_mean = epoch_log(phase, epoch, epoch_loss, meter, start)

            self.losses[phase].append(epoch_loss)
            #self.f1_scores[phase].append(f1_mean)
            #return epoch_loss, f1_mean
            print(epoch_loss)
            return epoch_loss


    def start(self):
       for epoch in range(self.num_epochs):
            self.iterate_train(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "amp": amp.state_dict(),
            }
            #val_loss, f1_mean = self.iterate_val(epoch, "val")
            val_loss = self.iterate_val(epoch, "val")
            torch.cuda.empty_cache()
            self.lr_list.append(self.optimizer.param_groups[0]['lr'])
            self.scheduler.step()
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                print('Now loss:', self.optimizer.param_groups[0]['lr'])
                state["best_loss"] = self.best_loss = val_loss
                #self.best_f1 = f1_mean
                torch.save(model.state_dict(), f'E:/Transformer/classification/model/{ph}_{use_model}_fold{fold_n}.pth')
            print()


# 配置文件
'''
共训练3个尺寸*3个模型，共9个模型，每个模型包含5折交叉验证
每个尺寸的每个模型K折拆分时都采用不同的随机种子
3个模型为:
use_model = 'seresnext101'
use_model = 'eff_b5'
use_model = 'eff_b6'
3个尺寸为:
image_size = (128,128)
image_size = (168,168)
image_size = (256,256)
'''
df_train = pd.read_csv('E:/Transformer/classification/Train/train_1w2.csv')  # 此csv即最“制作训练集csv文件”代码部分生成的csv文件
#image_size = (79, 95)
image_size = (95, 79)
num_cls = 3
# use_model = 'seresnext101'
use_model = 'eff_b7_1'
seed = 98
ph = 'seed_' + str(seed) + '_' + str(image_size[0])

# 训练
num_kf = 10
kf = StratifiedKFold(n_splits=num_kf, shuffle=True, random_state=seed)
kf = list(kf.split(np.arange(len(df_train)), df_train['cls']))
min_loss_list = []
for fold_n, (train_index, valid_index) in enumerate(kf):
    if fold_n in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        # if fold_n in [0]:
    #if fold_n in [0, 1, 2]:
        print(fold_n)
        if use_model == 'eff_b7_1':
            model = eff_b7_1()
            #model = eff_b7_2()
        if use_model == 'eff_b7_2':
            model = eff_b7_2()
        elif use_model == 'seresnext101_1':
            model = eff_b7_2()
        model_trainer = Trainer_kf(model, train_index, valid_index, num_epochs=100)
        # model_trainer = Trainer_kf(model, train_index, valid_index, num_epochs=80)
        model_trainer.start()
        losses = model_trainer.losses
        min_loss = np.min(losses['val'])
        print('=' * 30)
        print(f'fold_{fold_n} min loss:{min_loss}')
        print('=' * 30)
        min_loss_list.append(min_loss)
        torch.cuda.empty_cache()
        del model
        # del model
        gc.collect()
print(min_loss_list)
print(f'{num_kf}fold mean loss: {np.mean(min_loss_list)}')
