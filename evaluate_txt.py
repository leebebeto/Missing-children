import os
from PIL import Image
import random
import pickle
import torchvision.transforms as T # 이미지 전처리를 지원

import torch
from utils_txt import cos_dist, fixed_img_list

import numpy as np

from model import *
tst_data_dir = "/home/nas1_userE/jungsoolee/Face_dataset/FGNET_new_align"
txt_root = "/home/nas1_userE/jungsoolee/Missing-children/analysis_yong/Confusion_Matrix/FGNETC-Confusion"

# 데이터 관련 세팅
gray_scale = False

# Hyperparameter
feature_dim = 512

# GPU가 있을 경우 연산을 GPU에서 하고 없을 경우 CPU에서 진행
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
net_depth, drop_ratio, net_mode = 50, 0.6, 'ir_se'
model = Backbone(net_depth, drop_ratio, net_mode).to(dev)
model.load_state_dict(torch.load(model_path))
model.eval()

def fixed_img_list(text_pair):

    f = open(text_pair, 'r')
    lines = []

    while True:
        line = f.readline()
        if not line:
            break
        lines.append(line)
    f.close()

    random.shuffle(lines)
    return lines

trans_list = []
trans_list += [T.ToTensor()]
trans_list += [T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
t = T.Compose(trans_list)

def control_text_list(txt_root, txt_dir):
    text_path = os.path.join(txt_root, txt_dir)
    lines = sorted(fixed_img_list(text_path))
    pairs = [' '.join(line.split(' ')[1:]) for line in lines]
    labels = [int(line.split(' ')[0]) for line in lines]
    return pairs, labels


import tqdm


def verification(net, label_list, pair_list, transform):
    similarities = []
    labels = []
    assert len(label_list) == len(pair_list)
    best_pred = None
    best_similarities = None

    if len(label_list) == 0:
        return 0, 0, 0
    # 주어진 모든 이미지 pair에 대해 similarity 계산
    net.eval()
    #     import pdb; pdb.set_trace()
    with torch.no_grad():  # Test 때 GPU를 사용할 경우 메모리 절약을 위해 torch.no_grad() 내에서 하는 것이 좋다.
        for idx, pair in enumerate(tqdm.tqdm(pair_list)):
            path_1, path_2 = pair.split('.png /home')
            path_1 = path_1 + '.png'
            path_2 = '/home' + path_2
            path_2 = path_2[:-2]
            #             path_2 = path_2[:-1]
            #             path_1 = path_1.split('/')[-1]
            #             path_2 = path_2.split('/')[-1]

            img_1 = t(Image.open(path_1)).unsqueeze(dim=0).to(dev)
            img_2 = t(Image.open(path_2)).unsqueeze(dim=0).to(dev)
            imgs = torch.cat((img_1, img_2), dim=0)

            # Extract feature and save
            features = net(imgs)
            similarities.append(cos_dist(features[0], features[1]).cpu())
            label = int(label_list[idx])
            labels.append(label)

    best_accr = 0.0
    best_th = 0.0

    # 각 similarity들이 threshold의 후보가 된다
    list_th = similarities

    # list -> tensor

    similarities = torch.stack(similarities, dim=0)
    #     labels = torch.ByteTensor(labels)
    labels = torch.ByteTensor(label_list)

    pred_list = []

    # 각 threshold 후보에 대해 best accuracy를 측정
    for i, th in enumerate(list_th):
        pred = (similarities >= th)
        correct = (pred == labels)
        accr = torch.sum(correct).item() / correct.size(0)

        if accr > best_accr:
            best_accr = accr
            best_th = th.item()

            best_pred = pred
            best_similarities = similarities

    return best_accr, best_th, idx

import glob
txt_root = '/home/nas1_userE/jungsoolee/Face_dataset/txt_files'
# txt_dirs = glob.glob('/home/nas1_userE/jungsoolee/Face_dataset/txt_files/*')
# txt_dirs = [dir.split('/')[-1] for dir in txt_dirs]
txt_dir = 'lag.txt'
print(f'working on : {txt_dir}')
pair_list, label_list = control_text_list(txt_root, txt_dir)
best_accr, best_th, idx = verification(model, label_list, pair_list, transform=t)
print(f'txt_dir: {txt_dir}, best_accr: {best_accr}')