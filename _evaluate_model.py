# from config import get_config
import argparse
from learner import face_learner
from data.data_pipe import get_val_pair
from torchvision import transforms as trans

import time
import pdb
import os
import numpy as np

parser = argparse.ArgumentParser(description='for face verification')
parser.add_argument("--net_mode", help="which network, [ir, ir_se, mobilefacenet]",default='ir_se', type=str)
parser.add_argument("--net_depth", help="how many layers [50,100,152]", default=50, type=int)
parser.add_argument("--drop_ratio", help="ratio of drop out", default=0.6, type=float)
parser.add_argument("--model_path", help="evaluate model path", default='fgnetc_best_model_2021-06-01-12-15_accuracy:0.860_step:226000_casia_CASIA_POSITIVE_ZERO_05_MILE_3.pth', type=str)
parser.add_argument("--device", help="device", default='cuda', type=str)
parser.add_argument("--embedding_size", help='embedding_size', default=512, type=int)
parser.add_argument("--wandb", help="whether to use wandb", action='store_true')
parser.add_argument("--epochs", help="num epochs", default=50, type=int)
parser.add_argument("--batch_size", help="batch_size", default=64, type=int)
parser.add_argument("--loss", help="loss", default='Arcface', type=str)
args = parser.parse_args()

# conf = get_config(training=False)
learner = face_learner(args, inference=True)
save_path = '/home/nas1_temp/jooyeolyun/mia_params/'

# learner.load_state(conf, 'ir_se50.pth', model_only=True, from_save_folder=True)
model_path = os.path.join(save_path, args.model_path)
learner.load_state(args, model_path = model_path)
before_time = time.time()

#
# print('evaluating fgnetc')
# dataset_root = os.path.join('/home/nas1_userE/jungsoolee/Face_dataset/face_emore2')
# fgnetc = np.load(os.path.join(dataset_root, "FGNET_new_align_list.npy")).astype(np.float32)
# fgnetc_issame = np.load(os.path.join(dataset_root, "FGNET_new_align_label.npy"))
# fgnetc_accuracy, fgnetc_thres, roc_curve_tensor2, fgnetc_dist = learner.evaluate(args, fgnetc, fgnetc_issame, nrof_folds=10, tta=True)
# print('fgnetc - accuracy:{}, threshold:{}'.format(fgnetc_accuracy, fgnetc_thres))


############################################## original ##############################################


import os
from PIL import Image
import random
import pickle
import torchvision.transforms as T # 이미지 전처리를 지원
import torch
from utils_txt import cos_dist, fixed_img_list
import numpy as np



import torch
from model import *
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

def control_text_list(txt_root, txt_dir, data_dir=None):
    text_path = os.path.join(txt_root, txt_dir)
    lines = sorted(fixed_img_list(text_path))
    if data_dir is None:
        pairs = [' '.join(line.split(' ')[1:]) for line in lines]
        labels = [int(line.split(' ')[0]) for line in lines]
    elif data_dir == 'cacd_vs' or data_dir == 'morph':
        pairs = [' '.join(line.split(' ')[:2]) for line in lines]
        labels = [int(line.split(' ')[-1][0]) for line in lines]
    return pairs, labels


import tqdm

def verification(net, label_list, pair_list, transform, data_dir=None):
    similarities = []
    labels = []
    assert len(label_list) == len(pair_list)

    if len(label_list) == 0:
        return 0, 0, 0
    net.eval()
    with torch.no_grad():  # Test 때 GPU를 사용할 경우 메모리 절약을 위해 torch.no_grad() 내에서 하는 것이 좋다.
        for idx, pair in enumerate(tqdm.tqdm(pair_list)):
            if data_dir is None:
                if 'png' in pair:
                    path_1, path_2 = pair.split('.png /home')
                    path_1 = path_1 + '.png'
                elif 'jpg' in pair:
                    path_1, path_2 = pair.split('.jpg /home')
                    path_1 = path_1 + '.jpg'
                elif 'JPG' in pair:
                    path_1, path_2 = pair.split('.JPG /home')
                    path_1 = path_1 + '.JPG'
                path_2 = '/home' + path_2
                path_2 = path_2[:-2]
            elif data_dir == 'cacd_vs':
                image_root = '/home/nas1_userE/jungsoolee/Face_dataset/CACD_VS_single_112_RF'
                path_1, path_2 = pair.split(' ')
                path_1 = os.path.join(image_root, path_1)
                path_2 = os.path.join(image_root, path_2)

            elif data_dir == 'morph':
                image_root = '/home/nas1_userE/jungsoolee/Face_dataset/Album2_single_112_RF'
                path_1, path_2 = pair.split(' ')
                path_1 = os.path.join(image_root, path_1)
                path_2 = os.path.join(image_root, path_2)

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
    similarities = torch.stack(similarities, dim=0)
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

print('child-specific face recognition starts...')

import glob
txt_root = '/home/nas1_userE/jungsoolee/Face_dataset/txt_files'
txt_dir = 'agedb10_child.txt'
print(f'working on : {txt_dir}')
pair_list, label_list = control_text_list(txt_root, txt_dir)
best_accr, best_th, idx = verification(model, label_list, pair_list, transform=t)
print(f'txt_dir: {txt_dir}, best_accr: {best_accr}')

txt_dir = 'fgnet10_child.txt'
print(f'working on : {txt_dir}')
pair_list, label_list = control_text_list(txt_root, txt_dir)
best_accr, best_th, idx = verification(model, label_list, pair_list, transform=t)
print(f'txt_dir: {txt_dir}, best_accr: {best_accr}')

txt_dir = 'agedb20_child.txt'
print(f'working on : {txt_dir}')
pair_list, label_list = control_text_list(txt_root, txt_dir)
best_accr, best_th, idx = verification(model, label_list, pair_list, transform=t)
print(f'txt_dir: {txt_dir}, best_accr: {best_accr}')

txt_dir = 'fgnet20_child.txt'
print(f'working on : {txt_dir}')
pair_list, label_list = control_text_list(txt_root, txt_dir)
best_accr, best_th, idx = verification(model, label_list, pair_list, transform=t)
print(f'txt_dir: {txt_dir}, best_accr: {best_accr}')

txt_dir = 'agedb30_child.txt'
print(f'working on : {txt_dir}')
pair_list, label_list = control_text_list(txt_root, txt_dir)
best_accr, best_th, idx = verification(model, label_list, pair_list, transform=t)
print(f'txt_dir: {txt_dir}, best_accr: {best_accr}')

txt_dir = 'fgnet30_child.txt'
print(f'working on : {txt_dir}')
pair_list, label_list = control_text_list(txt_root, txt_dir)
best_accr, best_th, idx = verification(model, label_list, pair_list, transform=t)
print(f'txt_dir: {txt_dir}, best_accr: {best_accr}')

# txt_dir = 'agedb10_20_child.txt'
# print(f'working on : {txt_dir}')
# pair_list, label_list = control_text_list(txt_root, txt_dir)
# best_accr, best_th, idx = verification(model, label_list, pair_list, transform=t)
# print(f'txt_dir: {txt_dir}, best_accr: {best_accr}')
#
# txt_dir = 'fgnet10_20_child.txt'
# print(f'working on : {txt_dir}')
# pair_list, label_list = control_text_list(txt_root, txt_dir)
# best_accr, best_th, idx = verification(model, label_list, pair_list, transform=t)
# print(f'txt_dir: {txt_dir}, best_accr: {best_accr}')
#
# txt_dir = 'agedb20_30_child.txt'
# print(f'working on : {txt_dir}')
# pair_list, label_list = control_text_list(txt_root, txt_dir)
# best_accr, best_th, idx = verification(model, label_list, pair_list, transform=t)
# print(f'txt_dir: {txt_dir}, best_accr: {best_accr}')
#
# txt_dir = 'fgnet20_30_child.txt'
# print(f'working on : {txt_dir}')
# pair_list, label_list = control_text_list(txt_root, txt_dir)
# best_accr, best_th, idx = verification(model, label_list, pair_list, transform=t)
# print(f'txt_dir: {txt_dir}, best_accr: {best_accr}')

txt_dir = 'lag.txt'
print(f'working on : {txt_dir}')
pair_list, label_list = control_text_list(txt_root, txt_dir)
best_accr, best_th, idx = verification(model, label_list, pair_list, transform=t)
print(f'txt_dir: {txt_dir}, best_accr: {best_accr}')
#
# print('20gap datasets')
# txt_dir = 'fgnet20_child.txt'
# print(f'working on : {txt_dir}')
# pair_list, label_list = control_text_list(txt_root, txt_dir)
# best_accr, best_th, idx = verification(model, label_list, pair_list, transform=t)
# print(f'txt_dir: {txt_dir}, best_accr: {best_accr}')
#
# txt_dir = 'agedb20_child.txt'
# print(f'working on : {txt_dir}')
# pair_list, label_list = control_text_list(txt_root, txt_dir)
# best_accr, best_th, idx = verification(model, label_list, pair_list, transform=t)
# print(f'txt_dir: {txt_dir}, best_accr: {best_accr}')
#
# print('cross age datasets....')
# txt_dir = 'cacd_vs.txt'
# print(f'working on : {txt_dir}')
# pair_list, label_list = control_text_list(txt_root, txt_dir, data_dir='cacd_vs')
# best_accr, best_th, idx = verification(model, label_list, pair_list, transform=t, data_dir='cacd_vs')
# print(f'txt_dir: {txt_dir}, best_accr: {best_accr}')
#
# txt_dir = 'morph.txt'
# print(f'working on : {txt_dir}')
# pair_list, label_list = control_text_list(txt_root, txt_dir, data_dir='morph')
# best_accr, best_th, idx = verification(model, label_list, pair_list, transform=t, data_dir='morph')
# print(f'txt_dir: {txt_dir}, best_accr: {best_accr}')
#
# txt_dir = 'fgnet_normal_3000.txt'
# print(f'working on : {txt_dir}')
# pair_list, label_list = control_text_list(txt_root, txt_dir)
# best_accr, best_th, idx = verification(model, label_list, pair_list, transform=t)
# print(f'txt_dir: {txt_dir}, best_accr: {best_accr}')

# txt_dir = 'agedb_normal_3000.txt'
# print(f'working on : {txt_dir}')
# pair_list, label_list = control_text_list(txt_root, txt_dir)
# best_accr, best_th, idx = verification(model, label_list, pair_list, transform=t)
# print(f'txt_dir: {txt_dir}, best_accr: {best_accr}')

#
# print('general face recognition starts...')
# dataset_root = os.path.join('/home/nas1_userE/jungsoolee/Face_dataset/faces_emore')
# print('evaluating lfw')
# lfw, lfw_issame = get_val_pair(dataset_root, 'lfw')
# accuracy, best_threshold, roc_curve_tensor, dist = learner.evaluate(args, lfw, lfw_issame, nrof_folds=10, tta=True)
# print('lfw - accuracy:{}, threshold:{}'.format(accuracy, best_threshold))
#
# print('evaluating agedb_30')
# agedb_30, agedb_30_issame = get_val_pair(dataset_root, 'agedb_30')
# accuracy, best_threshold, roc_curve_tensor, _ = learner.evaluate(args, agedb_30, agedb_30_issame, nrof_folds=10, tta=True)
# print('agedb_30 - accuracy:{}, threshold:{}'.format(accuracy, best_threshold))
#
# print('evaluating calfw')
# calfw, calfw_issame = get_val_pair(dataset_root, 'calfw')
# accuracy, best_threshold, roc_curve_tensor, dist = learner.evaluate(args, calfw, calfw_issame, nrof_folds=10, tta=True)
# print('calfw - accuracy:{}, threshold:{}'.format(accuracy, best_threshold))
#
# cfp_fp, cfp_fp_issame = get_val_pair(dataset_root, 'cfp_fp')
# accuracy, best_threshold, roc_curve_tensor, _ = learner.evaluate(args, cfp_fp, cfp_fp_issame, nrof_folds=10, tta=True)
# print('cfp_fp - accuracy:{}, threshold:{}'.format(accuracy, best_threshold))
# # trans.ToPILImage()(roc_curve_tensor)
#
# cplfw, cplfw_issame = get_val_pair(dataset_root, 'cplfw')
# accuracy, best_threshold, roc_curve_tensor, _ = learner.evaluate(args, cplfw, cplfw_issame, nrof_folds=10, tta=True)
# print('cplfw - accuracy:{}, threshold:{}'.format(accuracy, best_threshold))
# trans.ToPILImage()(roc_curve_tensor)
#
#







################################################### NO NEED ###################################################

# cfp_ff, cfp_ff_issame = get_val_pair(conf.emore_folder, 'cfp_ff')
# accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, cfp_ff, cfp_ff_issame, nrof_folds=10, tta=True)
# print('cfp_ff - accuracy:{}, threshold:{}'.format(accuracy, best_threshold))
# trans.ToPILImage()(roc_curve_tensor)

# vgg2_fp, vgg2_fp_issame = get_val_pair(conf.emore_folder, 'vgg2_fp')
# accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, vgg2_fp, vgg2_fp_issame, nrof_folds=10, tta=True)
# print('vgg2_fp - accuracy:{}, threshold:{}'.format(accuracy, best_threshold))
# # trans.ToPILImage()(roc_curve_tensor)