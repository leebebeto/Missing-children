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
parser.add_argument("--model_path", help="evaluate model path", default='ours/fgnetc_best_model_2021-06-01-12-15_accuracy:0.860_step:226000_casia_CASIA_POSITIVE_ZERO_05_MILE_3.pth', type=str)
parser.add_argument("--device", help="device", default='cuda', type=str)
parser.add_argument("--embedding_size", help='embedding_size', default=512, type=int)
parser.add_argument("--wandb", help="whether to use wandb", action='store_true')
parser.add_argument("--epochs", help="num epochs", default=50, type=int)
parser.add_argument("--batch_size", help="batch_size", default=64, type=int)
parser.add_argument("--loss", help="loss", default='Arcface', type=str)
parser.add_argument("--test_dir", help="test dir", default='fgnet20', type=str)
args = parser.parse_args()

# conf = get_config(training=False)
learner = face_learner(args, inference=True)
# save_path = '/home/nas1_temp/jooyeolyun/mia_params/'

if args.loss == 'OECNN' or args.loss == 'DAL':
    from Backbone import *
else:
    from model import *


# learner.load_state(conf, 'ir_se50.pth', model_only=True, from_save_folder=True)
# model_path = os.path.join(save_path, args.model_path)
if args.loss == 'DAL':
    learner.model = DAL_model(head='cosface', n_cls= 10572, conf=args).to(args.device)
    model_path = os.path.join(args.model_path)
    learner.model.load_state_dict(torch.load(model_path))
    learner.model.eval()

elif args.loss == 'OECNN':
    learner.model = OECNN_model(head='cosface', n_cls= 10572, conf=args).to(args.device)
    model_path = os.path.join(args.model_path)
    learner.model.load_state_dict(torch.load(model_path))
    learner.model.eval()

else:
    model_path = os.path.join(args.model_path)
    learner.load_state(args, model_path = model_path)
    before_time = time.time()

import os
from PIL import Image
import random
import pickle
import torchvision.transforms as T # 이미지 전처리를 지원
import torch
from utils_txt import cos_dist, fixed_img_list
import numpy as np
import tqdm
import torch
# 데이터 관련 세팅
gray_scale = False

# Hyperparameter
feature_dim = 512

# GPU가 있을 경우 연산을 GPU에서 하고 없을 경우 CPU에서 진행
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
net_depth, drop_ratio, net_mode = 50, 0.6, 'ir_se'
if args.loss == 'DAL' or args.loss == 'OECNN':
    pass
else:
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



def verification(net, data_dict, transform):
    labels = []
    best_pred = None
    best_similarities = None

    top_dict = {1: 0.0, 5: 0.0, 10: 0.0}
    net.eval()
    with torch.no_grad():  # Test 때 GPU를 사용할 경우 메모리 절약을 위해 torch.no_grad() 내에서 하는 것이 좋다.
        for key, value in tqdm.tqdm(data_dict.items()):
            similarities = []
            path_1 = key
            query_image, gallery_image = None, None
            for path_2 in value:
                img_1 = t(Image.open(path_1)).unsqueeze(dim=0).to(dev)
                img_2 = t(Image.open(path_2)).unsqueeze(dim=0).to(dev)
                query_image =  img_1 if query_image == None else torch.cat((query_image, img_1), dim=0)
                gallery_image =  img_2 if gallery_image == None else torch.cat((gallery_image, img_2), dim=0)

            imgs = torch.cat((query_image, gallery_image), dim=0)
            features = None
            for i in range(0, imgs.shape[0], 64):
                if imgs.shape[0] - i > 64:
                    embedding = net(imgs[i: i +64])
                else:
                    embedding = net(imgs[i:])
                features = embedding if features == None else torch.cat((features, embedding))
            for i in range(int(imgs.shape[0]/2)):
                similarities.append(cos_dist(features[i], features[i+int(imgs.shape[0]/2)]).cpu())
            similarities = torch.stack(similarities)

            for k in [1, 5, 10]:
                top_sim, top_idx = torch.topk(similarities, k, dim=0, largest=True)
                if 0 in top_idx:
                    top_dict[k] += 1
    for k in [1, 5, 10]:
        top_dict[k] /= len(data_dict.keys())
        print(f'top {k} id acc: {top_dict[k]}')



def verification_dal(net, data_dict, transform):
    labels = []
    best_pred = None
    best_similarities = None

    top_dict = {1: 0.0, 5: 0.0, 10: 0.0}
    net.eval()
    with torch.no_grad():  # Test 때 GPU를 사용할 경우 메모리 절약을 위해 torch.no_grad() 내에서 하는 것이 좋다.
        for key, value in tqdm.tqdm(data_dict.items()):
            similarities = []
            path_1 = key
            query_image, gallery_image = None, None
            for path_2 in value:
                img_1 = t(Image.open(path_1)).unsqueeze(dim=0).to(dev)
                img_2 = t(Image.open(path_2)).unsqueeze(dim=0).to(dev)
                query_image =  img_1 if query_image == None else torch.cat((query_image, img_1), dim=0)
                gallery_image =  img_2 if gallery_image == None else torch.cat((gallery_image, img_2), dim=0)

            imgs = torch.cat((query_image, gallery_image), dim=0)
            features = None
            for i in range(0, imgs.shape[0], 64):
                if imgs.shape[0] - i > 64:
                    embedding = net(imgs[i: i +64], emb=True)
                else:
                    embedding = net(imgs[i:], emb=True)
                features = embedding if features == None else torch.cat((features, embedding))
            for i in range(int(imgs.shape[0]/2)):
                similarities.append(cos_dist(features[i], features[i+int(imgs.shape[0]/2)]).cpu())
            similarities = torch.stack(similarities)

            for k in [1, 5, 10]:
                top_sim, top_idx = torch.topk(similarities, k, dim=0, largest=True)
                if 0 in top_idx:
                    top_dict[k] += 1
    for k in [1, 5, 10]:
        top_dict[k] /= len(data_dict.keys())
        print(f'top {k} id acc: {top_dict[k]}')


import glob

# with open(f'/home/nas3_userL/jungsoolee/Face_dataset/txt_files/lag_identification.pickle', 'rb') as f:
#     data_dict = pickle.load(f)
#
# verification(model, data_dict, transform=t)
#
# print(f'working on : {args.test_dir}....')
if args.loss == 'OECNN' or args.loss == 'DAL':
    # with open(f'/home/nas3_userL/jungsoolee/Face_dataset/txt_files/{args.test_dir}_identification.pickle', 'rb') as f:
    #     data_dict = pickle.load(f)
    with open(f'/home/nas3_userL/jungsoolee/Face_dataset/txt_files/lag_identification.pickle', 'rb') as f:
        data_dict = pickle.load(f)

    verification_dal(learner.model, data_dict, transform=t)

else:
    # with open(f'/home/nas3_userL/jungsoolee/Face_dataset/txt_files/{args.test_dir}_identification.pickle', 'rb') as f:
    #     data_dict = pickle.load(f)
    with open(f'/home/nas3_userL/jungsoolee/Face_dataset/txt_files/lag_identification.pickle', 'rb') as f:
        data_dict = pickle.load(f)

    verification(model, data_dict, transform=t)

#
# print(f'working on : fgnetc....')
# with open('/home/nas3_userL/jungsoolee/Face_dataset/txt_files/fgnet20_identification.pickle', 'rb') as f:
#     data_dict = pickle.load(f)
#
# verification(model, data_dict, transform=t)
#
# print(f'working on : agedbc....')
# with open('/home/nas3_userL/jungsoolee/Face_dataset/txt_files/agedbc_identification.pickle', 'rb') as f:
#     data_dict = pickle.load(f)
#
# verification(model, data_dict, transform=t)


# print(f'working on : fgnetc20....')
# with open('/home/nas3_userL/jungsoolee/Face_dataset/txt_files/fgnetc20_identification.pickle', 'rb') as f:
#     data_dict = pickle.load(f)
#
# verification(model, data_dict, transform=t)

# print(f'working on : agedbc20....')
# with open('/home/nas3_userL/jungsoolee/Face_dataset/txt_files/agedbc20_identification.pickle', 'rb') as f:
#     data_dict = pickle.load(f)
#
# verification(model, data_dict, transform=t)

# print(f'working on : lag....')
# with open('/home/nas3_userL/jungsoolee/Face_dataset/txt_files/lag_identification.pickle', 'rb') as f:
#     data_dict = pickle.load(f)
#
# verification(model, data_dict, transform=t)


