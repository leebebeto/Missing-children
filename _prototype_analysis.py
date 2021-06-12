# from config import get_config
import argparse

from numpy.lib.index_tricks import fill_diagonal
from learner import face_learner
from data.data_pipe import get_val_pair
from torchvision import transforms
import torch

import time
import pdb
import os
import numpy as np
import glob
from PIL import Image

parser = argparse.ArgumentParser(description='for face verification')
parser.add_argument("--net_mode", help="which network, [ir, ir_se, mobilefacenet]",default='ir_se', type=str)
parser.add_argument("--net_depth", help="how many layers [50,100,152]", default=50, type=int)
parser.add_argument("--drop_ratio", help="ratio of drop out", default=0.6, type=float)
parser.add_argument("--model_path", help="evaluate model path", default='fgnetc_best_model_2021-06-01-12-15_accuracy:0.860_step:226000_casia_CASIA_POSITIVE_ZERO_05_MILE_3.pth', type=str)
parser.add_argument("--device", help="device", default='cuda', type=str)
parser.add_argument("--embedding_size", help='embedding_size', default=512, type=int)
parser.add_argument("--wandb", help="whether to use wandb", action='store_true')
parser.add_argument("--epochs", help="num epochs", default=50, type=int)
parser.add_argument("--loss", help="num epochs", default='Arcface', type=str)
parser.add_argument("--batch_size", help="batch_size", default=64, type=int)
args = parser.parse_args()

# conf = get_config(training=False)
learner = face_learner(args, inference=True, load_head=True)
save_path = '/home/nas1_temp/jooyeolyun/mia_params/baseline/'

# learner.load_state(conf, 'ir_se50.pth', model_only=True, from_save_folder=True)
model_path = os.path.join(save_path, 'fgnetc_best_model_2021-05-27-19-11_accuracy:0.842_step:119574_casia_arcface_baseline_64.pth')
head_path = os.path.join(save_path, 'fgnetc_best_head_2021-05-27-19-11_accuracy:0.842_step:119574_casia_arcface_baseline_64.pth')
learner.load_state(args, model_path = model_path, head_path=head_path)
learner.model.eval()

age_file = open('./dataset/casia-webface.txt').readlines()
id2age = {os.path.join(str(int(line.split(' ')[1].split('/')[1])), str(int(line.split(' ')[1].split('/')[2][:-4]))): float(line.split(' ')[2]) for line in age_file}
child_image2age = {os.path.join(str(int(line.split(' ')[1].split('/')[1])), str(int(line.split(' ')[1].split('/')[2][:-4]))): float(line.split(' ')[2]) for line in age_file if float(line.split(' ')[2]) < 13}
child_image2freq = {id.split('/')[0]: 0 for id in child_image2age.keys()}
for k, v in child_image2age.items():
    child_image2freq[k.split('/')[0]] += 1

child_identity_freq = {int(k): v for k, v in sorted(child_image2freq.items(), key=lambda item: item[1]) if v >= 10}
child_identity = list(child_identity_freq.keys())
print(f'child number: {len(child_identity)}')

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 1) child vs adult prototype
# for cls in child_identity:
#     child_image_temp = glob.glob(f'/home/nas1_userE/jungsoolee/Face_dataset/CASIA_REAL_NATIONAL/{cls}/*')
#     child_image = []
#     for image in child_image_temp:
#         id_img = '/'.join((image.split('/')[-2], image.split('/')[-1].split('_')[0]))[:-4]
#         try:
#             age = id2age[id_img]
#             if int(age) < 13:
#                 child_image.append(image)
#         except:
#             continue
#     batch, label = [], []
#     for image in child_image:
#         img = Image.open(image)
#         img = train_transform(img)
#         batch.append(img)
#         label.append(int(image.split('/')[-2]))
#     batch = torch.stack(batch).cuda()
#     label = torch.Tensor(label).cuda()
#
#     embedding = learner.model(batch)
#     kernel = learner.head.kernel
#     kernel_norm = l2_norm(kernel,axis=0)
#     cos_theta = torch.mm(embedding, kernel_norm)
#
#     child_flag = torch.zeros(cos_theta.shape[1])
#     child_flag[torch.Tensor(child_identity).long()]= 1
#     child_flag = (child_flag == 1)
#     adult_flag = ~child_flag
#
#     child_negative = cos_theta[:, child_flag].sum(dim=1) - cos_theta[:, cls]
#     adult_negative = cos_theta[:, adult_flag].sum(dim=1) - cos_theta[:, cls]
#
#     child_negative = child_negative.sum()/len(torch.where(child_flag==True)[0])
#     adult_negative = adult_negative.sum()/len(torch.where(adult_flag==True)[0])
#
#     print(f'child mean: {torch.rad2deg(torch.arccos(child_negative))} || adult mean: {torch.rad2deg(torch.arccos(adult_negative))}')

# # 2) child vs adult prototype within a class
# for cls in child_identity:
#     child_image_temp = glob.glob(f'/home/nas1_userE/jungsoolee/Face_dataset/CASIA_REAL_NATIONAL/{cls}/*')
#     child_image, adult_image = [], []
#     for image in child_image_temp:
#         id_img = '/'.join((image.split('/')[-2], image.split('/')[-1].split('_')[0]))[:-4]
#         try:
#             age = id2age[id_img]
#             if int(age) < 13:
#                 child_image.append(image)
#             else:
#                 adult_image.append(image)
#         except:
#             adult_image.append(image)

#     batch = []
#     for image in child_image:
#         img = Image.open(image)
#         img = train_transform(img)
#         batch.append(img)
#         label = int(image.split('/')[-2])
#     batch = torch.stack(batch).cuda()

#     import pdb; pdb.set_trace()
#     with torch.no_grad():
#         embedding = learner.model(batch)
#         kernel = learner.head.kernel
#         kernel_norm = l2_norm(kernel,axis=0)
#         cos_theta = torch.mm(embedding, kernel_norm)
#         cos_theta = cos_theta.clamp(-1,1)
#     # child_theta = cos_theta[:, cls].mean()
#     child_theta = torch.abs(torch.rad2deg(torch.arccos(cos_theta[:, cls])))
#     # child_theta= child_theta.mean()

#     batch = []
#     for image in adult_image:
#         img = Image.open(image)
#         img = train_transform(img)
#         batch.append(img)
#         label = int(image.split('/')[-2])
#     batch = torch.stack(batch).cuda()

#     with torch.no_grad():
#         embedding = learner.model(batch)
#         kernel = learner.head.kernel
#         kernel_norm = l2_norm(kernel,axis=0)
#         cos_theta = torch.mm(embedding, kernel_norm)
#         cos_theta = cos_theta.clamp(-1,1)
#     # adult_theta = cos_theta[:, cls].mean()
#     adult_theta = torch.abs(torch.rad2deg(torch.arccos(cos_theta[:, cls])))
#     # adult_theta= adult_theta.mean()
#     print(f'cls: {cls} || child mean: {child_theta} || adult mean: {adult_theta}')
#     # print(f'cls: {cls} || child mean: {torch.rad2deg(torch.arccos(child_theta))} || adult mean: {torch.rad2deg(torch.arccos(adult_theta))}')

# 3) inter-child similarity, inter-adult similarity
child_means = []
adult_means = []
for idx, cls in enumerate(child_identity):
    child_image_temp = glob.glob(f'/home/nas1_userE/jungsoolee/Face_dataset/CASIA_REAL_NATIONAL/{cls}/*')
    child_image, adult_image = [], []
    for image in child_image_temp:
        id_img = '/'.join((image.split('/')[-2], image.split('/')[-1].split('_')[0]))[:-4]
        try:
            age = id2age[id_img]
            if int(age) < 13:
                child_image.append(image)
            else:
                adult_image.append(image)
        except:
            adult_image.append(image)
    print('id {}: child-{}, adult-{}'.format(idx, len(child_image), len(adult_image)))
    if len(adult_image) < 10:
        print('excluding id {}'.format(idx))
        continue
    # child_image = child_image[:10]
    # adult_image = adult_image[:10]

    batch = []
    for image in child_image:
        img = Image.open(image)
        img = train_transform(img)
        batch.append(img)
        label = int(image.split('/')[-2])
    batch = torch.stack(batch).cuda()

    with torch.no_grad():
        embedding = learner.model(batch)
        # kernel = learner.head.kernel
        # kernel_norm = l2_norm(kernel,axis=0)
        # cos_theta = torch.mm(embedding, kernel_norm)
        # cos_theta = cos_theta.clamp(-1,1)
    euc_mean = torch.mean(embedding, dim=0)
    child_means.append(torch.div(euc_mean, torch.norm(euc_mean, keepdim=True)))


    # child_theta = torch.abs(torch.rad2deg(torch.arccos(cos_theta[:, cls])))
    # child_theta= child_theta.mean()

    batch = []
    for image in adult_image:
        img = Image.open(image)
        img = train_transform(img)
        batch.append(img)
        label = int(image.split('/')[-2])
    batch = torch.stack(batch).cuda()

    with torch.no_grad():
        embedding = learner.model(batch)
        # kernel = learner.head.kernel
        # kernel_norm = l2_norm(kernel,axis=0)
        # cos_theta = torch.mm(embedding, kernel_norm)
        # cos_theta = cos_theta.clamp(-1,1)
    # adult_theta = torch.abs(torch.rad2deg(torch.arccos(cos_theta[:, cls])))
    euc_mean = torch.mean(embedding, dim=0)
    adult_means.append(torch.div(euc_mean, torch.norm(euc_mean, keepdim=True)))

    # if idx ==100:
    #     break
    
import pdb; pdb.set_trace()
print('total of {} ids selected'.format(len(child_means)))
child_means = torch.stack(child_means)
adult_means = torch.stack(adult_means)

inter_child_sim = torch.mm(child_means, child_means.T).fill_diagonal_(0)
inter_adult_sim = torch.mm(adult_means, adult_means.T).fill_diagonal_(0)

inter_child_sum = torch.mean(inter_child_sim)
inter_adult_sum = torch.mean(inter_adult_sim)

print(inter_child_sum.item())
print(inter_adult_sum.item())

    # print(f'cls: {cls} || child mean: {child_theta} || adult mean: {adult_theta}')
    # print(f'cls: {cls} || child mean: {torch.rad2deg(torch.arccos(child_theta))} || adult mean: {torch.rad2deg(torch.arccos(adult_theta))}')