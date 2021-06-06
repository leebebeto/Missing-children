import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms

import os
import random
import argparse
from tqdm import tqdm

from data.data_pipe import CASIADataset
from model import BackboneMaruta, Arcface

import pdb


def main(args):

    random_seed = 4885
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    train_dataset = CASIADataset('/home/nas1_userE/jungsoolee/Face_dataset/CASIA_REAL_NATIONAL', train_transforms=train_transform,conf=args)
    class_num = train_dataset.class_num
    print('casia loader generated')

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = BackboneMaruta(50, 0.6, 'ir_se')
    head = Arcface(embedding_size=512, classnum=class_num)

    path_root = '/home/nas1_temp/jooyeolyun/mia_params/gradcam'

    # head_path = 'fgnetc_best_head_2021-06-01-09-57_accuracy:0.778_step:64000_casia_ARCFACE.pth'
    # model_path = 'fgnetc_best_model_2021-06-01-09-57_accuracy:0.778_step:64000_casia_ARCFACE.pth'
    # head_path = 'fgnetc_best_head_2021-06-01-17-39_accuracy:0.830_step:148000_casia_ARCFACE.pth'
    # model_path = 'fgnetc_best_model_2021-06-01-17-39_accuracy:0.830_step:148000_casia_ARCFACE.pth'
    head_path = 'fgnetc_best_head_2021-06-01-21-00_accuracy:0.848_step:186000_casia_ARCFACE.pth'
    model_path = 'fgnetc_best_model_2021-06-01-21-00_accuracy:0.848_step:186000_casia_ARCFACE.pth'
    # head_path = None
    # model_path = None

    model_type = model_path.split(':')[1][2:5] # accuracy

    model.load_state_dict(torch.load(os.path.join(path_root, model_path)))
    head.load_state_dict(torch.load(os.path.join(path_root, head_path)))

    model = model.cuda()
    head = head.cuda()

    for idx, (data, labels, age) in tqdm(enumerate(train_loader), ncols=80, total=len(train_loader)):
        model.eval()
        data = torch.unsqueeze(data[0], dim=0)
        labels = torch.unsqueeze(labels[0], dim=0)

        data, labels = data.cuda(), labels.cuda()
            
        embeddings = model(data)
        thetas = head(embeddings, labels) # prototype through head

        pred = thetas.argmax(dim=1)
        thetas[:, labels].backward()
        gradients = model.get_gradient()
        
        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = model.get_activations(data).detach()

        # weight the channels by corresponding gradients
        num_c = pooled_gradients.shape[0]
        for i in range(num_c):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        heatmap = torch.clamp(heatmap, min=0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)
        heatmap = torch.unsqueeze(torch.unsqueeze(heatmap, dim=0), dim=0)
        heatmap = F.interpolate(heatmap, size=(112, 112), mode='bilinear', align_corners=True)

        save_image(heatmap[0], './analysis/{}_{}_gradcam.png'.format(model_type, idx))
        save_image(data[0], './analysis/{}_{}_img.png'.format(model_type, idx))
        pdb.set_trace()
        if idx>100:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("--new_id", help="dummy", default=100, type=int)
    args = parser.parse_args()

    main(args)
