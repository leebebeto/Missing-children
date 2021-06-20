import torch.nn as nn
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.nn import Parameter
import os
import numpy as np
from dal_utils import accuracy
from collections import namedtuple
import math

#
# def l2_norm(input,axis=1):
#     norm = torch.norm(input,2,axis,True)
#     output = torch.div(input, norm)
#     return output

# class OECNN_Backbone(nn.Module):
#     def __init__(self, init_method, drop_rate=0):
#         super().__init__()
#         mods = nn.ModuleList()
#         mods.append(BN_Conv_ReLU(3, 64))
#         mods.append(BN_Conv_ReLU(64, 64))
#         mods.append(nn.MaxPool2d(kernel_size=2))
#         for _ in range(3):
#             mods.append(Block(64, 64))
#         mods.append(BN_Conv_ReLU(64, 128))
#         mods.append(nn.MaxPool2d(kernel_size=2))
#         for _ in range(4):
#             mods.append(Block(128, 128))
#         mods.append(BN_Conv_ReLU(128, 256))
#         mods.append(nn.MaxPool2d(kernel_size=2))
#         for _ in range(10):
#             mods.append(Block(256, 256))
#         mods.append(BN_Conv_ReLU(256, 512))
#         mods.append(nn.MaxPool2d(kernel_size=2))
#         for _ in range(3):
#             mods.append(Block(512, 512))
#         BN = nn.BatchNorm2d(512)
#         flatten = nn.Flatten()
#         dropout = nn.Dropout(drop_rate)
#         # fc = nn.Linear(512*7*6, 512)
#         fc = nn.Linear(512*7*7, 512)
#         final_BN = nn.BatchNorm1d(512)
#         self.seq = nn.Sequential(*mods, BN, flatten, dropout, fc, final_BN)
#         # initialize paras
#         if init_method is not None:
#             for mod in self.seq.modules():
#                 if isinstance(mod, nn.Conv2d):
#                     init_method['method'](mod.weight, **init_method['paras'])
#                     if mod.bias is not None:
#                         mod.bias.data.fill_(0)
#                 elif isinstance(mod, nn.BatchNorm2d):
#                     nn.init.constant_(mod.weight, 1)
#                 elif isinstance(mod, nn.Linear):
#                     init_method['method'](mod.weight, **init_method['paras'])
#                     if mod.bias is not None:
#                         mod.bias.data.fill_(0)
#         #time.sleep(1000)
#     def __str__(self):
#         return 'OECNN'
#
#     def forward(self, x):
#         return self.seq(x).renorm(2,0,1e-5).mul(1e5)
#
# def BN_Conv_ReLU(n_in, n_out, kernel_size=3, stride=1, padding=1, bias=False):
#     return nn.Sequential(nn.BatchNorm2d(n_in), nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
#                          nn.ReLU(inplace=True))
#
# class Block(nn.Module):
#     '''
#     Basic building block of OECNN (OECNN_Backbone), originally introduced in OECNN paper.
#     '''
#     def __init__(self, n_in, n_out, filter_sz=3, stride=1, block_sz=3):
#         super().__init__()
#         self.n_in = n_in
#         self.n_out = n_out
#         self.filter_sz = filter_sz
#         self.stride = stride
#         self.block_sz = block_sz
#         mods = []
#         mods.extend([nn.BatchNorm2d(n_in), nn.Conv2d(n_in, n_out, kernel_size=filter_sz, stride=self.stride, padding=1, bias=True), nn.ReLU(inplace=True)])
#         for _ in range(block_sz-1):
#             mods.extend([nn.BatchNorm2d(n_out), nn.Conv2d(n_out, n_out, kernel_size=filter_sz, stride=self.stride, padding=1, bias=True), nn.ReLU(inplace=True)])
#         self.seq = nn.Sequential(*mods)
#
#     def forward(self, x):
#         return self.seq(x) + x
#


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False), nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False), nn.BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR_SE(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks


class Backbone(nn.Module):
    def __init__(self, num_layers, drop_ratio, mode='ir'):
        super(Backbone, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(64))
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(drop_ratio),
                                          Flatten(),
                                          nn.Linear(512 * 7 * 7, 512),
                                          nn.BatchNorm1d(512))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x
        # return l2_norm(x)


class ArcfaceMargin(nn.Module):
    def __init__(self, n_cls, embedding_size, s=64., m=0.3):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(n_cls, embedding_size))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, xs, ys):
        logits = F.linear(F.normalize(xs), F.normalize(self.weight))
        if not self.training:
            return logits
        return logits.scatter(1, ys.view(-1, 1), (logits.gather(1, ys.view(-1, 1)).acos() + self.m).cos()).mul(self.s)

class CosMargin_v2(nn.Module):
    '''
    Loss defined in Cosface paper.
    '''
    def __init__(self, classnum, embedding_size=512,  s=64., m=0.35):
        super().__init__()
        self.m = m
        self.s = s
        self.fc = nn.Linear(embedding_size, classnum, bias=False)
        nn.init.xavier_uniform_(self.fc.weight)
    
    def forward(self, xs, labels):
        coses = self.fc(xs.renorm(2, 0, 1e-5).mul(1e5))
        if not self.training:
            return coses
        return coses.scatter_add(1, labels.view(-1,1), coses.new_full(labels.view(-1,1).size(), -self.m)).mul(self.s)

class CosMargin(nn.Module):
    '''
    Loss defined in Cosface paper.
    '''
    def __init__(self, classnum, embedding_size=512,  s=64., m=0.35):
        super().__init__()
        self.m = m
        self.s = s
        self.weight = nn.Parameter(torch.Tensor(classnum, embedding_size))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, xs, labels):
        coses = F.linear(F.normalize(xs), F.normalize(self.weight))
        if not self.training:
            return coses
        return coses.scatter_add(1, labels.view(-1,1), coses.new_full(labels.view(-1,1).size(), -self.m)).mul(self.s)

class RFM(nn.Module):
    '''
    Residual Factorization Module in paper.
    '''
    def __init__(self, n_in):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(n_in, n_in)
                                , nn.ReLU(inplace=True)
                                , nn.Linear(n_in, n_in)
                                , nn.ReLU(inplace=True))
    
    def forward(self, xs):
        return self.seq(xs)


class DAL_regularizer(nn.Module):
    '''
    Decorrelated Adversarial Learning module in paper.
    '''
    def __init__(self, n_in):
        super().__init__()
        self.w_age = nn.Linear(n_in, 1, bias=False)
        self.w_id = nn.Linear(n_in, 1, bias=False)
    
    def forward(self, features_age, features_id):
        vs_age = self.w_age(features_age)
        vs_id = self.w_id(features_id)
        rho = ((vs_age - vs_age.mean(dim=0)) * (vs_id - vs_id.mean(dim=0))).mean(dim=0).pow(2) \
                / ( (vs_age.var(dim=0) + 1e-6) * (vs_id.var(dim=0) + 1e-6))
        return rho




class DAL_model(nn.Module):
    '''
    The final ensemble model for training.
    '''
    def __init__(self, head, n_cls, embedding_size=512, init_method={'method': nn.init.kaiming_normal_, 'paras':{}}, conf=None):
        super().__init__()
        # self.backbone = OECNN_Backbone(init_method)
        self.backbone = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
        if head.lower() in 'cosface':
            self.margin_fc = CosMargin(n_cls, embedding_size=512, s=64.,m=0.35)  # 32 0.1 worked
        elif head.lower() in 'arcface':
            self.margin_fc = ArcfaceMargin(n_cls, embedding_size)
        self.DAL = DAL_regularizer(embedding_size)
        self.RFM = RFM(embedding_size)
        self.age_classifier = nn.Sequential(nn.Linear(embedding_size, embedding_size) \
                                            , nn.ReLU(inplace=True)
                                            , nn.Linear(embedding_size, embedding_size)
                                            , nn.ReLU(inplace=True)
                                            , nn.Linear(embedding_size, 8))
        self.id_cr = nn.CrossEntropyLoss()
        self.age_cr = nn.CrossEntropyLoss()

    def forward(self, xs, ys=None, agegrps=None, emb=False):
        # 512-D embedding
        embs = self.backbone(xs)
        embs_age = self.RFM(l2_norm(embs))
        embs_id = (embs - embs_age)
        if emb:
            # return embs
            return l2_norm(embs_id)
        # ID identifier
        logits = self.margin_fc(embs_id, ys)
        id_acc = accuracy(torch.max(logits, dim=1)[1], ys)
        idLoss = self.id_cr(logits, ys)
        # age classifier
        age_logits = self.age_classifier(embs_age)
        age_acc = accuracy(torch.max(age_logits, dim=1)[1], agegrps)
        ageLoss = self.age_cr(age_logits, agegrps)
        cano_cor = self.DAL(embs_age, embs_id)
        return idLoss, id_acc, ageLoss, age_acc, cano_cor



class OECNN_model(nn.Module):
    '''
    The final ensemble model for training.
    '''
    def __init__(self, head, n_cls, embedding_size=512, init_method={'method': nn.init.kaiming_normal_, 'paras':{}}, conf=None):
        super().__init__()
        self.backbone = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
        if head.lower() in 'cosface':
            self.margin_fc = CosMargin(n_cls, embedding_size=512, s=64.,m=0.35)  # 32 0.1 worked
        elif head.lower() in 'arcface':
            self.margin_fc = ArcfaceMargin(n_cls, embedding_size=512, s=64.,m=0.5)
        self.age_classifier = nn.Sequential(nn.Linear(1, 1))
        self.id_cr = nn.CrossEntropyLoss()
        self.age_cr = nn.MSELoss()

    def forward(self, xs, ys=None, agegrps=None, emb=False):
        # 512-D embedding
        embs = self.backbone(xs)
        age_input = embs.norm(dim=1, p=2).unsqueeze(1)
        age_logits = self.age_classifier(age_input)
        id_logits = self.margin_fc(l2_norm(embs), ys)
        if emb:
            return l2_norm(embs)
        id_acc = accuracy(torch.max(id_logits, dim=1)[1], ys)
        idLoss = self.id_cr(id_logits, ys)

        # age classifier
        agegrps = agegrps.unsqueeze(1)
        age_acc = accuracy(age_logits, agegrps)
        # import pdb; pdb.set_trace()
        # age_label = torch.zeros((xs.shape[0], 100)).cuda()
        # age_label[torch.arange(xs.shape[0]), agegrps] = 1
        ageLoss = self.age_cr(age_logits, agegrps.float())

        return idLoss, id_acc, ageLoss, age_acc
