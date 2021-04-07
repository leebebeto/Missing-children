import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from tensorboardX import SummaryWriter
import pandas as pd

from tqdm import tqdm
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from PIL import Image
import pickle
import math
import bcolz
import os
import glob
from data.data_pipe import get_train_loader, get_train_loader_d
from model import Backbone, Arcface, MobileFaceNet, l2_norm, GrowUP, Discriminator
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from verifacation import evaluate, evaluate_child

import pdb
import time

class face_learner(object):
    def __init__(self, conf, inference=False):

        if conf.use_mobilfacenet:
            self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
            print('MobileFaceNet model generated')
        else:
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
            self.growup = GrowUP().to(conf.device)
            self.discriminator = Discriminator().to(conf.device)
            print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))

        if not inference:

            self.milestones = conf.milestones
            self.loader, self.class_num = get_train_loader(conf)
            if conf.discriminator:
                self.child_loader, self.adult_loader = get_train_loader_d(conf)

            os.makedirs(conf.log_path, exist_ok=True)
            self.writer = SummaryWriter(conf.log_path)
            self.step = 0

            self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)
            
            # Will not use anymore
            if conf.use_dp:
                self.model = nn.DataParallel(self.model)
                self.head = nn.DataParallel(self.head)

            print(self.class_num)
            print(conf)

            print('two model heads generated')

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)
            
            if conf.use_mobilfacenet:
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                                    {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)
            else:
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)
            if conf.discriminator:
                self.optimizer_g = optim.Adam(self.growup.parameters(), lr = 1e-4, betas=(0.5,0.999))
                self.optimizer_g2 = optim.Adam(self.growup.parameters(), lr = 1e-4, betas=(0.5,0.999))
                self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr = 1e-4, betas=(0.5, 0.999))

            if conf.finetune_model_path is not None:
                self.optimizer = optim.SGD([
                                        {'params': paras_wo_bn, 'weight_decay': 5e-4},
                                        {'params': paras_only_bn}
                                    ], lr = conf.lr, momentum = conf.momentum)
            print('optimizers generated')

            self.board_loss_every = len(self.loader)//100
            self.evaluate_every = len(self.loader)//2
            self.save_every = len(self.loader)

            dataset_root = "/home/nas1_userD/yonggyu/Face_dataset/face_emore"
            self.lfw = np.load(os.path.join(dataset_root, "lfw_align_112_list.npy")).astype(np.float32)
            self.lfw_issame = np.load(os.path.join(dataset_root, "lfw_align_112_label.npy"))
            self.fgnetc = np.load(os.path.join(dataset_root, "FGNET_new_align_list.npy")).astype(np.float32)
            self.fgnetc_issame = np.load(os.path.join(dataset_root, "FGNET_new_align_label.npy"))
        else:
            # Will not use anymore
            # self.model = nn.DataParallel(self.model)
            self.threshold = conf.threshold


    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)
   
        
    def evaluate(self, conf, carray, issame, nrof_folds = 10, tta = True):
        self.model.eval()
        self.growup.eval()
        self.discriminator.eval()

        idx = 0
        embeddings = np.zeros([len(carray), conf.embedding_size])
        with torch.no_grad():
            while idx + conf.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + conf.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)).cpu() + self.model(fliped.to(conf.device)).cpu()
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch).cpu()
                else:
                    embeddings[idx:idx + conf.batch_size] = self.model(batch.to(conf.device)).cpu()
                idx += conf.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])            
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)).cpu() + self.model(fliped.to(conf.device)).cpu()
                    embeddings[idx:] = l2_norm(emb_batch).cpu()
                else:
                    embeddings[idx:] = self.model(batch.to(conf.device)).cpu()
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = transforms.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor
    
    def evaluate_child(self, conf, carray, issame, nrof_folds=10, tta=True):
        self.model.eval()
        self.growup.eval()
        self.discriminator.eval()
        idx = 0
        embeddings1 = np.zeros([len(carray)//2, conf.embedding_size])
        embeddings2 = np.zeros([len(carray)//2, conf.embedding_size])

        carray1 = carray[::2, ]
        carray2 = carray[1::2, ]

        with torch.no_grad():
            while idx + conf.batch_size <= len(carray1):
                batch = torch.tensor(carray1[idx:idx + conf.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.growup(self.model(batch.to(conf.device))).cpu() + \
                                self.growup(self.model(fliped.to(conf.device))).cpu()
                    embeddings1[idx:idx + conf.batch_size] = l2_norm(emb_batch).cpu()
                else:
                    embeddings1[idx:idx + conf.batch_size] = self.growup(self.model(batch.to(
                        conf.device))).cpu()
                idx += conf.batch_size
            if idx < len(carray1):
                batch = torch.tensor(carray1[idx:])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.growup(self.model(batch.to(conf.device))).cpu() + \
                                self.growup(self.model(fliped.to(conf.device))).cpu()
                    embeddings1[idx:] = l2_norm(emb_batch).cpu()
                else:
                    embeddings1[idx:] = self.growup(self.model(batch.to(conf.device))).cpu()

            while idx + conf.batch_size <= len(carray2):
                batch = torch.tensor(carray2[idx:idx + conf.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)).cpu() + \
                                self.model(fliped.to(conf.device)).cpu()
                    embeddings2[idx:idx + conf.batch_size] = l2_norm(emb_batch).cpu()
                else:
                    embeddings2[idx:idx + conf.batch_size] = self.model(batch.to(conf.device)).cpu()
                idx += conf.batch_size
            if idx < len(carray2):
                batch = torch.tensor(carray2[idx:])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)).cpu() + \
                                self.model(fliped.to(conf.device)).cpu()
                    embeddings2[idx:] = l2_norm(emb_batch).cpu()
                else:
                    embeddings2[idx:] = self.model(batch.to(conf.device)).cpu()

        tpr, fpr, accuracy, best_thresholds = evaluate_child(embeddings1, embeddings2, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = transforms.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

    def zero_grad(self):
        self.optimizer.zero_grad()
        self.optimizer_g.zero_grad()
        self.optimizer_d.zero_grad()

    def train(self, conf, epochs):
        self.model.train()
        running_loss = 0.
        for e in range(epochs):
            print('epoch {} started'.format(e))

            if e in self.milestones:
                self.schedule_lr()

            for imgs, labels, ages in tqdm(iter(self.loader)):

                self.optimizer.zero_grad()

                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)

                embeddings = self.model(imgs)
                thetas = self.head(embeddings, labels)

                loss = conf.ce_loss(thetas, labels)
                loss.backward()
                running_loss += loss.item()

                self.optimizer.step()
                
                if self.step % self.board_loss_every == 0 and self.step != 0: # XXX
                    print('tensorboard plotting....')
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    running_loss = 0.
                
                if self.step % self.evaluate_every == 0 and self.step != 0:
                    print('evaluating....')
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.lfw, self.lfw_issame)
                    self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor)
                    accuracy2, best_threshold2, roc_curve_tensor2 = self.evaluate(conf, self.lfw, self.lfw_issame)
                    self.board_val('fgent_c', accuracy2, best_threshold2, roc_curve_tensor2)

                    self.model.train()

                if self.step % self.save_every == 0 and self.step != 0:
                    print('saving model....')
                    # save with most recently calculated accuracy?
                    if conf.finetune_model_path is not None:
                        self.save_state(conf, accuracy2, extra=str(conf.data_mode) + '_' + str(conf.net_depth) \
                            + '_' + str(conf.batch_size) + 'finetune')
                    else:
                        self.save_state(conf, accuracy2, extra=str(conf.data_mode) + '_' + str(conf.net_depth) \
                            + '_' + str(conf.batch_size))

                self.step += 1
        if conf.finetune_model_path is not None:
            self.save_state(conf, accuracy, to_save_folder=True, extra=str(conf.data_mode)  + '_' + str(conf.net_depth) \
                + '_'+ str(conf.batch_size) +'_finetune')
        else:
            self.save_state(conf, accuracy, to_save_folder=True, extra=str(conf.data_mode)  + '_' + str(conf.net_depth) \
                + '_'+ str(conf.batch_size) +'_final')


    def train_age_invariant(self, conf, epochs):
        '''
        Our method
        '''
        self.model.train()
        running_loss = 0.
        l1_loss = 0
        for e in range(epochs):
            print('epoch {} started'.format(e))

            if e in self.milestones:
                self.schedule_lr()

            a_loader = iter(self.adult_loader)
            c_loader = iter(self.child_loader)
            for imgs, labels, ages in tqdm(iter(self.loader)):
                # loader : base loader that returns images with id
                # a_loader, c_loader : adult, child loader with same datasize
                # ages : 0 == child, 1== adult
                try:
                    imgs_a, labels_a = next(a_loader)
                    imgs_c, labels_c = next(c_loader)
                except StopIteration:
                    a_loader = iter(self.adult_loader)
                    c_loader = iter(self.child_loader)
                    imgs_a, labels_a = next(a_loader)
                    imgs_c, labels_c = next(c_loader)
                
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)
                imgs_a, labels_a = imgs_a.to(conf.device), labels_a.to(conf.device).type(torch.float32)
                imgs_c, labels_c = imgs_c.to(conf.device), labels_c.to(conf.device).type(torch.float32)
                bs_a = imgs_a.shape[0]

                imgs_ac = torch.cat([imgs_a, imgs_c], dim=0)

                ###########################
                #       Train head        #
                ###########################
                self.optimizer.zero_grad()
                self.optimizer_g2.zero_grad()
                self.growup.train()

                c = (ages == 0) # select children for enhancement

                embeddings = self.model(imgs)

                if sum(c) > 1: # there might be no childern in loader's batch
                    embeddings_c = embeddings[c]
                    embeddings_a_hat = self.growup(embeddings_c)
                    embeddings[c] = embeddings_a_hat
                elif sum(c) == 1:
                    self.growup.eval()
                    embeddings_c = embeddings[c]
                    embeddings_a_hat = self.growup(embeddings_c)
                    embeddings[c] = embeddings_a_hat

                thetas = self.head(embeddings, labels)

                loss = conf.ce_loss(thetas, labels)
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()
                self.optimizer_g2.step()

                ##############################
                #    Train discriminator     #
                ##############################
                self.optimizer_d.zero_grad()
                self.growup.train()
                _embeddings = self.model(imgs_ac)
                embeddings_a, embeddings_c = _embeddings[:bs_a], _embeddings[bs_a:]

                embeddings_a_hat = self.growup(embeddings_c)
                embeddings_ac = torch.cat([embeddings_a, embeddings_a_hat], dim=0)
                labels_ac = torch.cat([labels_a, labels_c], dim=0)
                pred_ac = torch.squeeze(self.discriminator(embeddings_ac))
                d_loss = conf.ls_loss(pred_ac, labels_ac)
                d_loss.backward()
                self.optimizer_d.step()

                #############################
                #      Train genertator     #
                #############################
                self.optimizer_g.zero_grad()
                embeddings_c = self.model(imgs_c)
                embeddings_a_hat = self.growup(embeddings_c)
                pred_c = torch.squeeze(self.discriminator(embeddings_c))
                labels_a = torch.ones_like(labels_c, dtype=torch.float)
                # generator should make child 1
                g_loss = conf.ls_loss(pred_c, labels_a)
                
                l1_loss = conf.l1_loss(embeddings_a_hat, embeddings_c)
                g_total_loss = g_loss + l1_loss
                g_total_loss.backward()
                # g_loss.backward()
                self.optimizer_g.step()

                if self.step % self.board_loss_every == 0 and self.step != 0: # XXX
                    print('tensorboard plotting....')
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    self.writer.add_scalar('d_loss', d_loss, self.step)
                    self.writer.add_scalar('g_loss', g_loss, self.step)
                    self.writer.add_scalar('l1_loss', l1_loss, self.step)
                    running_loss = 0.
                
                if self.step % self.evaluate_every == 0 and self.step != 0:
                    print('evaluating....')
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.lfw, self.lfw_issame)
                    self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor)
                    accuracy2, best_threshold2, roc_curve_tensor2 = self.evaluate_child(conf, self.fgnetc, self.fgnetc_issame)
                    self.board_val('fgent_c', accuracy2, best_threshold2, roc_curve_tensor2)

                    self.model.train()

                if self.step % self.save_every == 0 and self.step != 0:
                    print('saving model....')
                    # save with most recently calculated accuracy?
                    self.save_state(conf, accuracy2, extra=str(conf.data_mode) + '_' + str(conf.net_depth) \
                        + '_' + str(conf.batch_size) + conf.model_name)

                self.step += 1
        self.save_state(conf, accuracy2, to_save_folder=True, extra=str(conf.data_mode)  + '_' + str(conf.net_depth)\
             + '_'+ str(conf.batch_size) +'_discriminator_final')


    def analyze_angle(self, conf, name):
        '''
        Only works on age labeled vgg dataset, agedb dataset
        '''

        angle_table = [{0:set(), 1:set(), 2:set(), 3:set(), 4:set(), 5:set(), 6:set(), 7:set()} for i in range(self.class_num)]
        # batch = 0
        # _angle_table = torch.zeros(self.class_num, 8, len(self.loader)//conf.batch_size).to(conf.device)
        if conf.resume_analysis:
            self.loader = []
        for imgs, labels, ages in tqdm(iter(self.loader)):

            imgs = imgs.to(conf.device)
            labels = labels.to(conf.device)
            ages = ages.to(conf.device)

            embeddings = self.model(imgs)
            if conf.use_dp:
                kernel_norm = l2_norm(self.head.module.kernel,axis=0)
                cos_theta = torch.mm(embeddings,kernel_norm)
                cos_theta = cos_theta.clamp(-1,1)
            else:
                cos_theta = self.head.get_angle(embeddings)

            thetas = torch.abs(torch.rad2deg(torch.acos(cos_theta)))

            for i in range(len(thetas)):
                age_bin = 7
                if ages[i] < 26:
                    age_bin = 0 if ages[i] < 13 else 1 if ages[i] <19 else 2
                elif ages[i] < 66:
                    age_bin = int(((ages[i]+4)//10).item())
                angle_table[labels[i]][age_bin].add(thetas[i][labels[i]].item())
                
        if conf.resume_analysis:
            with open('analysis/angle_table.pkl','rb') as f:
                angle_table = pickle.load(f)
        else:
            with open('analysis/angle_table.pkl', 'wb') as f:
                pickle.dump(angle_table,f)
                
        count, avg_angle = [], []
        for i in range(self.class_num):
            count.append([len(single_set) for single_set in angle_table[i].values()])
            avg_angle.append([sum(list(single_set))/len(single_set) if len(single_set) else 0 # if set() size is zero, avg is zero
                                 for single_set in angle_table[i].values()])

        count_df = pd.DataFrame(count)
        avg_angle_df = pd.DataFrame(avg_angle)

        with pd.ExcelWriter('analysis/analyze_angle_{}_{}.xlsx'.format(conf.data_mode, name)) as writer:  
            count_df.to_excel(writer, sheet_name='count')
            avg_angle_df.to_excel(writer, sheet_name='avg_angle')

    def schedule_lr(self):
        for params in self.optimizer.param_groups:                 
            params['lr'] /= 10
        print(self.optimizer)
    
    
    def infer(self, conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = transforms.functional.hflip(img)
                emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:                        
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)
        
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum   


    def save_best_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path

        os.makedirs('work_space/models', exist_ok=True)
        torch.save(
            self.model.state_dict(), str(save_path) +
            ('lfw_best_model_{}_accuracy:{:.3f}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
        if not model_only:
            torch.save(
                self.head.state_dict(),str(save_path) +
                ('lfw_best_head_{}_accuracy:{:.3f}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
            torch.save(
                self.optimizer.state_dict(),str(save_path) +
                ('lfw_best_optimizer_{}_accuracy:{:.3f}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))


    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path

        os.makedirs('work_space/models', exist_ok=True)
        torch.save(
            self.model.state_dict(), str(save_path) +
            ('/model_{}_accuracy:{:.3f}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
        if not model_only:
            torch.save(
                self.head.state_dict(), str(save_path) +
                ('/head_{}_accuracy:{:.3f}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
            torch.save(
                self.optimizer.state_dict(), str(save_path) +
                ('/optimizer_{}_accuracy:{:.3f}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
    

    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False, analyze=False):
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        self.model.load_state_dict(torch.load(os.path.join(save_path, 'model_{}'.format(fixed_str))))
        if not model_only:
            self.head.load_state_dict(torch.load(save_path/'head_{}'.format(fixed_str)))
            if not analyze:
                self.optimizer.load_state_dict(torch.load(save_path/'optimizer_{}'.format(fixed_str)))
        