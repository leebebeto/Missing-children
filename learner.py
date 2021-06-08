import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
# from tensorboardX import SummaryWriter
# import pandas as pd
from sync_batchnorm import convert_model

from tqdm import tqdm
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from PIL import Image
import pickle
import math
import os
import glob
from data.data_pipe import de_preprocess, get_train_loader, get_val_data
from model import *
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from verification import evaluate, evaluate_dist
from torchvision.utils import save_image
import pdb
import wandb

class face_learner(object):
    def __init__(self, conf=None, inference=False):

        self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
        print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))

        self.conf = conf
        self.epoch = self.conf.epochs

        # For Tsne -> you can ignore these codes
        # self.head = Arcface(embedding_size=conf.embedding_size, classnum=11076).to(conf.device)

        if conf.wandb:
            # wandb.init(project=f"Face-Recognition(BMVC2021)")
            wandb.init(entity="davian-bmvc-face")
            wandb.run.name = conf.exp

        if not inference:
            self.alpha = conf.alpha
            # self.milestones = [6, 11, 16]
            # self.milestones = [8, 16, 24] # Ours 30 naive
            # self.milestones = [9, 15, 21]
            # self.milestones = [11, 16, 21]
            # self.milestones = [6, 11] # Sphereface paper 28epoch
            # self.milestones = [16, 24, 28] # Cosface paper 30epoch
            self.milestones = [28, 38, 46] # Superlong 50epoch

            if self.conf.loss == 'Curricular' or 'MILE28' in self.conf.exp:
                self.milestones = [28, 38, 46]  # Cosface paper 30epoch
                self.epoch= 50

            if self.conf.loss == 'Curricular':
                self.milestones = [28, 38, 46]  # Curricular face paper 50epoch

            if self.conf.short_milestone:
                self.milestones = [4, 8, 10]  # Curricular face paper 50epoch
                self.epoch= 12


            print(f'curr milestones: {self.milestones}')
            print(f'total epochs: {self.epoch}')

            for e in range(self.epoch):
                if conf.lambda_mode == 'normal':
                    child_lambda = 0.0 if (e == 0) or (e in self.milestones) else 1.0
                elif conf.lambda_mode == 'zero':
                    child_lambda = 0.0 if (e == 0) or (e >= self.milestones[0]) else self.conf.lambda_child * 1.0
                elif conf.lambda_mode == 'anneal9':
                    if (e == 0) or (e in self.milestones):
                        child_lambda = 0.0
                    elif e >= self.milestones[0]:
                        child_lambda = conf.lambda_child * 0.9 ** (e-self.milestones[0])
                    else:
                        child_lambda = 1

                elif conf.lambda_mode == 'anneal8':
                    if (e == 0) or (e in self.milestones):
                        child_lambda = 0.0
                    elif e >= self.milestones[0]:
                        child_lambda = conf.lambda_child * 0.8 ** (e-self.milestones[0])
                    else:
                        child_lambda = 1
                print(f'e: {e} / child_lambda: {child_lambda}')

            self.loader, self.class_num, self.ds, self.child_identity, self.child_identity_min, self.child_identity_max = get_train_loader(conf)
            self.log_path = os.path.join(conf.log_path, conf.data_mode, conf.exp)

            os.makedirs(self.log_path, exist_ok=True)
            # self.writer = SummaryWriter(self.log_path)
            self.step = 0

            if 'MIXUP' in conf.exp:
                self.class_num += conf.new_id

            if conf.loss == 'Arcface':
                self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)
            # Arcface with minus margin for children
            # elif conf.loss == 'ArcfaceMinus':
            #     self.head = ArcfaceMinus(embedding_size=conf.embedding_size, classnum=self.class_num, minus_m=conf.minus_m).to(conf.device)
            elif conf.loss == 'Cosface':
                self.head = CosineMarginProduct(embedding_size=conf.embedding_size, classnum=self.class_num, scale=conf.scale).to(conf.device)
            elif conf.loss == 'Sphereface':
                self.head = SphereMarginProduct(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)
            # elif conf.loss == 'LDAM':
            #     self.head = LDAMLoss(embedding_size=conf.embedding_size, classnum=self.class_num, max_m=conf.max_m, s=conf.scale, cls_num_list=self.ds.class_num_list).to(conf.device)
            elif conf.loss == 'Curricular':
                self.head = CurricularFace(in_features=conf.embedding_size, out_features=self.class_num).to(conf.device)
            elif conf.loss == 'MV-AM':
                self.head = FC(in_feature=conf.embedding_size, out_feature=self.class_num, fc_type='MV-AM').to(conf.device)
            elif conf.loss == 'MV-Arc':
                self.head = FC(in_feature=conf.embedding_size, out_feature=self.class_num, fc_type='MV-Arc').to(conf.device)
            elif conf.loss == 'Broad':
                self.head = BroadFaceArcFace(in_features=conf.embedding_size, out_features=self.class_num).to(conf.device)
            else:
                import sys
                print('wrong loss function.. exiting...')
                sys.exit(0)

            # Currently use Data Parallel as default
            if conf.use_dp:
                self.model = nn.DataParallel(self.model)
                self.head = nn.DataParallel(self.head)

                if conf.use_sync == True:
                    self.model = convert_model(self.model)


            print(self.class_num)
            print(conf)

            print('two model heads generated')

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)

            if conf.use_dp:
                self.optimizer1 = optim.SGD([
                                    {'params': paras_wo_bn + [self.head.module.kernel], 'weight_decay': 5e-4},
                                ], lr = conf.lr, momentum = conf.momentum)

                self.optimizer2 = optim.SGD([
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)


            else:
                self.optimizer1 = optim.SGD([
                                    {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                                ], lr = conf.lr, momentum = conf.momentum)

                self.optimizer2 = optim.SGD([
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)

            print('optimizers generated')
            # self.board_loss_every = len(self.loader)//100
            # self.evaluate_every = len(self.loader)//5
            # self.save_every = len(self.loader)//5

            self.board_loss_every = conf.loss_freq
            self.evaluate_every = conf.evaluate_freq
            self.save_every = conf.save_freq


            print(conf)
            print('training starts.... BMVC 2021....')

            # dataset_root= os.path.join('/home/nas1_userE/jungsoolee/Face_dataset/face_emore2')
            dataset_root= os.path.join('./dataset/face_emore2')
            # dataset_root= os.path.join(conf.home, 'dataset/face_emore2')
            # self.lfw, self.lfw_issame = get_val_data(dataset_root)
            # dataset_root = "./dataset/"
            self.lfw = np.load(os.path.join(dataset_root, "lfw_align_112_list.npy")).astype(np.float32)
            self.lfw_issame = np.load(os.path.join(dataset_root, "lfw_align_112_label.npy"))

            self.fgnetc = np.load(os.path.join(dataset_root, "FGNET_new_align_list.npy")).astype(np.float32)
            self.fgnetc_issame = np.load(os.path.join(dataset_root, "FGNET_new_align_label.npy"))
            # self.cfp_fp, self.cfp_fp_issame = get_val_data(dataset_root, 'cfp_fp')
            # self.agedb, self.agedb_issame = get_val_data(dataset_root, 'agedb_30')

        else:
            # Will not use anymore
            # self.model = nn.DataParallel(self.model)
            # self.threshold = conf.threshold
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
            self.threshold = 0


    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor, negative_wrong, positive_wrong):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_scalar('{}_negative_wrong'.format(db_name), negative_wrong, self.step)
        self.writer.add_scalar('{}_positive_wrong'.format(db_name), positive_wrong, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)

        print(f'{db_name} accuracy: {accuracy}')
    def evaluate(self, conf, carray, issame, nrof_folds = 10, tta = True):
        self.model.eval()
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
        tpr, fpr, accuracy, best_thresholds, dist = evaluate_dist(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = transforms.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor, dist

    def mixup_criterion(criterion, pred, y_a, y_b, lam=0.5):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    # training with other methods
    def train(self, conf, epochs):
        '''
        Train function for original vgg dataset
        XXX: Need to make a new funtion train_age_invariant(conf, epochs)
        '''
        self.model.train()

        running_loss = 0.            
        best_accuracy = 0.0
        ce_loss = nn.CrossEntropyLoss()
        print(f'total epoch: {self.epoch}')
        for e in range(self.epoch):
            print('epoch {} started'.format(e))

            if e in self.milestones:
                self.schedule_lr()

            for imgs, labels, ages in tqdm(iter(self.loader)):
            # for imgs, labels in tqdm(iter(self.loader)):
                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()
                if imgs.shape[0] == 1:
                    continue

                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)

                embeddings = self.model(imgs)
                # thetas = self.head(embeddings, labels)
                thetas = self.head(embeddings, labels, ages)

                if self.conf.loss == 'Broad':
                    loss= thetas
                else:
                    loss = ce_loss(thetas, labels)
                loss.backward()
                running_loss += loss.item()

                self.optimizer1.step()
                self.optimizer2.step()

                if self.step % self.board_loss_every == 0 and self.step != 0: # XXX
                    # print('wandb plotting....')
                    loss_board = running_loss / self.board_loss_every
                    # self.writer.add_scalar('train_loss', loss_board, self.step)
                    if self.conf.wandb:
                        wandb.log({
                            "train_loss": loss_board,
                        }, step=self.step)

                    running_loss = 0.

                # added wrong on evaluations
                if self.step % self.evaluate_every == 0 and self.step != 0:
                    self.model.eval()
                    print('evaluating....')
                    # # LFW evaluation
                    # accuracy, best_threshold, roc_curve_tensor, dist = self.evaluate(conf, self.lfw, self.lfw_issame)
                    # # NEGATIVE WRONG
                    # wrong_list = np.where((self.lfw_issame == False) & (dist < best_threshold))[0]
                    # negative_wrong = len(wrong_list)
                    # # POSITIVE WRONG
                    # wrong_list = np.where((self.lfw_issame == True) & (dist > best_threshold))[0]
                    # positive_wrong = len(wrong_list)
                    # self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor, negative_wrong, positive_wrong)
                    lfw_accuracy, lfw_thres, roc_curve_tensor2, lfw_dist = self.evaluate(conf, self.lfw, self.lfw_issame)
                    # NEGATIVE WRONG
                    wrong_list = np.where((self.lfw_issame == False) & (lfw_dist < lfw_thres))[0]
                    lfw_negative = len(wrong_list)
                    # POSITIVE WRONG
                    wrong_list = np.where((self.lfw_issame == True) & (lfw_dist > lfw_thres))[0]
                    lfw_positive = len(wrong_list)

                    # # CFP FP evaluation
                    # accuracy, best_threshold, roc_curve_tensor, dist = self.evaluate(conf, self.cfp_fp, self.cfp_fp_issame)
                    # # NEGATIVE WRONG
                    # wrong_list = np.where((self.cfp_fp_issame == False) & (dist < best_threshold))[0]
                    # negative_wrong = len(wrong_list)
                    # # POSITIVE WRONG
                    # wrong_list = np.where((self.cfp_fp_issame == True) & (dist > best_threshold))[0]
                    # positive_wrong = len(wrong_list)
                    # self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor, negative_wrong, positive_wrong)
                    #
                    # # agedb evaluation
                    # accuracy, best_threshold, roc_curve_tensor, dist = self.evaluate(conf, self.agedb, self.agedb_issame)
                    # # NEGATIVE WRONG
                    # wrong_list = np.where((self.agedb_issame == False) & (dist < best_threshold))[0]
                    # negative_wrong = len(wrong_list)
                    # # POSITIVE WRONG
                    # wrong_list = np.where((self.agedb_issame == True) & (dist > best_threshold))[0]
                    # positive_wrong = len(wrong_list)
                    # self.board_val('agedb', accuracy, best_threshold, roc_curve_tensor, negative_wrong, positive_wrong)

                    # FGNETC evaluation
                    fgnetc_accuracy, fgnetc_thres, roc_curve_tensor2, fgnetc_dist = self.evaluate(conf, self.fgnetc, self.fgnetc_issame)
                    # NEGATIVE WRONG
                    wrong_list = np.where((self.fgnetc_issame == False) & (fgnetc_dist < fgnetc_thres))[0]
                    fgnetc_negative = len(wrong_list)
                    # POSITIVE WRONG
                    wrong_list = np.where((self.fgnetc_issame == True) & (fgnetc_dist > fgnetc_thres))[0]
                    fgnetc_positive = len(wrong_list)
                    # self.board_val('fgent_c', accuracy2, best_threshold2, roc_curve_tensor2, fgnet_negative\, positive_wrong2)
                    print(f'fgnetc_acc: {fgnetc_accuracy}')

                    if self.conf.wandb:
                        wandb.log({
                            "lfw_acc": lfw_accuracy,
                            "lfw_best_threshold": lfw_thres,
                            "lfw_negative_wrong": lfw_negative,
                            "lfw_positive_wrong": lfw_positive,

                            "fgnet_c_acc": fgnetc_accuracy,
                            "fgnet_c_best_threshold": fgnetc_thres,
                            "fgnet_c_negative_wrong": fgnetc_negative,
                            "fgnet_c_positive_wrong": fgnetc_positive,
                        }, step=self.step)

                    self.model.train()
                    if self.step % self.save_every == 0 and self.step != 0:
                        print('saving model....')
                        # save with most recently calculated accuracy?
                        # if conf.finetune_model_path is not None:
                        #     self.save_state(conf, accuracy2,
                        #                     extra=str(conf.data_mode) + '_' + str(conf.net_depth) + '_' + str(
                        #                         conf.batch_size) + 'finetune')
                        # else:
                        #     self.save_state(conf, accuracy2,extra=str(conf.data_mode) + '_' + str(conf.exp) + '_' + str(conf.batch_size))

                        if fgnetc_accuracy > best_accuracy:
                            best_accuracy = fgnetc_accuracy
                            print('saving best model....')
                            self.save_best_state(conf, best_accuracy, extra=str(conf.data_mode) + '_' + str(conf.exp))
                self.step += 1

        # if conf.finetune_model_path is not None:
        #     self.save_state(conf, fgnetc_accuracy, to_save_folder=True, extra=str(conf.data_mode)  + '_' + str(conf.net_depth) + '_'+ str(conf.batch_size) +'_finetune')
        # else:
        #     self.save_state(conf, fgnetc_accuracy, to_save_folder=True, extra=str(conf.data_mode)  + '_' + str(conf.net_depth) + '_'+ str(conf.batch_size) +'_final')
        if conf.wandb:
            wandb.finish()
    # training with memory bank
    def train_memory(self, conf, epochs):
        '''
        Train function for original vgg dataset
        XXX: Need to make a new funtion train_age_invariant(conf, epochs)
        '''
        self.model.train()
        running_loss = 0.
        running_arcface_loss, running_child_loss, running_child_total_loss = 0.0, 0.0, 0.0
        running_mixup_loss, running_mixup_total_loss = 0.0, 0.0
        best_accuracy = 0.0
        ce_loss = nn.CrossEntropyLoss()
        # initialize memory bank
        # reversed shape to use like dictionary
        self.child_memory = nn.Parameter(torch.Tensor(self.class_num, conf.embedding_size)).to(conf.device)
        child_loss = 0.0
        mixup_loss = torch.tensor(0.0)
        self.child_labels = torch.tensor(self.child_identity).cuda()
        fgnetc_best_acc = 0.0
        print(f'total epoch: {self.epoch}')
        for e in range(self.epoch):
            print('epoch {} started'.format(e))
            if e in self.milestones:
                self.schedule_lr()
            for imgs, labels, ages in tqdm(iter(self.loader)):
                # for imgs, labels in tqdm(iter(self.loader)):
                child_idx = torch.where(ages == 0)[0]
                # self.child_identity += child_idx.numpy().tolist()
                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)

                embeddings = self.model(imgs)
                # thetas = self.head(embeddings, labels)
                thetas = self.head(embeddings, labels, ages)
                arcface_loss = ce_loss(thetas, labels)

                if conf.lambda_mode == 'normal':
                    child_lambda = 0.0 if (e == 0) or (e in self.milestones) else 1.0
                elif conf.lambda_mode == 'zero':
                    child_lambda = 0.0 if (e == 0) or (e >= self.milestones[0]) else self.conf.lambda_child * 1.0
                elif conf.lambda_mode == 'anneal9':
                    if (e == 0) or (e in self.milestones):
                        child_lambda = 0.0
                    elif e >= self.milestones[0]:
                        child_lambda = conf.lambda_child * 0.9 ** (e - self.milestones[0])
                    else:
                        child_lambda = 1
                elif conf.lambda_mode == 'anneal8':
                    if (e == 0) or (e in self.milestones):
                        child_lambda = 0.0
                    elif e >= self.milestones[0]:
                        child_lambda = conf.lambda_child * 0.8 ** (e - self.milestones[0])
                    else:
                        child_lambda = 1

                # child_lambda=1.0
                with torch.no_grad():
                    if len(child_idx) > 0:
                        if (e == 0) or (e in self.milestones):
                            self.child_memory[labels[child_idx]] = embeddings[child_idx].detach().clone()
                        else:
                            self.child_memory[labels[child_idx]] = (1-self.alpha) * embeddings[child_idx].detach().clone() + self.alpha * self.child_memory[child_idx].detach().clone()
                # ''' module for positive pair -> child memory bank '''
                child_embeddings = self.child_memory[self.child_labels]
                child_thetas = self.head(child_embeddings, self.child_labels)
                child_loss = ce_loss(child_thetas, self.child_labels)
                child_total_loss = child_lambda * child_loss
                loss = arcface_loss + child_total_loss
                loss.backward()
                running_loss += loss.item()

                running_arcface_loss += arcface_loss.item()
                # if 'POSITIVE' in conf.exp:
                running_child_loss += child_loss.item()
                running_child_total_loss += child_total_loss.item()

                if 'MIXUP' in conf.exp:
                    running_mixup_loss += mixup_loss.item()
                    running_mixup_total_loss += mixup_total_loss.item()

                self.optimizer1.step()
                self.optimizer2.step()

                del embeddings
                # del child_embeddings, child_labels, child_thetas
                del imgs, labels, thetas, arcface_loss
                del child_idx, ages

                if self.step % self.board_loss_every == 0 and self.step != 0:  # XXX
                    # print('tensorboard plotting....')
                    # print('wandb plotting....')
                    loss_board = running_loss / self.board_loss_every

                    arcface_loss_board = running_arcface_loss / self.board_loss_every
                    # self.writer.add_scalar('train_loss', loss_board, self.step)
                    # self.writer.add_scalar('arcface_loss', arcface_loss_board, self.step)

                    if self.conf.wandb:
                        wandb.log({
                            "train_loss": loss_board,
                            "arcface_total_loss": arcface_loss_board,
                        }, step=self.step)
                    child_loss_board = running_child_loss / self.board_loss_every
                    child_total_loss_board = running_child_total_loss / self.board_loss_every

                    if self.conf.wandb:
                        wandb.log({
                            "child_loss": child_loss_board,
                            "child_total_loss": child_total_loss_board,
                            "child_lambda": child_lambda,
                        }, step=self.step)

                        # self.writer.add_scalar('child_loss', child_loss_board, self.step)
                        # self.writer.add_scalar('child_total_loss', child_total_loss_board, self.step)
                    if 'MIXUP' in conf.exp:
                        mixup_loss_board = running_mixup_loss / self.board_loss_every
                        mixup_total_loss_board = running_mixup_total_loss / self.board_loss_every
                        if self.conf.wandb:
                            wandb.log({
                                "mixup_loss": mixup_loss_board,
                                "mixup_total_loss": mixup_total_loss_board,
                            }, step=self.step)

                        # self.writer.add_scalar('mixup_loss', mixup_loss_board, self.step)
                        # self.writer.add_scalar('mixup_total_loss', mixup_total_loss_board, self.step)

                    running_loss = 0.
                    running_arcface_loss = 0.0
                    running_child_loss = 0.0
                    running_child_total_loss = 0.0
                    running_mixup_loss = 0.0
                    running_mixup_total_loss = 0.0

                if self.step % self.evaluate_every == 0 and self.step != 0:
                    print('evaluating....')
                    # # LFW evaluation
                    # accuracy, best_threshold, roc_curve_tensor, dist = self.evaluate(conf, self.lfw, self.lfw_issame)
                    # # NEGATIVE WRONG
                    # wrong_list = np.where((self.lfw_issame == False) & (dist < best_threshold))[0]
                    # negative_wrong = len(wrong_list)
                    # # POSITIVE WRONG
                    # wrong_list = np.where((self.lfw_issame == True) & (dist > best_threshold))[0]
                    # positive_wrong = len(wrong_list)
                    # self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor, negative_wrong, positive_wrong)
                    # # FGNETC evaluation
                    # accuracy2, best_threshold2, roc_curve_tensor2, dist2 = self.evaluate(conf, self.fgnetc,
                    #                                                                      self.fgnetc_issame)
                    # # NEGATIVE WRONG
                    # wrong_list = np.where((self.fgnetc_issame == False) & (dist2 < best_threshold2))[0]
                    # negative_wrong2 = len(wrong_list)
                    # # POSITIVE WRONG
                    # wrong_list = np.where((self.fgnetc_issame == True) & (dist2 > best_threshold2))[0]
                    # positive_wrong2 = len(wrong_list)
                    # self.board_val('fgent_c', accuracy2, best_threshold2, roc_curve_tensor2, negative_wrong2,
                    #                positive_wrong2)

                    # FGNETC evaluation
                    fgnetc_accuracy, fgnetc_thres, roc_curve_tensor2, fgnetc_dist = self.evaluate(conf, self.fgnetc,
                                                                                                  self.fgnetc_issame)
                    # NEGATIVE WRONG
                    wrong_list = np.where((self.fgnetc_issame == False) & (fgnetc_dist < fgnetc_thres))[0]
                    fgnetc_negative = len(wrong_list)
                    # POSITIVE WRONG
                    wrong_list = np.where((self.fgnetc_issame == True) & (fgnetc_dist > fgnetc_thres))[0]
                    fgnetc_positive = len(wrong_list)
                    print(f'fgnetc_acc: {fgnetc_accuracy}')

                    if fgnetc_accuracy > fgnetc_best_acc:
                        fgnetc_best_acc = fgnetc_accuracy

                    if self.conf.wandb:
                        wandb.log({
                            "fgnet_c_best_acc": fgnetc_best_acc,
                            "fgnet_c_acc": fgnetc_accuracy,
                            "fgnet_c_best_threshold": fgnetc_thres,
                            "fgnet_c_negative_wrong": fgnetc_negative,
                            "fgnet_c_positive_wrong": fgnetc_positive,
                        }, step=self.step)



                    self.model.train()

                    if self.step % self.save_every == 0 and self.step != 0:
                        print('saving model....')
                        # save with most recently calculated accuracy?
                        # if conf.finetune_model_path is not None:
                        #     self.save_state(conf, accuracy2,
                        #                     extra=str(conf.data_mode) + '_' + str(conf.net_depth) + '_' + str(
                        #                         conf.batch_size) + 'finetune')
                        # else:
                        #     self.save_state(conf, accuracy2,extra=str(conf.data_mode) + '_' + str(conf.exp) + '_' + str(conf.batch_size))

                        if fgnetc_accuracy > best_accuracy:
                            best_accuracy = fgnetc_accuracy
                            print('saving best model....')
                            self.save_best_state(conf, best_accuracy, extra=str(conf.data_mode) + '_' + str(conf.exp))
                self.step += 1
        if conf.finetune_model_path is not None:
            self.save_state(conf, accuracy, to_save_folder=True,
                            extra=str(conf.data_mode) + '_' + str(conf.net_depth) + '_' + str(
                                conf.batch_size) + '_finetune')
        else:
            self.save_state(conf, accuracy, to_save_folder=True,
                            extra=str(conf.data_mode) + '_' + str(conf.net_depth) + '_' + str(
                                conf.batch_size) + '_final')

    # training with memory bank
    def train_adult_memory(self, conf, epochs):
        '''
        Train function for original vgg dataset
        XXX: Need to make a new funtion train_age_invariant(conf, epochs)
        '''
        self.model.train()
        running_loss = 0.
        running_arcface_loss, running_child_loss, running_child_total_loss = 0.0, 0.0, 0.0
        running_mixup_loss, running_mixup_total_loss = 0.0, 0.0
        best_accuracy = 0.0
        ce_loss = nn.CrossEntropyLoss()
        l1_loss = nn.L1Loss()
        # initialize memory bank
        # reversed shape to use like dictionary
        self.child_memory = nn.Parameter(torch.Tensor(self.class_num, conf.embedding_size)).to(conf.device)
        self.adult_memory = nn.Parameter(torch.Tensor(self.class_num, conf.embedding_size)).to(conf.device)
        child_loss = 0.0
        mixup_loss = torch.tensor(0.0)
        self.child_labels = torch.tensor(self.child_identity).cuda()
        self.adult_labels = torch.zeros(self.class_num).cuda()
        self.adult_labels[self.child_labels] = 1
        fgnetc_best_acc = 0.0
        print(f'total epoch: {self.epoch}')
        for e in range(self.epoch):
            print('epoch {} started'.format(e))
            if e in self.milestones:
                self.schedule_lr()
            for imgs, labels, ages in tqdm(iter(self.loader)):
                # for imgs, labels in tqdm(iter(self.loader)):
                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)

                child_idx = torch.where(ages == 0)[0]
                adult_flag = torch.index_select(self.adult_labels, 0, labels)
                adult_idx = labels[torch.where(adult_flag ==True)]

                embeddings = self.model(imgs)
                # thetas = self.head(embeddings, labels)
                thetas = self.head(embeddings, labels, ages)
                arcface_loss = ce_loss(thetas, labels)

                if conf.lambda_mode == 'normal':
                    child_lambda = 0.0 if (e == 0) or (e in self.milestones) else 1.0
                elif conf.lambda_mode == 'zero':
                    child_lambda = 0.0 if (e == 0) or (e >= self.milestones[0]) else self.conf.lambda_child * 1.0
                elif conf.lambda_mode == 'anneal9':
                    if (e == 0) or (e in self.milestones):
                        child_lambda = 0.0
                    elif e >= self.milestones[0]:
                        child_lambda = conf.lambda_child * 0.9 ** (e - self.milestones[0])
                    else:
                        child_lambda = 1
                elif conf.lambda_mode == 'anneal8':
                    if (e == 0) or (e in self.milestones):
                        child_lambda = 0.0
                    elif e >= self.milestones[0]:
                        child_lambda = conf.lambda_child * 0.8 ** (e - self.milestones[0])
                    else:
                        child_lambda = 1

                # child_lambda=1.0
                with torch.no_grad():
                    if len(child_idx) > 0:
                        if (e == 0) or (e in self.milestones):
                            self.child_memory[labels[child_idx]] = embeddings[child_idx].detach().clone()
                        else:
                            self.child_memory[labels[child_idx]] = (1-self.alpha) * embeddings[child_idx].detach().clone() + self.alpha * self.child_memory[child_idx].detach().clone()

                with torch.no_grad():
                    if len(adult_idx) > 0:
                        if (e == 0) or (e in self.milestones):
                            self.adult_memory[adult_idx] = embeddings[torch.where(adult_flag ==True)].detach().clone()
                        else:
                            self.adult_memory[adult_idx] = (1-self.alpha) * embeddings[torch.where(adult_flag ==True)].detach().clone() + self.alpha * self.child_memory[adult_idx].detach().clone()

                # ''' module for positive pair -> child memory bank '''
                child_embeddings = self.child_memory[self.child_labels]
                adult_embeddings = self.adult_memory[self.child_labels]
                child_thetas = self.head(child_embeddings, self.child_labels)
                adult_thetas = self.head(adult_embeddings, self.child_labels)

                child_thetas = torch.index_select(child_thetas, 1, self.child_labels).sum(dim=1)
                adult_thetas = torch.index_select(adult_thetas, 1, self.child_labels).sum(dim=1)
                child_loss = l1_loss(child_thetas, adult_thetas)
                # child_loss = ce_loss(child_thetas, self.child_labels)
                child_total_loss = child_lambda * child_loss
                loss = arcface_loss + child_total_loss
                loss.backward()
                running_loss += loss.item()

                running_arcface_loss += arcface_loss.item()
                # if 'POSITIVE' in conf.exp:
                running_child_loss += child_loss.item()
                running_child_total_loss += child_total_loss.item()

                if 'MIXUP' in conf.exp:
                    running_mixup_loss += mixup_loss.item()
                    running_mixup_total_loss += mixup_total_loss.item()

                self.optimizer1.step()
                self.optimizer2.step()

                del embeddings
                # del child_embeddings, child_labels, child_thetas
                del imgs, labels, thetas, arcface_loss
                del child_idx, ages

                if self.step % self.board_loss_every == 0 and self.step != 0:  # XXX
                    # print('tensorboard plotting....')
                    # print('wandb plotting....')
                    loss_board = running_loss / self.board_loss_every

                    arcface_loss_board = running_arcface_loss / self.board_loss_every
                    # self.writer.add_scalar('train_loss', loss_board, self.step)
                    # self.writer.add_scalar('arcface_loss', arcface_loss_board, self.step)

                    if self.conf.wandb:
                        wandb.log({
                            "train_loss": loss_board,
                            "arcface_total_loss": arcface_loss_board,
                        }, step=self.step)
                    child_loss_board = running_child_loss / self.board_loss_every
                    child_total_loss_board = running_child_total_loss / self.board_loss_every

                    if self.conf.wandb:
                        wandb.log({
                            "child_loss": child_loss_board,
                            "child_total_loss": child_total_loss_board,
                            "child_lambda": child_lambda,
                        }, step=self.step)

                        # self.writer.add_scalar('child_loss', child_loss_board, self.step)
                        # self.writer.add_scalar('child_total_loss', child_total_loss_board, self.step)
                    if 'MIXUP' in conf.exp:
                        mixup_loss_board = running_mixup_loss / self.board_loss_every
                        mixup_total_loss_board = running_mixup_total_loss / self.board_loss_every
                        if self.conf.wandb:
                            wandb.log({
                                "mixup_loss": mixup_loss_board,
                                "mixup_total_loss": mixup_total_loss_board,
                            }, step=self.step)

                        # self.writer.add_scalar('mixup_loss', mixup_loss_board, self.step)
                        # self.writer.add_scalar('mixup_total_loss', mixup_total_loss_board, self.step)

                    running_loss = 0.
                    running_arcface_loss = 0.0
                    running_child_loss = 0.0
                    running_child_total_loss = 0.0
                    running_mixup_loss = 0.0
                    running_mixup_total_loss = 0.0

                if self.step % self.evaluate_every == 0 and self.step != 0:
                    print('evaluating....')
                    # # LFW evaluation
                    # accuracy, best_threshold, roc_curve_tensor, dist = self.evaluate(conf, self.lfw, self.lfw_issame)
                    # # NEGATIVE WRONG
                    # wrong_list = np.where((self.lfw_issame == False) & (dist < best_threshold))[0]
                    # negative_wrong = len(wrong_list)
                    # # POSITIVE WRONG
                    # wrong_list = np.where((self.lfw_issame == True) & (dist > best_threshold))[0]
                    # positive_wrong = len(wrong_list)
                    # self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor, negative_wrong, positive_wrong)
                    # # FGNETC evaluation
                    # accuracy2, best_threshold2, roc_curve_tensor2, dist2 = self.evaluate(conf, self.fgnetc,
                    #                                                                      self.fgnetc_issame)
                    # # NEGATIVE WRONG
                    # wrong_list = np.where((self.fgnetc_issame == False) & (dist2 < best_threshold2))[0]
                    # negative_wrong2 = len(wrong_list)
                    # # POSITIVE WRONG
                    # wrong_list = np.where((self.fgnetc_issame == True) & (dist2 > best_threshold2))[0]
                    # positive_wrong2 = len(wrong_list)
                    # self.board_val('fgent_c', accuracy2, best_threshold2, roc_curve_tensor2, negative_wrong2,
                    #                positive_wrong2)

                    # FGNETC evaluation
                    fgnetc_accuracy, fgnetc_thres, roc_curve_tensor2, fgnetc_dist = self.evaluate(conf, self.fgnetc,
                                                                                                  self.fgnetc_issame)
                    # NEGATIVE WRONG
                    wrong_list = np.where((self.fgnetc_issame == False) & (fgnetc_dist < fgnetc_thres))[0]
                    fgnetc_negative = len(wrong_list)
                    # POSITIVE WRONG
                    wrong_list = np.where((self.fgnetc_issame == True) & (fgnetc_dist > fgnetc_thres))[0]
                    fgnetc_positive = len(wrong_list)
                    print(f'fgnetc_acc: {fgnetc_accuracy}')

                    if fgnetc_accuracy > fgnetc_best_acc:
                        fgnetc_best_acc = fgnetc_accuracy

                    if self.conf.wandb:
                        wandb.log({
                            "fgnet_c_best_acc": fgnetc_best_acc,
                            "fgnet_c_acc": fgnetc_accuracy,
                            "fgnet_c_best_threshold": fgnetc_thres,
                            "fgnet_c_negative_wrong": fgnetc_negative,
                            "fgnet_c_positive_wrong": fgnetc_positive,
                        }, step=self.step)



                    self.model.train()

                    if self.step % self.save_every == 0 and self.step != 0:
                        print('saving model....')
                        # save with most recently calculated accuracy?
                        # if conf.finetune_model_path is not None:
                        #     self.save_state(conf, accuracy2,
                        #                     extra=str(conf.data_mode) + '_' + str(conf.net_depth) + '_' + str(
                        #                         conf.batch_size) + 'finetune')
                        # else:
                        #     self.save_state(conf, accuracy2,extra=str(conf.data_mode) + '_' + str(conf.exp) + '_' + str(conf.batch_size))

                        if fgnetc_accuracy > best_accuracy:
                            best_accuracy = fgnetc_accuracy
                            print('saving best model....')
                            self.save_best_state(conf, best_accuracy, extra=str(conf.data_mode) + '_' + str(conf.exp))
                self.step += 1
        if conf.finetune_model_path is not None:
            self.save_state(conf, accuracy, to_save_folder=True,
                            extra=str(conf.data_mode) + '_' + str(conf.net_depth) + '_' + str(
                                conf.batch_size) + '_finetune')
        else:
            self.save_state(conf, accuracy, to_save_folder=True,
                            extra=str(conf.data_mode) + '_' + str(conf.net_depth) + '_' + str(
                                conf.batch_size) + '_final')

    def train_mixup(self, conf, epochs):
        '''
        Train function for original vgg dataset
        XXX: Need to make a new funtion train_age_invariant(conf, epochs)
        '''
        self.model.train()

        running_loss = 0.            
        best_accuracy = 0.0
        ce_loss = nn.CrossEntropyLoss()
        print(f'total epoch: {self.epoch}')
        for e in range(self.epoch):
            print('epoch {} started'.format(e))

            if e in self.milestones:
                self.schedule_lr()

            for imgs, labels, ages in tqdm(iter(self.loader)):
            # for imgs, labels in tqdm(iter(self.loader)):
                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()
                label1, label2 = labels
                if imgs.shape[0] == 1:
                    continue

                imgs = imgs.to(conf.device)
                label1 = label1.to(conf.device)
                label2 = label2.to(conf.device)
                label2 = torch.where(label2==-1, label1, label2)

                embeddings = self.model(imgs)
                # thetas = self.head(embeddings, labels)
                thetas1 = self.head(embeddings, label1, ages)
                thetas2 = self.head(embeddings, label2, ages)

                loss = 0.5 * ce_loss(thetas1, label1) + 0.5 * ce_loss(thetas2, label2)
                loss.backward()
                running_loss += loss.item()

                self.optimizer1.step()
                self.optimizer2.step()

                if self.step % self.board_loss_every == 0 and self.step != 0: # XXX
                    # print('wandb plotting....')
                    loss_board = running_loss / self.board_loss_every
                    # self.writer.add_scalar('train_loss', loss_board, self.step)
                    if self.conf.wandb:
                        wandb.log({
                            "train_loss": loss_board,
                        }, step=self.step)

                    running_loss = 0.

                # added wrong on evaluations
                if self.step % self.evaluate_every == 0 and self.step != 0:
                    self.model.eval()
                    print('evaluating....')
                    # # LFW evaluation
                    # accuracy, best_threshold, roc_curve_tensor, dist = self.evaluate(conf, self.lfw, self.lfw_issame)
                    # # NEGATIVE WRONG
                    # wrong_list = np.where((self.lfw_issame == False) & (dist < best_threshold))[0]
                    # negative_wrong = len(wrong_list)
                    # # POSITIVE WRONG
                    # wrong_list = np.where((self.lfw_issame == True) & (dist > best_threshold))[0]
                    # positive_wrong = len(wrong_list)
                    # self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor, negative_wrong, positive_wrong)
                    lfw_accuracy, lfw_thres, roc_curve_tensor2, lfw_dist = self.evaluate(conf, self.lfw, self.lfw_issame)
                    # NEGATIVE WRONG
                    wrong_list = np.where((self.lfw_issame == False) & (lfw_dist < lfw_thres))[0]
                    lfw_negative = len(wrong_list)
                    # POSITIVE WRONG
                    wrong_list = np.where((self.lfw_issame == True) & (lfw_dist > lfw_thres))[0]
                    lfw_positive = len(wrong_list)

                    # # CFP FP evaluation
                    # accuracy, best_threshold, roc_curve_tensor, dist = self.evaluate(conf, self.cfp_fp, self.cfp_fp_issame)
                    # # NEGATIVE WRONG
                    # wrong_list = np.where((self.cfp_fp_issame == False) & (dist < best_threshold))[0]
                    # negative_wrong = len(wrong_list)
                    # # POSITIVE WRONG
                    # wrong_list = np.where((self.cfp_fp_issame == True) & (dist > best_threshold))[0]
                    # positive_wrong = len(wrong_list)
                    # self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor, negative_wrong, positive_wrong)
                    #
                    # # agedb evaluation
                    # accuracy, best_threshold, roc_curve_tensor, dist = self.evaluate(conf, self.agedb, self.agedb_issame)
                    # # NEGATIVE WRONG
                    # wrong_list = np.where((self.agedb_issame == False) & (dist < best_threshold))[0]
                    # negative_wrong = len(wrong_list)
                    # # POSITIVE WRONG
                    # wrong_list = np.where((self.agedb_issame == True) & (dist > best_threshold))[0]
                    # positive_wrong = len(wrong_list)
                    # self.board_val('agedb', accuracy, best_threshold, roc_curve_tensor, negative_wrong, positive_wrong)

                    # FGNETC evaluation
                    fgnetc_accuracy, fgnetc_thres, roc_curve_tensor2, fgnetc_dist = self.evaluate(conf, self.fgnetc, self.fgnetc_issame)
                    # NEGATIVE WRONG
                    wrong_list = np.where((self.fgnetc_issame == False) & (fgnetc_dist < fgnetc_thres))[0]
                    fgnetc_negative = len(wrong_list)
                    # POSITIVE WRONG
                    wrong_list = np.where((self.fgnetc_issame == True) & (fgnetc_dist > fgnetc_thres))[0]
                    fgnetc_positive = len(wrong_list)
                    # self.board_val('fgent_c', accuracy2, best_threshold2, roc_curve_tensor2, fgnet_negative\, positive_wrong2)
                    print(f'fgnetc_acc: {fgnetc_accuracy}')

                    if self.conf.wandb:
                        wandb.log({
                            "lfw_acc": lfw_accuracy,
                            "lfw_best_threshold": lfw_thres,
                            "lfw_negative_wrong": lfw_negative,
                            "lfw_positive_wrong": lfw_positive,

                            "fgnet_c_acc": fgnetc_accuracy,
                            "fgnet_c_best_threshold": fgnetc_thres,
                            "fgnet_c_negative_wrong": fgnetc_negative,
                            "fgnet_c_positive_wrong": fgnetc_positive,
                        }, step=self.step)

                    self.model.train()
                    if self.step % self.save_every == 0 and self.step != 0:
                        print('saving model....')
                        # save with most recently calculated accuracy?
                        # if conf.finetune_model_path is not None:
                        #     self.save_state(conf, accuracy2,
                        #                     extra=str(conf.data_mode) + '_' + str(conf.net_depth) + '_' + str(
                        #                         conf.batch_size) + 'finetune')
                        # else:
                        #     self.save_state(conf, accuracy2,extra=str(conf.data_mode) + '_' + str(conf.exp) + '_' + str(conf.batch_size))

                        if fgnetc_accuracy > best_accuracy:
                            best_accuracy = fgnetc_accuracy
                            print('saving best model....')
                            self.save_best_state(conf, best_accuracy, extra=str(conf.data_mode) + '_' + str(conf.exp))
                self.step += 1

        # if conf.finetune_model_path is not None:
        #     self.save_state(conf, fgnetc_accuracy, to_save_folder=True, extra=str(conf.data_mode)  + '_' + str(conf.net_depth) + '_'+ str(conf.batch_size) +'_finetune')
        # else:
        #     self.save_state(conf, fgnetc_accuracy, to_save_folder=True, extra=str(conf.data_mode)  + '_' + str(conf.net_depth) + '_'+ str(conf.batch_size) +'_final')
        if conf.wandb:
            wandb.finish()


    def analyze_angle(self, conf):
        '''
        Only works on age labeled vgg dataset, agedb dataset
        '''

        angle_table = [{0:set(), 1:set(), 2:set(), 3:set(), 4:set(), 5:set(), 6:set(), 7:set()} for i in range(self.class_num)]
        # batch = 0
        # _angle_table = torch.zeros(self.class_num, 8, len(self.loader)//conf.batch_size).to(conf.device)
        print('starting analyzing angle....')
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
                # import pdb; pdb.set_trace()

        count, avg_angle = [], []
        for i in range(self.class_num):
            count.append([len(single_set) for single_set in angle_table[i].values()])
            avg_angle.append([sum(list(single_set))/len(single_set) if len(single_set) else 0 # if set() size is zero, avg is zero
                                 for single_set in angle_table[i].values()])

        count_df = pd.DataFrame(count)
        avg_angle_df = pd.DataFrame(avg_angle)

        os.makedirs('analysis', exist_ok= True)
        with pd.ExcelWriter(f'analysis/analyze_angle_{conf.exp}.xlsx'.format(conf.data_mode)) as writer:
            count_df.to_excel(writer, sheet_name='count')
            avg_angle_df.to_excel(writer, sheet_name='avg_angle')


        # if conf.resume_analysis:
        #     with open('analysis/angle_table.pkl','rb') as f:
        #         angle_table = pickle.load(f)
        # else:
        #     with open('analysis/angle_table.pkl', 'wb') as f:
        #         pickle.dump(angle_table,f)


    def schedule_lr(self):
        for params in self.optimizer1.param_groups:
            params['lr'] /= 10
        for params in self.optimizer2.param_groups:
            params['lr'] /= 10

        print(self.optimizer1)
        print(self.optimizer2)

    
    def infer(self, conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        for img in faces:
            if tta:
                mirror = transforms.functional.hflip(img)
                emb = self.model(test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:                        
                embs.append(self.model(test_transform(img).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)
        
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum   


    def save_best_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        # if to_save_folder:
        #     save_path = conf.save_path
        # else:
        #     save_path = conf.model_path
        save_path = os.path.join(conf.model_path, conf.exp)
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_path, ('fgnetc_best_model_{}_accuracy:{:.3f}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra))))
        if not model_only:
            torch.save(
                self.head.state_dict(), os.path.join(save_path, ('fgnetc_best_head_{}_accuracy:{:.3f}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra))))
            # torch.save(
            #     self.optimizer1.state_dict(), os.path.join( save_path, ('lfw_best_optimizer1_{}_accuracy:{:.3f}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra))))
            # torch.save(
            #     self.optimizer2.state_dict(), os.path.join( save_path), ('lfw_best_optimizer2_{}_accuracy:{:.3f}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))


    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path

        os.makedirs(conf.model_path, exist_ok=True)
        torch.save(
            self.model.state_dict(), str(save_path) +
            ('/model_{}_accuracy:{:.3f}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
        if not model_only:
            torch.save(
                self.head.state_dict(), str(save_path) +
                ('/head_{}_accuracy:{:.3f}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
            torch.save(
                self.optimizer1.state_dict(), str(save_path) +
                ('/optimizer1_{}_accuracy:{:.3f}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
            torch.save(
                self.optimizer2.state_dict(), str(save_path) +
                ('/optimizer2_{}_accuracy:{:.3f}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))


    def load_state(self, conf, model_path = None, head_path = None):
        # if conf.use_dp == True:
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
            print('model loaded...')
        if head_path is not None:
            self.head.load_state_dict(torch.load(head_path))
            print('head loaded...')
        # else:
        #     if model_path is not None:
        #         self.model.load_state_dict(torch.load(model_path))
        #         print('model loaded...')
        #     if head_path is not None:
        #         self.head.load_state_dict(torch.load(head_path))
        #         print('head loaded...')
