import numpy as np
import pandas as pd
import wandb
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
# from tensorboardX import SummaryWriter
# import pandas as pd
# from sync_batchnorm import convert_model

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
# from utils import get_time, gen_plot, hflip_batch, separate_bn_paras, model_profile
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from verification import evaluate, evaluate_dist
from torchvision.utils import save_image
from Backbone import DAL_model, OECNN_model
from itertools import chain
from utils_txt import cos_dist, fixed_img_list

# for using Partial FC
from partial_fc import *
import losses

class face_learner(object):
    def __init__(self, conf=None, inference=False, load_head=False):

        self.conf = conf
        self.epoch = self.conf.epochs
        if conf.loss == 'DAL':
            if conf.data_mode == 'ms1m':
                self.model = DAL_model(head='cosface', n_cls= 85742, conf=self.conf).to(conf.device)
            else:
                self.model = DAL_model(head='cosface', n_cls= 10572, conf=self.conf).to(conf.device)
            self.trainingDAL = False

        elif conf.loss == 'OECNN':
            if conf.data_mode == 'ms1m':
                self.model = OECNN_model(head='cosface', n_cls= 85742, conf=self.conf).to(conf.device)
            else:
                self.model = OECNN_model(head='cosface', n_cls= 10572, conf=self.conf).to(conf.device)
            self.trainingDAL = False

        else:
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
        print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
        # (self.modemodel_profilel)

        # For Tsne -> you can ignore these codes
        # self.head = Arcface(embedding_size=conf.embedding_size, classnum=11076).to(conf.device)

        if load_head:
            self.head = Arcface(embedding_size=conf.embedding_size, classnum=10572, args=self.conf).to(conf.device)

        if conf.wandb:
            import wandb

            # wandb.init(project=f"Face-Recognition(BMVC2021)")
            wandb.init(
                project='Missing-children', 
                entity="davian-bmvc-face", 
                name=conf.exp, )

        if conf.tensorboard:
            from tensorboardX import SummaryWriter
            # self.writer = SummaryWriter(f'tensorboard_log/{conf.exp}')
            self.writer = SummaryWriter(f'result/summary/{conf.exp}')

        if not inference:
            # self.alpha = conf.alpha
            # self.milestones = [6, 11, 16]
            # self.milestones = [8, 16, 24] # Ours 30 naive
            # self.milestones = [9, 15, 21]
            # self.milestones = [11, 16, 21]
            # self.milestones = [6, 11] # Sphereface paper 28epoch
            # self.milestones = [16, 24, 28] # Cosface paper 30epoch
            self.milestones = [28, 38, 46] # Superlong 50epoch
            if self.conf.data_mode == 'webface':
                self.milestones = [8, 12, 16]
                self.epoch = 20

            if 'baseline_arcface' in self.conf.exp:
            # if self.conf.loss == 'Arcface':
                if self.conf.data_mode == 'ms1m':
                    self.milestones = [8, 14]  # Cosface paper 30epoch
                    self.epoch= 16

                else:
                    self.milestones = [21, 30]  # Cosface paper 30epoch
                    self.epoch= 33
            #
            # if self.conf.loss == 'Cosface':
            #     self.milestones = [16, 25, 29]  # Cosface paper 30epoch
            #     self.epoch= 31

            if self.conf.loss == 'OECNN':
                self.milestones = [9, 15, 18]  # Cosface paper 30epoch
                self.epoch= 21

            if self.conf.loss == 'Curricular' or 'MILE28' in self.conf.exp or 'inter' in self.conf.exp:
                if self.conf.data_mode == 'ms1m':
                    # self.milestones = [14, 19, 23]  # half milestones
                    # self.epoch= 25
                    self.milestones = [28, 38, 46] # Long milestones, epochs
                    self.epoch = 50
                else:
                    self.milestones = [28, 38, 46]  # Cosface paper 30epoch
                    self.epoch= 50

            # if self.conf.loss == 'DAL':
            #     self.milestones = [22, 33, 38]  # Cosface paper 30epoch
            #     self.epoch= 40

            if self.conf.loss == 'Curricular':
                self.milestones = [28, 38, 46]  # Curricular face paper 50epoch

            if self.conf.short_milestone:
                self.milestones = [21, 30]  # Curricular face paper 50epoch
                self.epoch= 33

            if conf.loss == 'Sphere':
                self.milestones = [16, 24]  # Curricular face paper 50epoch
                self.epoch= 28


            self.loader, self.class_num, self.ds, self.child_identity, self.child_identity_min, self.child_identity_max = get_train_loader(conf)
            self.log_path = os.path.join(conf.log_path, conf.data_mode, conf.exp)

            os.makedirs(self.log_path, exist_ok=True)
            # self.writer = SummaryWriter(self.log_path)
            self.step = 0


            if 'MIXUP' in conf.exp:
                self.class_num += conf.new_id

            if conf.loss == 'Arcface':
                self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num, args=self.conf).to(conf.device)
            elif conf.loss == 'PartialFC':
                try:
                    world_size = int(os.environ["WORLD_SIZE"])
                    rank = int(os.environ["RANK"])
                    distributed.init_process_group("nccl")
                except KeyError:
                    world_size = 1
                    rank = 0
                    distributed.init_process_group(
                        backend="nccl",
                        init_method="tcp://127.0.0.1:12584",
                        rank=rank,
                        world_size=world_size,
                    )
                # torch.multiprocessing.spawn(train_fn, args=(world_size,), nprocs=world_size)
                self.model = torch.nn.parallel.DistributedDataParallel(module=self.model, broadcast_buffers=True, device_ids=[0,1,2,3], bucket_cap_mb=16, find_unused_parameters=False)
                self.head = PartialFC(margin_loss=losses.ArcFace(), embedding_size=512, num_classes=self.class_num, sample_rate=1.0)

            elif conf.loss == 'Cosface':
                self.head = CosineMarginProduct(embedding_size=conf.embedding_size, classnum=self.class_num, scale=conf.scale).to(conf.device)
            elif conf.loss == 'Sphere':
                self.head = AngularPenaltySMLoss(in_features=conf.embedding_size, out_features=self.class_num).to(conf.device)
            elif conf.loss == 'LDAM':
                self.head = LDAMLoss(embedding_size=conf.embedding_size, classnum=self.class_num, max_m=conf.max_m, s=conf.scale).to(conf.device)
            elif conf.loss == 'Curricular':
                self.head = CurricularFace(in_features=conf.embedding_size, out_features=self.class_num).to(conf.device)
            elif conf.loss == 'MV-AM':
                self.head = FC(in_feature=conf.embedding_size, out_feature=self.class_num, fc_type='MV-AM').to(conf.device)
            elif conf.loss == 'MV-Arc':
                self.head = FC(in_feature=conf.embedding_size, out_feature=self.class_num, fc_type='MV-Arc').to(conf.device)
            elif conf.loss == 'Broad':
                self.head = BroadFaceArcFace(in_features=conf.embedding_size, out_features=10572).to(conf.device)
                root_path = 'work_space/models_serious/interclass_MSE_proto1_broad_scratch/train_final'
                model_path = os.path.join(root_path, 'fgnetc_best_model_2021-07-01-00-29_accuracy:0.500_step:275976_casia_interclass_MSE_proto1_broad_scratch.pth')
                head_path = os.path.join(root_path, 'fgnetc_best_head_2021-07-01-00-29_accuracy:0.500_step:275976_casia_interclass_MSE_proto1_broad_scratch.pth')
                self.model.load_state_dict(torch.load(model_path))
                self.head.load_state_dict(torch.load(head_path))
                print('broad face loaded ...')
                self.milestones = [13, 18]
                self.epoch = 20
                # import pdb; pdb.set_trace()
            # else:
            #     import sys
            #     print('wrong loss function.. exiting...')
            #     sys.exit(0)
            if conf.use_pretrain or conf.use_adult_memory_pretrain:
                root_path = 'work_space/models_serious/baseline_arcface_4885'
                # model_path = os.path.join(root_path, 'fgnetc_best_model_2021-06-11-12-06_accuracy:0.864_step:240000_casia_baseline_arcface_4885.pth')
                # head_path = os.path.join(root_path, 'fgnetc_best_head_2021-06-11-12-06_accuracy:0.864_step:240000_casia_baseline_arcface_4885.pth')
                # self.milestones = [6, 3, 1]  # Superlong 50epoch
                # self.epoch = 10

                model_path = os.path.join(root_path, 'fgnetc_best_model_2021-06-11-12-06_accuracy:0.864_step:240000_casia_baseline_arcface_4885.pth')
                head_path = os.path.join(root_path, 'fgnetc_best_head_2021-06-11-12-06_accuracy:0.864_step:240000_casia_baseline_arcface_4885.pth')
                self.milestones = [6, 9, 11]  # Superlong 50epoch
                self.epoch = 15


                self.model.load_state_dict(torch.load(model_path))
                self.head.load_state_dict(torch.load(head_path))

                print('model loaded ...')
                print(self.milestones)
                print(self.epoch)

            # Currently use Data Parallel as default
            if conf.use_dp:
                self.model = nn.DataParallel(self.model)
                self.head = nn.DataParallel(self.head)

                # if conf.use_sync == True:
                #     self.model = convert_model(self.model)

            print(f'curr milestones: {self.milestones}')
            print(f'total epochs: {self.epoch}')

            print(self.class_num)
            print(conf)

            print('two model heads generated')

            if conf.loss == 'DAL':
                self.optbb = optim.SGD(chain(self.model.age_classifier.parameters(),
                                             self.model.RFM.parameters(),
                                             self.model.margin_fc.parameters(),
                                             self.model.backbone.parameters()), lr=conf.lr, momentum=0.9)
                self.optDAL = optim.SGD(self.model.DAL.parameters(), lr=conf.lr, momentum=0.9)

            elif conf.loss == 'OECNN':
                self.optimizer = optim.SGD(chain(self.model.age_classifier.parameters(),
                                             self.model.margin_fc.parameters(),
                                             self.model.backbone.parameters()), lr=conf.lr, momentum=0.9)
            else:
                paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)

                if conf.loss == 'PartialFC':
                    self.optimizer1 = optim.SGD([{'params': paras_wo_bn + list(self.head.parameters()), 'weight_decay': 5e-4},], lr=conf.lr, momentum=conf.momentum)

                    self.optimizer2 = optim.SGD([
                        {'params': paras_only_bn}
                    ], lr=conf.lr, momentum=conf.momentum)

                else:
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
            self.evaluate_every = len(self.loader)
            # self.evaluate_every = conf.evaluate_freq
            self.save_every = conf.save_freq

            print(conf)
            print('training starts.... BMVC 2021....')

            # # dataset_root= os.path.join('/home/nas4_user/jungsoolee/Face_dataset/face_emore2')
            # dataset_root= os.path.join('./dataset/face_emore2')
            # # dataset_root= os.path.join(conf.home, 'dataset/face_emore2')
            # # self.lfw, self.lfw_issame = get_val_data(dataset_root)
            # # dataset_root = "./dataset/"
            # self.lfw = np.load(os.path.join(dataset_root, "lfw_align_112_list.npy")).astype(np.float32)
            # self.lfw_issame = np.load(os.path.join(dataset_root, "lfw_align_112_label.npy"))
            #
            # self.fgnetc = np.load(os.path.join(dataset_root, "FGNET_new_align_list.npy")).astype(np.float32)
            # self.fgnetc_issame = np.load(os.path.join(dataset_root, "FGNET_new_align_label.npy"))
            # # self.cfp_fp, self.cfp_fp_issame = get_val_data(dataset_root, 'cfp_fp')
            # # self.agedb, self.agedb_issame = get_val_data(dataset_root, 'agedb_30')

            self.fgnet10_best_acc, self.fgnet20_best_acc, self.fgnet30_best_acc, self.average_best_acc = 0.0, 0.0, 0.0, 0.0
            self.agedb10_best_acc, self.agedb20_best_acc, self.agedb30_best_acc, self.lag_best_acc= 0.0, 0.0, 0.0, 0.0

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
    def evaluate_prev(self, conf, carray, issame, nrof_folds = 10, tta = True):
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

    def evaluate_dal_prev(self, conf, carray, issame, nrof_folds=10, tta=True):
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(carray), conf.embedding_size])
        with torch.no_grad():
            while idx + conf.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + conf.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model.inference(batch.to(conf.device)).cpu() + self.model.inference(fliped.to(conf.device)).cpu()
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch).cpu()
                else:
                    embeddings[idx:idx + conf.batch_size] = self.model.inference(batch.to(conf.device)).cpu()
                idx += conf.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model.inference(batch.to(conf.device)).cpu() + self.model.inference(fliped.to(conf.device)).cpu()
                    embeddings[idx:] = l2_norm(emb_batch).cpu()
                else:
                    embeddings[idx:] = self.model.inference(batch.to(conf.device)).cpu()

        tpr, fpr, accuracy, best_thresholds, dist = evaluate_dist(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = transforms.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor, dist

    def control_text_list(self, txt_root, txt_dir, data_dir=None):
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

    def verification(self, net, label_list, pair_list, transform, data_dir=None):
        similarities = []
        labels = []
        assert len(label_list) == len(pair_list)

        trans_list = []
        trans_list += [transforms.ToTensor()]
        trans_list += [transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
        t = transforms.Compose(trans_list)

        if len(label_list) == 0:
            return 0, 0, 0
        net.eval()
        with torch.no_grad():  # Test 때 GPU를 사용할 경우 메모리 절약을 위해 torch.no_grad() 내에서 하는 것이 좋다.
            for idx, pair in enumerate(tqdm(pair_list)):
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

                    # if 'png' in pair:
                    #     path_1, path_2 = pair.split('.png ')
                    #     path_1 = path_1 + '.png'
                    #     path_2 = path_2[:-1]
                    # elif 'jpg' in pair:
                    #     path_1, path_2 = pair.split('.jpg ')
                    #     path_1 = path_1 + '.jpg'
                    #     path_2 = path_2[:-1]
                    # elif 'JPG' in pair:
                    #     path_1, path_2 = pair.split('.JPG ')
                    #     path_1 = path_1 + '.JPG'
                    #     path_2 = path_2[:-1]

                elif data_dir == 'cacd_vs':
                    image_root = '/home/nas4_user/jungsoolee/Face_dataset/CACD_VS_single_112_RF'
                    path_1, path_2 = pair.split(' ')
                    path_1 = os.path.join(image_root, path_1)
                    path_2 = os.path.join(image_root, path_2)

                elif data_dir == 'morph':
                    image_root = '/home/nas4_user/jungsoolee/Face_dataset/Album2_single_112_RF'
                    path_1, path_2 = pair.split(' ')
                    path_1 = os.path.join(image_root, path_1)
                    path_2 = os.path.join(image_root, path_2)

                img_1 = t(Image.open(path_1)).unsqueeze(dim=0).to(self.conf.device)
                img_2 = t(Image.open(path_2)).unsqueeze(dim=0).to(self.conf.device)
                imgs = torch.cat((img_1, img_2), dim=0)

                # Extract feature and save
                if self.conf.loss == 'DAL' or self.conf.loss == 'OECNN':
                    features = net(imgs, emb=True)
                else:
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

    def evaluate_new_total(self):

        trans_list = []
        trans_list += [transforms.ToTensor()]
        trans_list += [transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
        t = transforms.Compose(trans_list)

        txt_root = '/home/nas4_user/jungsoolee/Face_dataset/txt_files'
        # txt_root = '/home/nas1_temp/jooyeolyun/Datasets/FaceRecog_txt'
        # txt_root = './dataset/txt_files'
        # txt_root = './dataset/761-testset'

        # if self.conf.dfc:
        #     txt_root = './dataset/../bebeto_test/761-testset-revised2'
        # else:
        #     txt_root = '../bebeto_face_dataset/761-testset-revised'

        # txt_dir = 'fgnet10_child.txt'
        # print(f'working on : {txt_dir}')
        # pair_list, label_list = self.control_text_list(txt_root, txt_dir)
        # fgnet10_best_acc, fgnet10_best_th, fgnet10_idx = self.verification(self.model, label_list, pair_list, transform=t)
        # # fgnet10_best_acc, fgnet10_best_th, fgnet10_idx = 0.5, 0.5, 0.5
        # print(f'txt_dir: {txt_dir}, best_accr: {fgnet10_best_acc}')
        # if fgnet10_best_acc > self.fgnet10_best_acc:
        #     self.fgnet10_best_acc = fgnet10_best_acc
        #     print('saving best fgnet10_best_acc model....')
        #     self.save_best_state_new(self.conf, 'fgnet10', self.fgnet10_best_acc, extra=str(self.conf.data_mode) + '_' + str(self.conf.exp))

        txt_dir = 'fgnet20_child.txt'
        print(f'working on : {txt_dir}')
        pair_list, label_list = self.control_text_list(txt_root, txt_dir)
        fgnet20_best_acc, fgnet20_best_th, fgnet20_idx = self.verification(self.model, label_list, pair_list, transform=t)
        # fgnet20_best_acc, fgnet20_best_th, fgnet20_idx = 0.5, 0.5, 0.5
        print(f'txt_dir: {txt_dir}, best_accr: {fgnet20_best_acc}')
        if fgnet20_best_acc > self.fgnet20_best_acc:
            self.fgnet20_best_acc = fgnet20_best_acc
            print('saving best fgnet20_best_acc model....')
            self.save_best_state_new(self.conf, 'fgnet20', self.fgnet20_best_acc, extra=str(self.conf.data_mode) + '_' + str(self.conf.exp))

        txt_dir = 'fgnet30_child.txt'
        print(f'working on : {txt_dir}')
        pair_list, label_list = self.control_text_list(txt_root, txt_dir)
        fgnet30_best_acc, fgnet30_best_th, fgnet30_idx = self.verification(self.model, label_list, pair_list, transform=t)
        # fgnet30_best_acc, fgnet30_best_th, fgnet30_idx = 0.5, 0.5, 0.5
        print(f'txt_dir: {txt_dir}, best_accr: {fgnet30_best_acc}')
        if fgnet30_best_acc > self.fgnet30_best_acc:
            self.fgnet30_best_acc = fgnet30_best_acc
            print('saving best fgnet30_best_acc model....')
            self.save_best_state_new(self.conf, 'fgnet30', self.fgnet30_best_acc, extra=str(self.conf.data_mode) + '_' + str(self.conf.exp))

        # txt_dir = 'agedb10_child.txt'
        # print(f'working on : {txt_dir}')
        # pair_list, label_list = self.control_text_list(txt_root, txt_dir)
        # agedb10_best_acc, agedb10_best_th, agedb10_idx = self.verification(self.model, label_list, pair_list,transform=t)
        # # agedb10_best_acc, agedb10_best_th, agedb10_idx =  0.5, 0.5, 0.5
        # print(f'txt_dir: {txt_dir}, best_accr: {fgnet10_best_acc}')
        # if agedb10_best_acc > self.agedb10_best_acc:
        #     self.agedb10_best_acc = agedb10_best_acc
        #     print('saving best agedb10_best_acc model....')
        #     self.save_best_state_new(self.conf, 'agedb10', self.agedb10_best_acc, extra=str(self.conf.data_mode) + '_' + str(self.conf.exp))

        txt_dir = 'agedb20_child.txt'
        print(f'working on : {txt_dir}')
        pair_list, label_list = self.control_text_list(txt_root, txt_dir)
        agedb20_best_acc, agedb20_best_th, agedb20_idx = self.verification(self.model, label_list, pair_list, transform=t)
        # agedb20_best_acc, agedb20_best_th, agedb20_idx = 0.5, 0.5, 0.5
        print(f'txt_dir: {txt_dir}, best_accr: {agedb20_best_acc}')
        if agedb20_best_acc > self.agedb20_best_acc:
            self.agedb20_best_acc = agedb20_best_acc
            print('saving best agedb20_best_acc model....')
            self.save_best_state_new(self.conf, 'agedb20', self.agedb20_best_acc, extra=str(self.conf.data_mode) + '_' + str(self.conf.exp))

        txt_dir = 'agedb30_child.txt'
        print(f'working on : {txt_dir}')
        pair_list, label_list = self.control_text_list(txt_root, txt_dir)
        agedb30_best_acc, agedb30_best_th, agedb30_idx = self.verification(self.model, label_list, pair_list, transform=t)
        # agedb30_best_acc, agedb30_best_th, agedb30_idx = 0.5, 0.5, 0.5
        print(f'txt_dir: {txt_dir}, best_accr: {agedb30_best_acc}')
        if agedb30_best_acc > self.agedb30_best_acc:
            self.agedb30_best_acc = agedb30_best_acc
            print('saving best agedb30_best_acc model....')
            self.save_best_state_new(self.conf, 'agedb30', self.agedb30_best_acc, extra=str(self.conf.data_mode) + '_' + str(self.conf.exp))

        # txt_dir = 'lag.txt'
        # print(f'working on : {txt_dir}')
        # pair_list, label_list = self.control_text_list(txt_root, txt_dir)
        # lag_best_acc, lag_best_th, lag_idx = self.verification(self.model, label_list, pair_list, transform=t)
        # # lag_best_acc, lag_best_th, lag_idx = 0.5, 0.5, 0.5
        # print(f'txt_dir: {txt_dir}, best_accr: {lag_best_acc}')
        # if lag_best_acc > self.lag_best_acc:
        #     self.lag_best_acc = lag_best_acc
        #     print('saving best lag_best_acc model....')
        #     self.save_best_state_new(self.conf, 'lag', self.lag_best_acc, extra=str(self.conf.data_mode) + '_' + str(self.conf.exp))

        # average_acc = (fgnet10_best_acc + fgnet20_best_acc + fgnet30_best_acc + agedb10_best_acc + agedb20_best_acc + agedb30_best_acc + lag_best_acc) / 7
        # if average_acc > self.average_best_acc:
        #     self.average_best_acc = average_acc
        #     print('saving best average model....')
        #     self.save_best_state_new(self.conf, 'average', self.average_best_acc, extra=str(self.conf.data_mode) + '_' + str(self.conf.exp))

        if self.conf.wandb:
            wandb.log({
                # "fgnet10_acc": fgnet10_best_acc,
                "fgnet20_acc": fgnet20_best_acc,
                "fgnet30_acc": fgnet30_best_acc,
                # "agedb10_acc": agedb10_best_acc,
                "agedb20_acc": agedb20_best_acc,
                "agedb30_acc": agedb30_best_acc,
                # "lag_acc": lag_best_acc,
            }, step=self.step)

        if self.conf.wandb:
            wandb.log({
                # "best_fgnet10_acc": self.fgnet10_best_acc,
                "best_fgnet20_acc": self.fgnet20_best_acc,
                "best_fgnet30_acc": self.fgnet30_best_acc,
                # "best_agedb10_acc": self.agedb10_best_acc,
                "best_agedb20_acc": self.agedb20_best_acc,
                "best_agedb30_acc": self.agedb30_best_acc,
                # "best_lag_acc": self.lag_best_acc,
            }, step=self.step)

        if self.conf.tensorboard:
            self.writer.add_scalar("acc/fgnet20_acc", fgnet20_best_acc, self.step)
            self.writer.add_scalar("acc/fgnet30_acc", fgnet30_best_acc, self.step)
            self.writer.add_scalar("acc/agedb20_acc", agedb20_best_acc, self.step)
            self.writer.add_scalar("acc/agedb30_acc", agedb30_best_acc, self.step)

            self.writer.add_scalar("acc/best_fgnet20_acc", self.fgnet20_best_acc, self.step)
            self.writer.add_scalar("acc/best_fgnet30_acc", self.fgnet30_best_acc, self.step)
            self.writer.add_scalar("acc/best_agedb20_acc", self.agedb20_best_acc, self.step)
            self.writer.add_scalar("acc/best_agedb30_acc", self.agedb30_best_acc, self.step)

        print(f'fgnet20: {self.fgnet20_best_acc} || fgnet30: {self.fgnet30_best_acc} || agedb20: {self.agedb20_best_acc} || agedb30: {self.agedb30_best_acc}')

    def evaluate_new(self):
        trans_list = []
        trans_list += [transforms.ToTensor()]
        trans_list += [transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
        t = transforms.Compose(trans_list)

        txt_root = '/home/nas4_user/jungsoolee/Face_dataset/txt_files'
        # txt_root = '/home/nas1_temp/jooyeolyun/Datasets/FaceRecog_txt'
        # txt_root = '/home/nas4_user/jungsoolee/Missing-children/txt_files_sh'
        # txt_root = './dataset/txt_files_sh'
        # if self.conf.dfc:
        #     txt_root = './dataset/../bebeto_test/761-testset-revised2'
        # else:
        #     txt_root = '../bebeto_face_dataset/761-testset-revised'

        txt_dir = 'agedb30_child.txt'
        print(f'working on : {txt_dir}')
        pair_list, label_list = self.control_text_list(txt_root, txt_dir)
        agedb30_best_acc, agedb30_best_th, agedb30_idx = self.verification(self.model, label_list, pair_list, transform=t)
        # fgnet30_best_acc, fgnet30_best_th, fgnet30_idx = 0.5, 0.5, 0.5
        print(f'txt_dir: {txt_dir}, best_accr: {agedb30_best_acc}')
        if agedb30_best_acc > self.agedb30_best_acc:
            self.agedb30_best_acc = agedb30_best_acc
            print('saving best agedb30_best_acc model....')
            self.save_best_state_new(self.conf, 'agedb30', self.agedb30_best_acc, extra=str(self.conf.data_mode) + '_' + str(self.conf.exp))

        if self.conf.wandb:
            wandb.log({
                "agedb30_acc": agedb30_best_acc,
            }, step=self.step)

        if self.conf.tensorboard:
            self.writer.add_scalar("acc/agedb30_acc", agedb30_best_acc, self.step)

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
            # for imgs, labels, ages in iter(self.loader):
            # for imgs, labels in tqdm(iter(self.loader)):
                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()
                if imgs.shape[0] == 1:
                    continue

                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)

                embeddings = self.model(imgs)
                thetas = self.head(embeddings, labels)
                # thetas = self.head(embeddings, labels, ages)
                if self.conf.loss == 'Broad' or self.conf.loss == 'Sphere' or self.conf.loss =='PartialFC':
                    loss= thetas
                else:
                    loss = ce_loss(thetas, labels)

                childs = (ages == 0)
                if conf.weighted_ce and torch.sum(childs) > 0:
                    child_thetas = thetas[childs]
                    child_labels = labels[childs]
                    adult_thetas = thetas[~childs]
                    adult_labels = labels[~childs]
                    # child images: 9326, adult images: 481297
                    # child_ce = conf.reweight_ratio * 51.6080849239 * ce_loss(child_thetas, child_labels)
                    child_ce = ce_loss(child_thetas, child_labels)
                    # adult_ce = ce_loss(adult_thetas, adult_labels)
                    adult_ce = conf.reweight_ratio * ce_loss(adult_thetas, adult_labels)


                    loss= child_ce + adult_ce

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

                    if self.conf.tensorboard:
                        self.writer.add_scalar("loss/train_loss", loss_board, self.step)

                    running_loss = 0.

                # if self.step == 0 and self.conf.evaluate_debugging:
                #     self.evaluate_new_total()
                #     print('evaluating debugging finished...')

                # added wrong on evaluations
                if e >= self.milestones[1] and self.step % self.evaluate_every == 0 and self.step != 0:
                # if e >= self.milestones[1] and self.step % self.evaluate_every == 0:
                    self.model.eval()
                    self.evaluate_new_total()
                    print('evaluating....')
                    self.model.train()

                elif e < self.milestones[1] and self.step % self.evaluate_every == 0 and self.step != 0:
                # elif e < self.milestones[1] and self.step % self.evaluate_every == 0 and self.step > 0:
                    self.model.eval()
                    self.evaluate_new()
                    print('evaluating....')
                    self.model.train()
                #
                # self.model.eval()
                # self.evaluate_new_total()
                # self.evaluate_new()
                # self.model.train()


                self.step += 1

        if conf.wandb:
            wandb.finish()


    def set_train_mode(self, state):
        self.trainingDAL = not state
        self.set_grads(self.model.RFM, state)
        self.set_grads(self.model.backbone, state)
        self.set_grads(self.model.margin_fc, state)
        self.set_grads(self.model.age_classifier, state)
        self.set_grads(self.model.DAL, not state)


    def set_grads(self,mod, state):
        for para in mod.parameters():
            para.requires_grad = state

    def flip_grads(self,mod):
        for para in mod.parameters():
            if para.requires_grad:
                para.grad = - para.grad

    def train_dal(self, conf, epochs):
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

                if imgs.shape[0] == 1:
                    continue

                if self.step % 70 == 0:  # canonical maximization procesure
                    self.set_train_mode(False)
                elif self.step % 70 == 20:  # RFM optimize procesure
                    self.set_train_mode(True)

                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)
                ages = ages.to(conf.device)

                idLoss, id_acc, ageLoss, age_acc, cc = self.model(imgs, labels, ages)
                total_loss = idLoss + ageLoss * 0.1 + cc * 0.1
                if self.trainingDAL:
                    self.optDAL.zero_grad()
                    total_loss.backward()
                    self.flip_grads(self.model.DAL)
                    self.optDAL.step()
                else:
                    self.optbb.zero_grad()
                    total_loss.backward()
                    self.optbb.step()

                if self.step % self.board_loss_every == 0 and self.step != 0:  # XXX
                    # print('wandb plotting....')
                    # loss_board = running_loss / self.board_loss_every
                    # self.writer.add_scalar('train_loss', loss_board, self.step)
                    if self.conf.wandb:
                        wandb.log({
                            'DAL_idLoss': idLoss,
                            'DAL_id_acc': id_acc,
                            'DAL_ageLoss': ageLoss,
                            'DAL_age_acc': age_acc,
                        }, step=self.step)

                    if self.conf.tensorboard:
                        self.writer.add_scalar("loss/DAL_idLoss", idLoss, self.step)
                        self.writer.add_scalar("loss/DAL_id_acc", id_acc, self.step)
                        self.writer.add_scalar("loss/DAL_ageLoss", ageLoss, self.step)
                        self.writer.add_scalar("loss/DAL_age_acc", age_acc, self.step)


                    running_loss = 0.

                # added wrong on evaluations
                # if e >= self.milestones[1] and self.step % self.evaluate_every == 0 and self.step != 0:
                if e >= self.milestones[1] and self.step % self.evaluate_every == 0:
                    self.model.eval()
                    self.evaluate_new_total()
                    print('evaluating....')
                    self.model.train()

                # elif e < self.milestones[1] and self.step % self.evaluate_every == 0 and self.step != 0:
                elif e < self.milestones[1] and self.step % self.evaluate_every == 0:
                    self.model.eval()
                    self.evaluate_new()
                    print('evaluating....')
                    self.model.train()
                self.step += 1

        # if conf.finetune_model_path is not None:
        #     self.save_state(conf, fgnetc_accuracy, to_save_folder=True, extra=str(conf.data_mode)  + '_' + str(conf.net_depth) + '_'+ str(conf.batch_size) +'_finetune')
        # else:
        #     self.save_state(conf, fgnetc_accuracy, to_save_folder=True, extra=str(conf.data_mode)  + '_' + str(conf.net_depth) + '_'+ str(conf.batch_size) +'_final')
        if conf.wandb:
            wandb.finish()

    def train_oecnn(self, conf, epochs):
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

                if imgs.shape[0] == 1:
                    continue

                self.optimizer.zero_grad()
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)
                ages = ages.to(conf.device)

                idLoss, id_acc, ageLoss, age_acc = self.model(imgs, labels, ages)
                total_loss = idLoss + conf.oecnn_lambda * ageLoss
                total_loss.backward()
                self.optimizer.step()

                if self.step % self.board_loss_every == 0 and self.step != 0:  # XXX
                    # print('wandb plotting....')
                    # loss_board = running_loss / self.board_loss_every
                    # self.writer.add_scalar('train_loss', loss_board, self.step)
                    if self.conf.wandb:
                        wandb.log({
                            'DAL_idLoss': idLoss,
                            'DAL_id_acc': id_acc,
                            'DAL_ageLoss': ageLoss,
                            'DAL_age_acc': age_acc,
                        }, step=self.step)

                    if self.conf.tensorboard:
                        self.writer.add_scalar("loss/DAL_idLoss", idLoss, self.step)
                        self.writer.add_scalar("loss/DAL_id_acc", id_acc, self.step)
                        self.writer.add_scalar("loss/DAL_ageLoss", ageLoss, self.step)
                        self.writer.add_scalar("loss/DAL_age_acc", age_acc, self.step)


                    running_loss = 0.

                if self.step == 0:
                    self.evaluate_new_total()
                    print('evaluating debugging finished...')

                # if e >= self.milestones[1] and self.step % self.evaluate_every == 0 and self.step != 0:
                if e >= self.milestones[1] and self.step % self.evaluate_every == 0:
                    self.model.eval()
                    self.evaluate_new_total()
                    print('evaluating....')
                    self.model.train()

                # elif e < self.milestones[1] and self.step % self.evaluate_every == 0 and self.step != 0:
                elif e < self.milestones[1] and self.step % self.evaluate_every == 0:
                    self.model.eval()
                    self.evaluate_new()
                    print('evaluating....')
                    self.model.train()
                self.step += 1

        # if conf.finetune_model_path is not None:
        #     self.save_state(conf, fgnetc_accuracy, to_save_folder=True, extra=str(conf.data_mode)  + '_' + str(conf.net_depth) + '_'+ str(conf.batch_size) +'_finetune')
        # else:
        #     self.save_state(conf, fgnetc_accuracy, to_save_folder=True, extra=str(conf.data_mode)  + '_' + str(conf.net_depth) + '_'+ str(conf.batch_size) +'_final')
        if conf.wandb:
            wandb.finish()
    # training with memory bank
    def train_prototype(self, conf, epochs):
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
        child_loss = 0.0
        mixup_loss = torch.tensor(0.0)
        self.child_labels = torch.tensor(self.child_identity).cuda()
        fgnetc_best_acc = 0.0
        if conf.prototype_loss == 'MSE':
            criterion_prototype = nn.MSELoss()
        elif conf.prototype_loss == 'CE':
            criterion_prototype = nn.CrossEntropyLoss()
        else:
            criterion_prototype = nn.L1Loss()
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
                thetas = self.head(embeddings, labels)
                # thetas = self.head(embeddings, labels, ages)
                if self.conf.loss == 'Broad' or self.conf.loss == 'Sphere':
                    arcface_loss = thetas
                else:
                    arcface_loss = ce_loss(thetas, labels)

                # arcface_loss = ce_loss(thetas, labels)

                # new prototype method
                if self.conf.prototype_mode == 'all':
                    child_lambda = self.conf.lambda_child * 1
                elif self.conf.prototype_mode == 'zero':
                    child_lambda = 0.0 if e > self.milestones[0] else self.conf.lambda_child * 1.0
                elif self.conf.prototype_mode == 'second':
                    child_lambda = 1.0 if e > self.milestones[0] else self.conf.lambda_child * 0.0

                # self.head.kernel = (512, # of classes)
                if conf.data_mode == 'ms1m' or conf.data_mode == 'ms1m_cctv':
                    kernel = self.head.kernel[:, self.child_identity]
                    if conf.loss == 'Cosface' or conf.loss == 'MV-AM' or conf.loss == 'Broad':
                        prototype_matrix = torch.mm(l2_norm(kernel, axis=0), l2_norm(kernel, axis=0).T)
                    else:
                        prototype_matrix = torch.mm(l2_norm(kernel, axis=0).T, l2_norm(kernel, axis=0))
                else:
                    if conf.loss == 'Cosface' or conf.loss == 'MV-AM' or conf.loss == 'Broad':
                        prototype_matrix = torch.mm(l2_norm(self.head.kernel, axis=0), l2_norm(self.head.kernel, axis=0).T)
                    else:
                        prototype_matrix = torch.mm(l2_norm(self.head.kernel, axis=0).T, l2_norm(self.head.kernel, axis=0))
                    prototype_matrix = prototype_matrix[:, self.child_identity]
                    prototype_matrix = prototype_matrix[self.child_identity, :]

                if conf.prototype_loss == 'CE':
                    prototype_label = torch.arange(prototype_matrix.shape[0]).to(conf.device)
                else:
                    prototype_label = torch.eye(prototype_matrix.shape[0]).to(conf.device)

                child_loss = criterion_prototype(prototype_matrix, prototype_label)
                child_total_loss = child_lambda * child_loss

                childs = (ages == 0)
                if conf.weighted_ce and torch.sum(childs) > 0:
                    child_thetas = thetas[childs]
                    child_labels = labels[childs]
                    adult_thetas = thetas[~childs]
                    adult_labels = labels[~childs]
                    # child images: 9326, adult images: 481297
                    # child_ce = conf.reweight_ratio * 51.6080849239 * ce_loss(child_thetas, child_labels)
                    # child_ce = conf.reweight_ratio * 51.6080849239 * ce_loss(child_thetas, child_labels)
                    child_ce = ce_loss(child_thetas, child_labels)
                    # adult_ce = ce_loss(adult_thetas, adult_labels)
                    adult_ce = conf.reweight_ratio * ce_loss(adult_thetas, adult_labels)
                    arcface_loss= child_ce + adult_ce
                loss = arcface_loss + child_total_loss

                loss.backward()
                running_loss += loss.item()

                running_arcface_loss += arcface_loss.item()
                # if 'POSITIVE' in conf.exp:
                running_child_loss += child_loss.item()
                running_child_total_loss += child_total_loss.item()

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

                    if self.conf.wandb:
                        wandb.log({
                            "train_loss": loss_board,
                            "arcface_total_loss": arcface_loss_board,
                        }, step=self.step)

                    if self.conf.tensorboard:
                        self.writer.add_scalar('train_loss', loss_board, self.step)
                        self.writer.add_scalar('arcface_total_loss', arcface_loss_board, self.step)


                    child_loss_board = running_child_loss / self.board_loss_every
                    child_total_loss_board = running_child_total_loss / self.board_loss_every

                    if self.conf.wandb:
                        wandb.log({
                            "child_loss": child_loss_board,
                            "child_total_loss": child_total_loss_board,
                            "child_lambda": child_lambda,
                        }, step=self.step)

                        if self.conf.tensorboard:
                            self.writer.add_scalar('child_loss', child_loss_board, self.step)
                            self.writer.add_scalar('child_total_loss', child_total_loss_board, self.step)
                            self.writer.add_scalar('child_lambda', child_lambda, self.step)


                    running_loss = 0.
                    running_arcface_loss = 0.0
                    running_child_loss = 0.0
                    running_child_total_loss = 0.0
                    running_mixup_total_loss = 0.0

                # if e >= self.milestones[1] and self.step % self.evaluate_every == 0 and self.step != 0:
                if e >= self.milestones[1] and self.step % self.evaluate_every == 0:
                    self.model.eval()
                    self.evaluate_new_total()
                    print('evaluating....')
                    self.model.train()

                # elif e < self.milestones[1] and self.step % self.evaluate_every == 0 and self.step != 0:
                elif e < self.milestones[1] and self.step % self.evaluate_every == 0:
                    self.model.eval()
                    self.evaluate_new()
                    print('evaluating....')
                    self.model.train()

                self.step += 1

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
        if self.conf.loss == 'DAL':
            for params in self.optbb.param_groups:
                params['lr'] /= 10
            for params in self.optDAL.param_groups:
                params['lr'] /= 10
            print(self.optbb)
            print(self.optDAL)
        elif self.conf.loss == 'OECNN':
            for params in self.optimizer.param_groups:
                params['lr'] /= 10
            print(self.optimizer)
        else:
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
        # torch.save(self.child_memory, os.path.join(save_path, ('fgnetc_best_child_{}_accuracy:{:.3f}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra))))
        # torch.save(self.adult_memory, os.path.join(save_path, ('fgnetc_best_adult_{}_accuracy:{:.3f}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra))))
        if not model_only:
            torch.save(
                self.head.state_dict(), os.path.join(save_path, ('fgnetc_best_head_{}_accuracy:{:.3f}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra))))
            # torch.save(
            #     self.optimizer1.state_dict(), os.path.join( save_path, ('lfw_best_optimizer1_{}_accuracy:{:.3f}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra))))
            # torch.save(
            #     self.optimizer2.state_dict(), os.path.join( save_path), ('lfw_best_optimizer2_{}_accuracy:{:.3f}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))

    def save_best_state_new(self, conf, test_dir, accuracy, to_save_folder=False, extra=None, model_only=False):
        save_path = os.path.join(conf.model_path, conf.exp, test_dir)
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_path, ('fgnetc_best_model_{}_accuracy:{:.3f}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra))))
        # if not model_only:
        #     torch.save(self.head.state_dict(), os.path.join(save_path, ('fgnetc_best_head_{}_accuracy:{:.3f}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra))))

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
