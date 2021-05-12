from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from torchvision import transforms
import pdb

import os
def get_config(training = True):
    conf = edict()
    conf.data_path = Path('data')

    conf.work_path = Path('work_space/')
    conf.model_path = conf.work_path/'models'
    conf.log_path = conf.work_path/'log'
    conf.save_path = '/home/nas1_userE/jungsoolee/BMVC/pretrained/'

    conf.batch_size = 64 # irse net depth 50
    conf.batch_size_d = 64 # child 4 + adult 4 = 8 XXX
#   conf.batch_size = 200 # mobilefacenet
    conf.input_size = [112,112]
    conf.embedding_size = 512
    conf.use_mobilfacenet = False
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.net_mode = 'ir_se' # or 'ir'

    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf.use_dp = False # XXX: Must Turn off!

    conf.oversample_by = 1
    conf.test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    conf.train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    conf.exp = str(conf.net_depth)
    # conf.data_mode = 'vgg'
    conf.data_mode = 'casia'
    # conf.data_mode = 'ms1m'
    conf.resume_analysis = False
    conf.finetune_model_path = None
    conf.discriminator = False
    conf.model_name = ''

    # conf.vgg_folder = '/home/nas1_userE/Face_dataset/faces_vgg_112'
    conf.vgg_folder = '/home/nas1_userE/jungsoolee/Face_dataset/Vgg_age_label'
    # conf.vgg_folder = '/home/nas1_userD/yonggyu/Face_dataset/vgg'
    conf.ms1m_folder = '/home/nas1_userE/jungsoolee/Face_dataset/ms1m-refined-112'
    conf.emore_folder = '/home/nas1_userE/jungsoolee/Face_dataset/faces_emore'
    conf.agedb_folder = '/home/nas1_userE/jungsoolee/Face_dataset/AgeDB_new_align'
    conf.agedb_balanced_folder = '/home/nas1_temp/jooyeolyun/AgeDB_balanced'
    conf.insta_folder = '/home/nas1_userD/yonggyu/Face_dataset/instagram'
    # conf.casia_folder = '/home/nas1_userD/yonggyu/Face_dataset/casia'
    conf.casia_folder = '/home/nas1_userE/jungsoolee/Face_dataset/CASIA_REAL_NATIONAL'
    conf.casia_babymonster60_folder = '/home/nas1_userE/jungsoolee/Face_dataset/CASIA_REAL_BabyMonster60'
    conf.casia_babymonster100_folder = '/home/nas1_userE/jungsoolee/Face_dataset/CASIA_REAL_BabyMonster100'
    conf.casia_babymonster300_folder = '/home/nas1_userE/jungsoolee/Face_dataset/CASIA_REAL_BabyMonster300'
    conf.casia_babymonster500_folder = '/home/nas1_userE/jungsoolee/Face_dataset/CASIA_REAL_BabyMonster500'
    conf.casia_prettymonster60_folder = '/home/nas1_userE/jungsoolee/Face_dataset/CASIA_REAL_PrettyMonster60'
    conf.casia_prettymonster100_folder = '/home/nas1_userE/jungsoolee/Face_dataset/CASIA_REAL_PrettyMonster100'
    conf.casia_prettymonster300_folder = '/home/nas1_userE/jungsoolee/Face_dataset/CASIA_REAL_PrettyMonster300'
    conf.casia_prettymonster500_folder = '/home/nas1_userE/jungsoolee/Face_dataset/CASIA_REAL_PrettyMonster500'
    conf.casia_prettiermonster60_folder = '/home/nas1_userE/jungsoolee/Face_dataset/CASIA_REAL_PrettierMonster60'
    conf.casia_prettiermonster100_folder = '/home/nas1_userE/jungsoolee/Face_dataset/CASIA_REAL_PrettierMonster100'
    conf.casia_prettiermonster300_folder = '/home/nas1_userE/jungsoolee/Face_dataset/CASIA_REAL_PrettierMonster300'
    conf.casia_prettiermonster500_folder = '/home/nas1_userE/jungsoolee/Face_dataset/CASIA_REAL_PrettierMonster500'

    conf.child_folder = "/home/nas1_userD/yonggyu/Face_dataset/domain_cls_10000_d/0"
    conf.adult_folder = "/home/nas1_userD/yonggyu/Face_dataset/domain_cls_10000_d/1"

#--------------------Training Config ------------------------    
    if training:        
        conf.log_path = os.path.join(conf.work_path, 'log', conf.data_mode, conf.exp)
        conf.save_path = os.path.join(conf.work_path, 'save')

        # conf.weight_decay = 5e-4
        conf.lr = 1e-3
        conf.momentum = 0.9

        # conf.milestones = [4, 7, 10]
        # conf.milestones = [12,15,18]
        # conf.milestones = [6, 11, 16]
        conf.milestones = [8, 16, 24] # for 30 epoch
        conf.pin_memory = True
        # conf.num_workers = 4 # when batchsize is 200
        conf.num_workers = 3
        conf.ce_loss = CrossEntropyLoss()
        conf.l1_loss = L1Loss()
        conf.ls_loss = MSELoss()
#--------------------Inference Config ------------------------
    else:
        conf.facebank_path = conf.data_path/'facebank'
        conf.threshold = 1.5
        conf.face_limit = 10 
        # when inference, at maximum detect 10 faces in one image, my laptop is slow
        conf.min_face_size = 30 
        # the larger this value, the faster deduction, comes with tradeoff in small faces

    return conf