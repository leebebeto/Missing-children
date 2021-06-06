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
args = parser.parse_args()

# conf = get_config(training=False)
learner = face_learner(args, inference=True)
# save_path = '/home/nas1_temp/jooyeolyun/mia_params/baseline'
save_path = '/home/nas1_temp/jooyeolyun/mia_params/ours'

# learner.load_state(conf, 'ir_se50.pth', model_only=True, from_save_folder=True)
model_path = os.path.join(save_path, args.model_path)
learner.load_state(args, model_path = model_path)
before_time = time.time()

# print('evaluating fgnetc')
# dataset_root = os.path.join('/home/nas1_userE/jungsoolee/Face_dataset/face_emore2')
# fgnetc = np.load(os.path.join(dataset_root, "FGNET_new_align_list.npy")).astype(np.float32)
# fgnetc_issame = np.load(os.path.join(dataset_root, "FGNET_new_align_label.npy"))
# fgnetc_accuracy, fgnetc_thres, roc_curve_tensor2, fgnetc_dist = learner.evaluate(args, fgnetc, fgnetc_issame, nrof_folds=10, tta=True)
# print('fgnetc - accuracy:{}, threshold:{}'.format(fgnetc_accuracy, fgnetc_thres))

dataset_root = os.path.join('/home/nas1_userE/jungsoolee/Face_dataset/faces_emore')
# print('evaluating lfw')
# lfw, lfw_issame = get_val_pair(dataset_root, 'lfw')
# accuracy, best_threshold, roc_curve_tensor, dist = learner.evaluate(args, lfw, lfw_issame, nrof_folds=10, tta=True)
# print('lfw - accuracy:{}, threshold:{}'.format(accuracy, best_threshold))

# print('evaluating agedb_30')
# agedb_30, agedb_30_issame = get_val_pair(dataset_root, 'agedb_30')
# accuracy, best_threshold, roc_curve_tensor, _ = learner.evaluate(args, agedb_30, agedb_30_issame, nrof_folds=10, tta=True)
# print('agedb_30 - accuracy:{}, threshold:{}'.format(accuracy, best_threshold))

print('evaluating calfw')
calfw, calfw_issame = get_val_pair(dataset_root, 'calfw')
accuracy, best_threshold, roc_curve_tensor, dist = learner.evaluate(args, calfw, calfw_issame, nrof_folds=10, tta=True)
print('calfw - accuracy:{}, threshold:{}'.format(accuracy, best_threshold))

# cfp_ff, cfp_ff_issame = get_val_pair(conf.emore_folder, 'cfp_ff')
# accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, cfp_ff, cfp_ff_issame, nrof_folds=10, tta=True)
# print('cfp_ff - accuracy:{}, threshold:{}'.format(accuracy, best_threshold))
# # trans.ToPILImage()(roc_curve_tensor)
#
# cfp_fp, cfp_fp_issame = get_val_pair(conf.emore_folder, 'cfp_fp')
# accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, cfp_fp, cfp_fp_issame, nrof_folds=10, tta=True)
# print('cfp_fp - accuracy:{}, threshold:{}'.format(accuracy, best_threshold))
# # trans.ToPILImage()(roc_curve_tensor)
#
# cplfw, cplfw_issame = get_val_pair(conf.emore_folder, 'cplfw')
# accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, cplfw, cplfw_issame, nrof_folds=10, tta=True)
# print('cplfw - accuracy:{}, threshold:{}'.format(accuracy, best_threshold))
# # trans.ToPILImage()(roc_curve_tensor)
#
# vgg2_fp, vgg2_fp_issame = get_val_pair(conf.emore_folder, 'vgg2_fp')
# accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, vgg2_fp, vgg2_fp_issame, nrof_folds=10, tta=True)
# print('vgg2_fp - accuracy:{}, threshold:{}'.format(accuracy, best_threshold))
# # trans.ToPILImage()(roc_curve_tensor)
#
