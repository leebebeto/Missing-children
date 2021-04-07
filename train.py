from config import get_config
from Learner import face_learner
import argparse
import torch
import numpy as np
import random

# python train.py -net mobilefacenet -b 200 -w 4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')

    # training
    parser.add_argument("--epochs", help="training epochs", default=20, type=int)
    parser.add_argument("--lr",help='learning rate',default=1e-3, type=float)
    parser.add_argument("--momentum",help='momentum',default=0.9, type=float)
    parser.add_argument("--batch_size", help="batch_size", default=64, type=int)
    parser.add_argument("--num_workers", help="workers number", default=3, type=int)
    parser.add_argument("--data_mode", help="use which database, [vgg, ms1m, emore, ms1m_vgg_concat, a, vgg_agedb_insta, vgg_adgedb_balanced]",default='vgg', type=str)
    parser.add_argument("--finetune_model_path", help='finetune using balanced agedb', default=None, type=str)
    parser.add_argument("--finetune_head_path", help='head path', default=None, type=str)
    parser.add_argument("--vgg_folder", help='directory for vgg_folder', default='/home/nas1_userE/jungsoolee/Face_dataset/Vgg_age_label', type=str)
    parser.add_argument("--ms1m_folder", help='directory for vgg_folder', default='/home/nas1_userE/jungsoolee/Face_dataset/ms1m-refined-112', type=str)
    parser.add_argument("--emore_folder", help='directory for vgg_folder', default='/home/nas1_userE/jungsoolee/Face_dataset/faces_emore', type=str)
    parser.add_argument("--agedb_folder", help='directory for vgg_folder', default='/home/nas1_userE/jungsoolee/Face_dataset/AgeDB_new_align', type=str)
    parser.add_argument("--agedb_balanced_folder", help='directory for vgg_folder', default='/home/nas1_temp/jooyeolyun/AgeDB_balanced', type=str)
    parser.add_argument("--insta_folder", help='directory for vgg_folder', default='/home/nas1_userD/yonggyu/Instagram_face_preprocessed', type=str)
    parser.add_argument("--exp", help='experiment name', default=None, type=str)

    # model
    parser.add_argument("--net_mode", help="which network, [ir, ir_se, mobilefacenet]",default='ir_se', type=str)
    parser.add_argument("--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument("--embedding_size", help='embedding_size', default=512, type=int)
    parser.add_argument("--drop_ratio", help="ratio of drop out", default=0.6, type=float)
    parser.add_argument("--device", help="cuda or cpu", default='cuda', type=str)

    # logging
    parser.add_argument("--data_path", help='path for loading data', default='data', type=str)
    parser.add_argument("--work_path", help='path for saving models & logs', default='work_space', type=str)
    parser.add_argument("--model_path", help='path for saving models', default='work_space/models', type=str)
    parser.add_argument("--log_path", help='path for saving logs', default='work_space/log', type=str)

    args = parser.parse_args()

    conf = get_config(exp = args.exp, data_mode=args.data_mode)

    random_seed = 4885
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    # if args.net_mode == 'mobilefacenet':
    #     conf.use_mobilfacenet = True
    # else:
    conf.net_mode = args.net_mode
    conf.net_depth = args.net_depth
    _milestone = ''
    for i in conf.milestones:
        _milestone += ('_'+str(i))
    # conf.exp = str(conf.net_depth) + _milestone
    # conf.exp = args.exp
    conf.lr = args.lr
    conf.batch_size = args.batch_size
    conf.num_workers = args.num_workers
    conf.data_mode = args.data_mode
    conf.finetune_model_path = args.finetune_model_path
    conf.finetune_head_path = args.finetune_head_path


    learner = face_learner(conf)
    if conf.finetune_model_path is not None:
        conf.lr = args.lr * 0.001
        learner.load_state(conf, conf.finetune_model_path, conf.finetune_head_path, model_only=False, from_save_folder=False, analyze=True) # analyze true == does not load optim.
        print(f'{conf.finetune_model_path} model loaded...')
    # learner.train_positive(conf, args.epochs)
    learner.train(conf, args.epochs)