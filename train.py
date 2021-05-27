from learner import face_learner
import argparse
import torch
import numpy as np
import random
import os

# python train.py -net mobilefacenet -b 200 -w 4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')

    # training
    parser.add_argument("--epochs", help="training epochs", default=20, type=int)
    parser.add_argument("--lr",help='learning rate',default=1e-1, type=float)
    parser.add_argument("--momentum",help='momentum',default=0.9, type=float)
    parser.add_argument("--batch_size", help="batch_size", default=64, type=int)
    parser.add_argument("--num_workers", help="workers number", default=3, type=int)
    parser.add_argument("--data_mode", help="use which database, [casia, vgg, ms1m, emore, ms1m_vgg_concat, a, vgg_agedb_insta, vgg_adgedb_balanced]",default='casia', type=str)
    parser.add_argument("--finetune_model_path", help='finetune using balanced agedb', default=None, type=str)
    parser.add_argument("--finetune_head_path", help='head path', default=None, type=str)
    parser.add_argument("--use_dp", help='use data parallel', action= 'store_true')
    parser.add_argument("--use_sync", help='use sync batchnorm', default=True)
    parser.add_argument("--exp", help='experiment name', default='debugging', type=str)
    parser.add_argument("--angle", help='whether to analyze angles', default=False)
    parser.add_argument("--casia_vgg_mode", help='how to select vgg', default='random')
    parser.add_argument("--minus_m", help='margin for negative pair of child', default=0.5, type=float)
    parser.add_argument("--alpha", help='update ratio for memory bank', default=0.9, type=float)
    parser.add_argument("--new_id", help='number of new identities', default=100, type=int)
    parser.add_argument("--lambda_mode", help='lambda option for memory bank', default='normal', type=str)
    parser.add_argument("--lambda_mixup", help='lambda for mixup', default=1.0, type=float)
    parser.add_argument("--use_memory", help='whether to use memory', default=False)
    parser.add_argument("--use_sorted", help='whether to sort child index', default='random', type=str)

    # data path -> added temporarily
    parser.add_argument("--vgg_folder", help='vgg folder directory', default='/home/nas1_userD/yonggyu/Face_dataset/vgg')
    parser.add_argument("--agedb_folder", help='agedb folder directory', default='/home/nas1_userE/jungsoolee/Face_dataset/AgeDB_new_align')
    parser.add_argument("--insta_folder", help='instagram folder directory', default='/home/nas1_userD/yonggyu/Face_dataset/instagram')

    # model
    parser.add_argument("--net_mode", help="which network, [ir, ir_se, mobilefacenet]",default='ir_se', type=str)
    parser.add_argument("--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument("--embedding_size", help='embedding_size', default=512, type=int)
    parser.add_argument("--drop_ratio", help="ratio of drop out", default=0.6, type=float)
    parser.add_argument("--device", help="cuda or cpu", default='cuda', type=str)
    parser.add_argument("--loss", help="Arcface", default='Arcface', type=str)
    parser.add_argument("--max_m", help="max_m for LDAM", default=1.0, type=float)
    parser.add_argument("--scale", help="scale factor for LDAM", default=64, type=int)

    # logging
    parser.add_argument("--data_path", help='path for loading data', default='data', type=str)
    parser.add_argument("--work_path", help='path for saving models & logs', default='work_space', type=str)
    parser.add_argument("--model_path", help='path for saving models', default='work_space/models_serious', type=str)
    parser.add_argument("--log_path", help='path for saving logs', default='work_space/log_serious', type=str)

    args = parser.parse_args()
    args.home = os.path.expanduser('~')

    # fix random seeds
    random_seed = 4885
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    # init learner
    learner = face_learner(args)
    # codes for fine tune
    if args.finetune_model_path is not None:
        args.lr = args.lr * 0.001
        learner.load_state(args, args.finetune_model_path, args.finetune_head_path, model_only=False, from_save_folder=False, analyze=True) # analyze true == does not load optim.
        print(f'{args.finetune_model_path} model loaded...')

    # analyze angles with a pretrained model
    if args.angle == 'True':
        save_path = './work_space/models/'
        # model_path = os.path.join(save_path, 'model_2021-04-10-00-23_accuracy:0.846_step:113442_casia_SYNC_64.pth')
        # head_path = os.path.join(save_path, 'head_2021-04-10-00-23_accuracy:0.846_step:113442_casia_SYNC_64.pth')

        model_path = os.path.join(save_path, 'model_2021-04-10-07-09_accuracy:0.770_step:153300_casia_LDAM_05_64_64.pth')
        head_path = os.path.join(save_path, 'head_2021-04-10-07-09_accuracy:0.770_step:153300_casia_LDAM_05_64_64.pth')

        learner.load_state(args, model_path = model_path, head_path = head_path)
        print(f'pretrained {model_path} loaded finished...')
        learner.analyze_angle(args)
        import sys
        sys.exit(0)

    # actual training
    if args.use_memory == 'True': # our memory bank method
        print('using memory bank...')
        learner.train_memory(args, args.epochs)
    else:
        learner.train(args, args.epochs) # other normal training
