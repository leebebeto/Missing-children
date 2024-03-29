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
    parser.add_argument("--epochs", help="training epochs", default=50, type=int)
    parser.add_argument("--lr",help='learning rate',default=1e-1, type=float)
    parser.add_argument("--momentum",help='momentum',default=0.9, type=float)
    parser.add_argument("--batch_size", help="batch_size", default=64, type=int)
    parser.add_argument("--num_workers", help="workers number", default=16, type=int)
    parser.add_argument("--data_mode", help="use which database, [casia, vgg, ms1m, emore, ms1m_vgg_concat, a, vgg_agedb_insta, prettiermonster47, prettiermonster92]",default='casia', type=str)
    parser.add_argument("--finetune_model_path", help='finetune using balanced agedb', default=None, type=str)
    parser.add_argument("--finetune_head_path", help='head path', default=None, type=str)
    parser.add_argument("--use_dp", help='use data parallel', action= 'store_true')
    parser.add_argument("--use_sync", help='use sync batchnorm', default=True)
    parser.add_argument("--exp", help='experiment name', default='debugging', type=str)
    parser.add_argument("--use_memory", help='whether to use memory', action='store_true')
    parser.add_argument("--use_sorted", help='whether to sort child index', default='random', type=str)
    parser.add_argument("--lambda_child", help='lambda for child loss', default=1.0, type=float)
    parser.add_argument("--child_filter", help='whether to filter child, threshold:0', default=0, type=int)
    parser.add_argument("--use_adult_memory", help='whether to use adult memory', action='store_true')
    parser.add_argument("--short_milestone", help='whether to use ', action='store_true')
    parser.add_argument("--seed", help='seed', default=4885, type=int)
    parser.add_argument("--child_margin", help='child margin', default=0.5, type=float)
    parser.add_argument("--weighted_ce", help='re-weight cross entropy', default=False, action='store_true')
    parser.add_argument("--log_degree", help='whether to log degree', action='store_true')
    parser.add_argument("--use_prototype", help='whether to use prototype', action='store_true')
    parser.add_argument("--prototype_mode", help='whether to use prototype', default='all', type=str)
    parser.add_argument("--prototype_loss", help='loss type', default='L1', type=str)
    parser.add_argument("--oecnn_lambda", help='oecnn lambda', default=0.001, type=float)

    # data path -> added temporarily
    parser.add_argument("--vgg_folder", help='vgg folder directory', default='/home/nas1_userD/yonggyu/Face_dataset/vgg')
    parser.add_argument("--agedb_folder", help='agedb folder directory', default='/home/nas3_userL/jungsoolee/Face_dataset/AgeDB_new_align')
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
    parser.add_argument("--wandb", help='whether to use wandb', action='store_true')
    parser.add_argument("--loss_freq", help="frequency for loss boarding", default=100, type=int)
    parser.add_argument("--evaluate_freq", help="max_m for LDAM", default=2000, type=int)
    parser.add_argument("--save_freq", help="scale factor for LDAM", default=2000, type=int)

    #past
    parser.add_argument("--angle", help='whether to analyze angles', default=False)
    parser.add_argument("--casia_vgg_mode", help='how to select vgg', default='random')
    parser.add_argument("--minus_m", help='margin for negative pair of child', default=0.5, type=float)
    parser.add_argument("--lambda_mode", help='lambda option for memory bank', default='zero', type=str)
    parser.add_argument("--lambda_mixup", help='lambda for mixup', default=1.0, type=float)
    parser.add_argument("--vanilla_mixup", help='ablation for mixup', action='store_true')
    parser.add_argument("--feature_level", help='whether to use adult memory as feature level', action='store_true')
    parser.add_argument("--positive_zero", help='whether to use adult memory as zero', action='store_true')
    parser.add_argument("--positive_one", help='whether to use adult memory as one', action='store_true')
    parser.add_argument("--use_arccos", help='whether to use arccos', action='store_true')
    parser.add_argument("--original_positive", help='whether to use original positive loss', action='store_true')
    parser.add_argument("--positive_lambda", help='positive lambda for positive loss', default=1.0, type=float)
    parser.add_argument("--negative_lambda", help='negative lambda for negative loss', default=1.0, type=float)
    parser.add_argument("--positive_ce", help='whether to use positive ce loss', action='store_true')
    parser.add_argument("--memory_include", help='whether to include pretty in oversampling', action='store_true')
    parser.add_argument("--use_strange", help='whether to use positive strange loss', action='store_true')
    parser.add_argument("--use_pretrain", help='whether to use pretrained model', action='store_true')
    parser.add_argument("--use_adult_memory_pretrain", help='whether to use pretrained model 2', action='store_true')
    parser.add_argument("--main_lambda", help='main lambda for main loss', default=1.0, type=float)
    parser.add_argument("--alpha", help='update ratio for memory bank', default=0.5, type=float)
    parser.add_argument("--new_id", help='number of new identities', default=100, type=int)



    args = parser.parse_args()
    args.home = os.path.expanduser('~')

    # fix random seeds
    random_seed = args.seed
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
    if args.use_prototype: # our memory bank method
        print('using prototype ...')
        learner.train_prototype(args, args.epochs)
    elif args.loss == 'DAL':
        learner.train_dal(args, args.epochs)
    elif args.loss == 'OECNN':
        learner.train_oecnn(args, args.epochs)
    # elif args.use_adult_memory: # our memory bank method
    #     print('using adult memory bank...')
    #     learner.train_adult_memory(args, args.epochs)
    # elif args.use_adult_memory_pretrain: # our memory bank method
    #     print('using adult memory bank pretrain...')
    #     learner.train_adult_memory_pretrain(args, args.epochs)
    # elif args.vanilla_mixup:
    #     print('vanilla mixup...')
    #     learner.train_mixup(args, args.epochs)
    else:
        learner.train(args, args.epochs) # other normal training
