from config import get_config
from Learner import face_learner
import argparse
import torch
import numpy as np
import random

# python train.py -net mobilefacenet -b 200 -w 4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-e", "--epochs", help="training epochs", default=20, type=int)
    parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, mobilefacenet]",default='ir_se', type=str)
    parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument('-lr','--lr',help='learning rate',default=1e-3, type=float)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=64, type=int)
    parser.add_argument("-w", "--num_workers", help="workers number", default=3, type=int)
    parser.add_argument("-d", "--data_mode", help="use which database, [vgg, ms1m, emore, ms1m_vgg_concat, vgg_agedb, vgg_agedb_insta, vgg_adgedb_balanced]",default='vgg', type=str)
    parser.add_argument("-f", "--finetune_model_path", help='finetune using balanced agedb', default=None, type=str)
    args = parser.parse_args()

    conf = get_config()

    random_seed = 4885
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    if args.net_mode == 'mobilefacenet':
        conf.use_mobilfacenet = True
    else:
        conf.net_mode = args.net_mode
        conf.net_depth = args.net_depth
    _milestone = ''
    for i in conf.milestones:
        _milestone += ('_'+str(i))
    conf.exp = str(conf.net_depth) + _milestone
        
    conf.lr = args.lr
    conf.batch_size = args.batch_size
    conf.num_workers = args.num_workers
    conf.data_mode = args.data_mode
    conf.finetune_model_path = args.finetune_model_path

    learner = face_learner(conf)
    if conf.finetune_model_path is not None:
        conf.lr = args.lr * 0.01
        learner.load_state(conf, conf.finetune_model_path, model_only=False, from_save_folder=False, analyze=True) # analyze true == does not load optim.
    learner.train(conf, args.epochs)