import torch
from config import get_config
from Learner import face_learner
import argparse

# python analyze_trained_model.py -net ir_se --batch_size 64 --data_mode vgg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-e", "--epochs", help="training epochs", default=1, type=int)
    parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, mobilefacenet]",default='ir_se', type=str)
    parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument('-lr','--lr',help='dummy learning rate to pass config.py',default=0, type=float)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=64, type=int)
    parser.add_argument("-w", "--num_workers", help="workers number", default=3, type=int)
    parser.add_argument("-d", "--data_mode", help="use which database, [vgg, ms1m, emore, concat]",default='vgg', type=str)
    args = parser.parse_args()

    conf = get_config()

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
    learner = face_learner(conf)
    print(conf)
    learner.load_state(conf, '2021-02-13-02-58_accuracy:0.957857142857143_step:465785_None.pth', model_only=False, from_save_folder=False, analyze=True)
    learner.analyze_angle(conf)
