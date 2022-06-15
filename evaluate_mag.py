# from config import get_config
import argparse
from learner import face_learner
from data.data_pipe import get_val_pair

from sklearn.model_selection import KFold
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate

import torch
from model import *

from PIL import Image
import torchvision.transforms as T
from utils_txt import cos_dist, fixed_img_list
import tqdm, time, os, glob, pickle, random

def fixed_img_list(text_pair):

    f = open(text_pair, 'r')
    lines = []

    while True:
        line = f.readline()
        if not line:
            break
        lines.append(line)
    f.close()

    random.shuffle(lines)
    return lines

def control_text_list(text_path, kist=False):
    if kist:
        img_list, pair_list = text_path[0], text_path[1]
        with open(img_list, 'r') as f:
            img_list = f.readlines()
        with open(pair_list, 'r') as f:
            pair_list = f.readlines()
        pairs = img_list
        labels = pair_list
    else:
        lines = sorted(fixed_img_list(text_path))
        pairs = [' '.join(line.split(' ')[1:]) for line in lines]
        labels = [int(line.split(' ')[0]) for line in lines]
    return pairs, labels


def calculate_roc(thresholds, embeddings0, embeddings1,
                  actual_issame, nrof_folds=10, subtract_mean=False):
    assert (embeddings0.shape[0] == embeddings1.shape[0])
    assert (embeddings0.shape[1] == embeddings1.shape[1])

    nrof_pairs = min(len(actual_issame), embeddings0.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings0[train_set], embeddings1[train_set]]), axis=0)
        else:
            mean = 0.

        dist = distance_(embeddings0 - mean, embeddings1 - mean)
        
        import pdb; pdb.set_trace()

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy

def distance_(embeddings0, embeddings1):
    # Distance based on cosine similarity
    dot = np.sum(np.multiply(embeddings0, embeddings1), axis=1)
    norm = np.linalg.norm(embeddings0, axis=1) * np.linalg.norm(embeddings1, axis=1)
    # shaving
    similarity = np.clip(dot / norm, -1., 1.)
    dist = np.arccos(similarity) / math.pi
    return dist

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def verification_mag_kist(net, label_list, pair_list, transform, data_dir=None):
    similarities = []
    labels = []
    assert 2 * len(label_list) == len(pair_list)

    trans_list = []
    trans_list += [T.ToTensor()]
    trans_list += [T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    t = T.Compose(trans_list)

    embeddings0, embeddings1, targets = [], [], []
    if len(label_list) == 0:
        return 0, 0, 0

    net.eval()
    with torch.no_grad():  # Test 때 GPU를 사용할 경우 메모리 절약을 위해 torch.no_grad() 내에서 하는 것이 좋다.
        for idx in tqdm.tqdm(range(len(label_list))):
            idx_1, idx_2, label = int(label_list[idx].split(' ')[0]), int(label_list[idx].split(' ')[1]), int(label_list[idx].split(' ')[-1].split('\n')[0])
            path_1, path_2 = pair_list[idx_1], pair_list[idx_2]
            path_1 = '/home/nas4_user/jungsoolee/FaceRecog_TestSet/img/' + '/'.join(path_1.split('/')[-2:]).split('\n')[0]
            path_2 = '/home/nas4_user/jungsoolee/FaceRecog_TestSet/img/' + '/'.join(path_2.split('/')[-2:]).split('\n')[0]
            img_1 = t(Image.open(path_1)).unsqueeze(dim=0).cuda()
            img_2 = t(Image.open(path_2)).unsqueeze(dim=0).cuda()
            imgs = torch.cat((img_1, img_2), dim=0)

            features = net(imgs)
            embeddings0.append(features[0])
            embeddings1.append(features[1])
            targets.append(label)

    # embeddings0 = np.vstack(embeddings0)
    # embeddings1 = np.vstack(embeddings1)
    # targets = np.vstack(targets).reshape(-1, )
    embeddings0 = torch.stack(embeddings0).detach().cpu().numpy()
    embeddings1 = torch.stack(embeddings1).detach().cpu().numpy()
    targets = np.array(targets)
    # import pdb; pdb.set_trace()

    thresholds = np.arange(0, 4, 0.01)
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings0, embeddings1, targets, nrof_folds=args.test_folds, subtract_mean=True)
    print('EVAL with MAG - Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    return np.mean(accuracy), np.std(accuracy)

def verification_mag(net, label_list, pair_list, transform, data_dir=None):
    similarities = []
    labels = []
    assert len(label_list) == len(pair_list)

    trans_list = []
    trans_list += [T.ToTensor()]
    trans_list += [T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    t = T.Compose(trans_list)

    embeddings0, embeddings1, targets = [], [], []
    if len(label_list) == 0:
        return 0, 0, 0

    net.eval()
    with torch.no_grad():  # Test 때 GPU를 사용할 경우 메모리 절약을 위해 torch.no_grad() 내에서 하는 것이 좋다.
        for idx, pair in enumerate(tqdm.tqdm(pair_list)):
            if data_dir is None:
                if 'png' in pair:
                    path_1, path_2 = pair.split('.png ')
                    path_1 = path_1 + '.png'
                    path_2 = path_2[:-1]
                elif 'jpg' in pair:
                    path_1, path_2 = pair.split('.jpg ')
                    path_1 = path_1 + '.jpg'
                    path_2 = path_2[:-1]
                elif 'JPG' in pair:
                    path_1, path_2 = pair.split('.JPG ')
                    path_1 = path_1 + '.JPG'
                    path_2 = path_2[:-1]

            path_1 = '/home/nas4_user/jungsoolee/Face_dataset/' + '/'.join(path_1.split('/')[-2:])
            path_2 = '/home/nas4_user/jungsoolee/Face_dataset/' + '/'.join(path_2.split('/')[-2:])

            img_1 = t(Image.open(path_1)).unsqueeze(dim=0).cuda()
            img_2 = t(Image.open(path_2)).unsqueeze(dim=0).cuda()
            imgs = torch.cat((img_1, img_2), dim=0)

            features = net(imgs)
            embeddings0.append(features[0])
            embeddings1.append(features[1])
            # similarities.append(cos_dist(features[0], features[1]).cpu())
            label = int(label_list[idx])
            # labels.append(label)
            targets.append(label)

    # embeddings0 = np.vstack(embeddings0)
    # embeddings1 = np.vstack(embeddings1)
    # targets = np.vstack(targets).reshape(-1, )
    embeddings0 = torch.stack(embeddings0).detach().cpu().numpy()
    embeddings1 = torch.stack(embeddings1).detach().cpu().numpy()
    targets = np.array(targets)

    # np.dot(embeddings0, embeddings1.T)
    thresholds = np.arange(0, 4, 0.01)
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings0, embeddings1, targets, nrof_folds=args.test_folds, subtract_mean=True)
    print('EVAL with MAG - Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))

def verification_ours(net, label_list, pair_list, transform, data_dir=None):
    similarities = []
    labels = []
    assert len(label_list) == len(pair_list)

    trans_list = []
    trans_list += [T.ToTensor()]
    trans_list += [T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    t = T.Compose(trans_list)

    if len(label_list) == 0:
        return 0, 0, 0
    net.eval()
    with torch.no_grad():  # Test 때 GPU를 사용할 경우 메모리 절약을 위해 torch.no_grad() 내에서 하는 것이 좋다.
        for idx, pair in enumerate(tqdm.tqdm(pair_list)):
            if data_dir is None:
                if 'png' in pair:
                    path_1, path_2 = pair.split('.png ')
                    path_1 = path_1 + '.png'
                    path_2 = path_2[:-1]
                elif 'jpg' in pair:
                    path_1, path_2 = pair.split('.jpg ')
                    path_1 = path_1 + '.jpg'
                    path_2 = path_2[:-1]
                elif 'JPG' in pair:
                    path_1, path_2 = pair.split('.JPG ')
                    path_1 = path_1 + '.JPG'
                    path_2 = path_2[:-1]

            path_1 = '/home/nas4_user/jungsoolee/Face_dataset/' + '/'.join(path_1.split('/')[-2:])
            path_2 = '/home/nas4_user/jungsoolee/Face_dataset/' + '/'.join(path_2.split('/')[-2:])

            img_1 = t(Image.open(path_1)).unsqueeze(dim=0).cuda()
            img_2 = t(Image.open(path_2)).unsqueeze(dim=0).cuda()
            imgs = torch.cat((img_1, img_2), dim=0)

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
    print(f'EVAL with OURS - best acc: {best_accr}')
    return best_accr, best_th, idx


parser = argparse.ArgumentParser(description='for face verification')
parser.add_argument("--net_mode", help="which network, [ir, ir_se, mobilefacenet]",default='ir_se', type=str)
parser.add_argument("--net_depth", help="how many layers [50,100,152]", default=50, type=int)
parser.add_argument("--drop_ratio", help="ratio of drop out", default=0.6, type=float)
parser.add_argument("--model_path", help="evaluate model path", default='ours/fgnetc_best_model_2021-06-01-12-15_accuracy:0.860_step:226000_casia_CASIA_POSITIVE_ZERO_05_MILE_3.pth', type=str)
parser.add_argument("--device", help="device", default='cuda', type=str)
parser.add_argument("--embedding_size", help='embedding_size', default=512, type=int)
parser.add_argument("--wandb", help="whether to use wandb", action='store_true')
parser.add_argument("--tensorboard", help="whether to use tensorboard", action='store_true')
parser.add_argument("--epochs", help="num epochs", default=50, type=int)
parser.add_argument("--batch_size", help="batch_size", default=64, type=int)
parser.add_argument("--loss", help="loss", default='Arcface', type=str)
parser.add_argument("--test_dir", help="test dir", default='fgnet20', type=str)
parser.add_argument("--test_folds", help="test dir", default=10, type=int)
parser.add_argument("--exp", help="exp name", default='kist_deployment', type=str)
args = parser.parse_args()

os.environ["OMP_NUM_THREADS"] = "1"

# conf = get_config(training=False)
_wandb = args.wandb
args.wandb = False
learner = face_learner(args, inference=True)
args.wandb = _wandb
# save_path = '/home/nas1_temp/jooyeolyun/mia_params/'

# model_path = os.path.join(args.model_path)
# learner.load_state(args, model_path = model_path)
before_time = time.time()

# 데이터 관련 세팅
gray_scale = False

# Hyperparameter
feature_dim = 512

# GPU가 있을 경우 연산을 GPU에서 하고 없을 경우 CPU에서 진행
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
net_depth, drop_ratio, net_mode = 50, 0.6, 'ir_se'

trans_list = []
trans_list += [T.ToTensor()]
trans_list += [T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
t = T.Compose(trans_list)

model = Backbone(net_depth, drop_ratio, net_mode).to(dev)
baseline_models = glob.glob('/home/nas1_temp/jooyeolyun/mia_params/baseline/*/*/*') + glob.glob('/home/nas1_temp/jooyeolyun/mia_params/ours/*/*')
# baseline_models = glob.glob('/home/nas1_temp/jooyeolyun/mia_params/ours/*/*')
baseline_models = [model for model in baseline_models if 'head' not in model.split('/')[-1]]
# import pdb; pdb.set_trace()

# # BASELINES
# for ckpt in baseline_models:
#     if 'ours' in ckpt:
#         if ckpt.split('/')[-1].split('_')[-1].split('.')[0] == '5678' or ckpt.split('/')[-1].split('_')[-1].split('.')[0] == '1234':
#             seed = int(ckpt.split('/')[-1].split('_')[-1].split('.')[0])
#         else:
#             seed = 4885
#         test_set, method = ckpt.split('/')[6], ckpt.split('/')[5]
#     else:
#         test_set, method, seed = ckpt.split('/')[7], ckpt.split('/')[6].split('_')[1], int(ckpt.split('/')[6].split('_')[2])

#     if seed != 4885 or test_set == 'lag': # TEMPORARY MEASURE
#         continue

#     model.load_state_dict(torch.load(ckpt))
#     print(f'{test_set}-{method}-{seed} model loaded...')

#     if args.wandb:
#         import wandb
#         wandb.init(
#             project='uncategorized', 
#             entity="davian-bmvc-face", 
#             name=f'kist_deploy_{test_set}_{method}_{seed}')


#     model.eval()
#     pairs, labels = control_text_list(text_path=[f'/home/nas4_user/jungsoolee/FaceRecog_TestSet/img.list',
#                                         f'/home/nas4_user/jungsoolee/FaceRecog_TestSet/pair.list'],
#                                         kist=True)
#     acc, std = verification_mag_kist(model, labels, pairs, transform=t)

#     if args.wandb:
#         wandb.log({
#             f'kist_test_set ACC': acc,
#             f'kist_test_set STD': std,
#         }, step=0)
#         wandb.finish()


# SINGLE MODEL
name = 'LR_Interclass'
seed = 4885
ckpt = '/home/nas1_temp/jooyeolyun/repos/Missing-children/work_space/models_serious/\
interclass_MSE_proto1_LR/agedb20/fgnetc_best_model_2022-05-18-13-03_accuracy:0.721_step:337304_casia_interclass_MSE_proto1_LR.pth'


if args.wandb:
    import wandb
    name = f'kist_deploy_{name}_{seed}'

    wandb.init(
        project='uncategorized', 
        entity="davian-bmvc-face", 
        name=name)

# model.load_state_dict(torch.load(ckpt))
model.eval()
pairs, labels = control_text_list(text_path=[f'/home/nas4_user/jungsoolee/FaceRecog_TestSet/img.list',
                                    f'/home/nas4_user/jungsoolee/FaceRecog_TestSet/pair.list'],
                                    kist=True)
acc, std = verification_mag_kist(model, labels, pairs, transform=t)

if args.wandb:
    wandb.log({
        f'kist_test_set ACC': acc,
        f'kist_test_set STD': std,
    }, step=0)
    wandb.finish()

# original test sets
# pairs, labels = control_text_list(f'/home/nas4_user/jungsoolee/Missing-children/txt_files_sh/{args.test_dir}_child.txt')
# verification_mag(model, labels, pairs, transform=t)
# verification_ours(model, labels, pairs, transform=t)
