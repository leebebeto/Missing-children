from config import get_config
import argparse
from learner import face_learner
from data.data_pipe import get_val_pair
from torchvision import transforms as trans

import torch
from PIL import Image
import numpy as np
import os
import torchvision.transforms as T
import torch.nn.functional as F
from verification import evaluate_dist
from model import Backbone, Arcface, MobileFaceNet, l2_norm
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from torchvision import transforms
from torchvision.utils import save_image
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def inference_tsne(wrong_image_a, wrong_image_b, fgnetc_label, model):
    our_features = []    # save embedding vectors   size: [314, 512], 314: image 개수, 512: embedding vector size
    labels = []          # save labels              size: [1, 314]    8개의 class 이므로 1~8까지 label이 존재

    cls, label = 0, 0
    with torch.no_grad():
        # img_tensor = []
        # for person_root in root_list:
        #     img_list = os.listdir(person_root)
        #     for img_name in img_list:
        #         img_path = os.path.join(person_root, img_name)
        #         new_cls = img_path.split("/")[-2]
        #
        #         if cls != new_cls:
        #             label += 1
        #             cls = new_cls
        #
        #         img = Image.open(img_path).convert("RGB")
        # import pdb; pdb.set_trace()
        # wrong_image_a = t(wrong_image_a)
        # wrong_image_b = t(wrong_image_b)

        # img = t(img).unsqueeze(0)
        # img = img.to(device)

        # img_tensor.append(img.squeeze(0))
        # labels.append(label)

        # img_tensor = torch.stack(img_tensor)
        wrong_images = torch.cat((torch.tensor(wrong_image_a), torch.tensor(wrong_image_b)), dim=0)
        wrong_images= wrong_images.to(device)
        # img_tensor = img_tensor.to(device)
        img_tensor = []
        idx = 0
        batch_size = 100
        while idx + batch_size <= len(wrong_images):
            print(f'index: {idx}')
            batch = torch.tensor(wrong_images[idx:idx + batch_size])
            feature = model(batch)
            img_tensor.append(feature)
            idx += batch_size

        if idx < len(wrong_images):
            batch = torch.tensor(wrong_images[idx:])
            feature = model(batch)
            img_tensor.append(feature)
        feature = torch.cat(img_tensor[:-1], dim=0)
        feature = torch.cat((feature, img_tensor[-1]), dim=0)
        feature = F.normalize(feature)
    return feature, labels

def evaluate_js(conf, carray, issame, model, nrof_folds=10, tta=True):
    model.eval()
    idx = 0
    embeddings = np.zeros([len(carray), conf.embedding_size])
    with torch.no_grad():
        while idx + conf.batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + conf.batch_size])
            if tta:
                fliped = hflip_batch(batch)
                emb_batch = learner.model(batch.to(conf.device)).cpu() + learner.model(fliped.to(conf.device)).cpu()
                embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch).cpu()
            else:
                embeddings[idx:idx + conf.batch_size] = learner.model(batch.to(conf.device)).cpu()
            idx += conf.batch_size
        if idx < len(carray):
            batch = torch.tensor(carray[idx:])
            if tta:
                fliped = hflip_batch(batch)
                emb_batch = learner.model(batch.to(conf.device)).cpu() + learner.model(fliped.to(conf.device)).cpu()
                embeddings[idx:] = l2_norm(emb_batch).cpu()
            else:
                embeddings[idx:] = learner.model(batch.to(conf.device)).cpu()

    tpr, fpr, accuracy, best_thresholds, dist = evaluate_dist(embeddings, issame, nrof_folds)
    # buf = gen_plot(fpr, tpr)
    # roc_curve = Image.open(buf)
    # roc_curve_tensor = transforms.ToTensor()(roc_curve)
    return accuracy.mean(), best_thresholds.mean(), dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-e", "--epochs", help="training epochs", default=20, type=int)
    parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, mobilefacenet]",default='ir_se', type=str)
    parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument('-lr','--lr',help='learning rate',default=1e-3, type=float)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=64, type=int)
    parser.add_argument("-w", "--num_workers", help="workers number", default=3, type=int)
    parser.add_argument("-d", "--data_mode", help="use which database, [vgg, ms1m, emore, ms1m_vgg_concat, a, vgg_agedb_insta, vgg_adgedb_balanced]",default='vgg', type=str)
    parser.add_argument("-f", "--finetune_model_path", help='finetune using balanced agedb', default=None, type=str)
    parser.add_argument("--finetune_head_path", help='head path', default=None, type=str)
    parser.add_argument("--exp", help='experiment name', default=None, type=str)
    args = parser.parse_args()


    conf = get_config(exp = args.exp, data_mode=args.data_mode)
    conf.finetune_model_path = args.finetune_model_path



    # Load model
    learner = face_learner(conf, inference=True)
    # learner.load_state(conf, '2021-02-10-18-53_accuracy:0.9575714285714285_step:147090_None.pth', None, model_only=True, from_save_folder=True)
    # import pdb; pdb.set_trace()
    learner.load_state(conf, conf.finetune_model_path, None, model_only=False, from_save_folder=False, analyze=True)  # analyze true == does not load optim.

    fgnetc = np.load('/home/nas1_userD/yonggyu/Face_dataset/face_emore/FGNET_new_align_list.npy').astype(np.float32)
    fgnetc_issame = np.load('/home/nas1_userD/yonggyu/Face_dataset/face_emore/FGNET_new_align_label.npy')
    fgnetc_label_temp = open('/home/nas1_userC/yonggyu/Face_Recognition/txt_files/fgnet_children.txt').readlines()
    accuracy2, best_threshold2, dist = evaluate_js(conf, fgnetc, fgnetc_issame, learner.model)

    # wrong_list = fgnetc_issame[((fgnetc_issame == False) & (dist < best_threshold2)) | ((fgnetc_issame == True) & (dist > best_threshold2))]
    # wrong_list = np.where(((fgnetc_issame == False) & (dist < best_threshold2)) | ((fgnetc_issame == True) & (dist > best_threshold2)))
    # plot_num = 100

    wrong_list = np.where((fgnetc_issame == True) & (dist < best_threshold2))[0]
    wrong_index_a, wrong_index_b = wrong_list * 2, wrong_list * 2 + 1
    wrong_image_a, wrong_image_b = fgnetc[wrong_index_a], fgnetc[wrong_index_b]
    fgnetc_label = [fgnetc_label_temp[number].split('_')[0].split('A')[0] for number in wrong_list]
    fgnetc_label = fgnetc_label + fgnetc_label

    import pdb; pdb.set_trace()
    # Transform
    t = T.Compose([
        T.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
     ])

    our_features, labels = inference_tsne(wrong_image_a, wrong_image_b, fgnetc_label, learner.model)
    # our_features = torch.cat(our_features, dim=0)
    # labels = torch.tensor(labels).squeeze(0)


    # tensor to numpy
    our_features = our_features.cpu().numpy()
    labels = np.array(labels)

    # TSNE
    tsne = TSNE(n_components=2, random_state=0)

    X_2d = tsne.fit_transform(our_features)

    today = today = '_'.join(str(datetime.datetime.now()).split(' ')[0].split('-')[1:])

    width_size = 20
    plt.figure(figsize=(width_size, width_size))
    for i in range(len(X_2d)):
        # print(i)
        if i > int(len(fgnetc_label)/2):
            plt.scatter(X_2d[i][0], X_2d[i][1], alpha=0.1)
        else:
            plt.scatter(X_2d[i][0], X_2d[i][1])
        plt.annotate(fgnetc_label[i], xy=(X_2d[i][0], X_2d[i][1]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    # import pdb; pdb.set_trace()
    # # Visualize tsne
    # target_ids = range(1, 9)
    # plt.figure(figsize=(6, 5))
    # colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'purple'
    # for i, c, label in zip(target_ids, colors, np.array([1,2,3,4,5,6,7,8])):
    #     plt.scatter(X_2d[labels == i, 0], X_2d[labels == i, 1], c=c, label=label)
    #
    # import pdb; pdb.set_trace()
    os.makedirs('TSNE', exist_ok=True)
    plt.savefig(f'TSNE/FGNETC_POSITIIVE_CORRECT_all_{width_size}.png')
