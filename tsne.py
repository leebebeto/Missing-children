from config import get_config
import argparse
from learner import face_learner
from data.data_pipe import get_val_pair
from torchvision import transforms as trans
import glob
import tqdm

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
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def inference_tsne(wrong_image_a, wrong_image_b, fgnetc_label, model):
    our_features = []    # save embedding vectors   size: [314, 512], 314: image 개수, 512: embedding vector size
    labels = []          # save labels              size: [1, 314]    8개의 class 이므로 1~8까지 label이 존재
    cls, label = 0, 0
    with torch.no_grad():
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

        import pdb; pdb.set_trace()

        feature = F.normalize(feature)
    return feature

def inference_tsne_all(model, transform=None):
    our_features = []    # save embedding vectors   size: [314, 512], 314: image 개수, 512: embedding vector size
    labels = []          # save labels              size: [1, 314]    8개의 class 이므로 1~8까지 label이 존재

    cls, label = 0, 0
    with torch.no_grad():
        # image_list = glob.glob('/home/nas1_userE/jungsoolee/Face_dataset/FGNET_new_align/*')
        image_list = glob.glob('/home/nas1_userE/jungsoolee/Face_dataset/FGNET_new_align/001*') + glob.glob('/home/nas1_userE/jungsoolee/Face_dataset/FGNET_new_align/002*') \
                     + glob.glob('/home/nas1_userE/jungsoolee/Face_dataset/FGNET_new_align/007*') + glob.glob('/home/nas1_userE/jungsoolee/Face_dataset/FGNET_new_align/010*')
        pbar = tqdm.tqdm(total = len(image_list)/2)
        feature_list, label_list, child_list = [], [], []
        idx = 0
        batch_size = 100

        for image_index in range(0, len(image_list), 2):
            pbar.update(1)
            input_tensor = []
            image1 = Image.open(image_list[image_index])
            image2 = Image.open(image_list[image_index+1])
            image1 = transform(image1)
            image2 = transform(image2)
            input_tensor.append(image1)
            input_tensor.append(image2)
            image = torch.stack(input_tensor)
            image = image.cuda()
            feature = model(image)
            feature_list.append(feature[0])
            feature_list.append(feature[1])

            age1 = int(image_list[image_index].split('/')[-1].split('A')[1][:2])
            age1 = 1 if age1 < 18 else 0

            age2 = int(image_list[image_index+1].split('/')[-1].split('A')[1][:2])
            age2 = 1 if age2 < 18 else 0

            child_list.append(age1)
            child_list.append(age2)


            label_list.append(int(image_list[image_index].split('/')[-1].split('A')[0]))
            label_list.append(int(image_list[image_index+1].split('/')[-1].split('A')[0]))


        # while idx + batch_size <= len(image_list):
        #     print(f'index: {idx}')
        #     batch = torch.tensor(wrong_images[idx:idx + batch_size])
        #     import pdb; pdb.set_trace()
        #     feature = model(batch)
        #     img_tensor.append(feature)
        #     idx += batch_size
        #
        # if idx < len(wrong_images):
        #     batch = torch.tensor(wrong_images[idx:])
        #     feature = model(batch)
        #     img_tensor.append(feature)
        # feature = torch.cat(img_tensor[:-1], dim=0)
        # feature = torch.cat((feature, img_tensor[-1]), dim=0)
        # feature = F.normalize(feature)
        #

        feature_list = torch.stack(feature_list)
        import pdb; pdb.set_trace()
        # feature_list = feature_list.view(-1, 512)
        labels = torch.tensor(np.array(label_list))
        feature = F.normalize(feature_list)

    return feature, labels, child_list

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
    parser.add_argument("--tsne_exp", help='experiment name of tsne', default='NEGATIVE', type=str)
    parser.add_argument("--loss", help='loss function', default='Arcface', type=str)
    parser.add_argument("--use_dp", help='use data parallel', default=True)

    args = parser.parse_args()


    conf = get_config(exp = args.exp, data_mode=args.data_mode)
    conf.finetune_model_path = args.finetune_model_path
    conf.finetune_head_path = args.finetune_head_path
    conf.use_dp = args.use_dp


    # Load model
    learner = face_learner(conf, inference=True)
    # learner.load_state(conf, '2021-02-10-18-53_accuracy:0.9575714285714285_step:147090_None.pth', None, model_only=True, from_save_folder=True)
    # import pdb; pdb.set_trace()
    learner.load_state(conf, model_path = conf.finetune_model_path)  # analyze true == does not load optim.
    # learner.load_state(conf, head_path = conf.finetune_head_path)  # analyze true == does not load optim.

    # Transform
    t = T.Compose([
        T.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
     ])
    if 'head' in args.tsne_exp:
        os.makedirs('weight_norm', exist_ok=True)
        norm = torch.norm(learner.head.kernel, dim=0).detach().cpu().numpy()
        index_list = np.arange(norm.shape[0])
        width_size = 20
        plt.figure(figsize=(width_size, width_size))
        plt.plot(index_list[:10575], norm[:10575])
        plt.plot(index_list[10575:], norm[10575:], 'r')
        plt.savefig(f'weight_norm/{args.tsne_exp}.png')
        plt.close()

        import sys
        sys.exit(0)



    if 'ALL' in args.tsne_exp:
        our_features, labels, child_list = inference_tsne_all(learner.model, transform = t)
        our_features = our_features.view(-1, 512)
        fgnetc_label = labels
        wrong_list = our_features

    else:
        fgnetc = np.load('/home/nas1_userD/yonggyu/Face_dataset/face_emore/FGNET_new_align_list.npy').astype(np.float32)
        fgnetc_issame = np.load('/home/nas1_userD/yonggyu/Face_dataset/face_emore/FGNET_new_align_label.npy')
        fgnetc_label_temp = open('/home/nas1_userC/yonggyu/Face_Recognition/txt_files/fgnet_children.txt').readlines()
        accuracy2, best_threshold2, dist = evaluate_js(conf, fgnetc, fgnetc_issame, learner.model)

        negative_wrong =  np.where((fgnetc_issame == False) & (dist < best_threshold2))[0]
        negative_correct =  np.where((fgnetc_issame == False) & (dist > best_threshold2))[0]
        positive_wrong =  np.where((fgnetc_issame == True) & (dist > best_threshold2))[0]
        positive_correct =  np.where((fgnetc_issame == True) & (dist < best_threshold2))[0]
        print(f'negative_wrong : {negative_wrong.shape[0]}')
        print(f'negative_correct: {negative_correct.shape[0]}')
        print(f'positive_wrong : {positive_wrong.shape[0]}')
        print(f'positive_correct: {positive_correct.shape[0]}')
        print(f'total: {negative_wrong.shape[0] + negative_correct.shape[0] + positive_wrong.shape[0] + positive_correct.shape[0]}')
        import sys
        sys.exit(0)

        if args.tsne_exp == 'NEGATIVE_WRONG':
            wrong_list = np.where((fgnetc_issame == False) & (dist < best_threshold2))[0]
            dist = dist[wrong_list]
        elif args.tsne_exp == 'POSITIVE_WRONG':
            wrong_list = np.where((fgnetc_issame == True) & (dist > best_threshold2))[0]
            dist = dist[wrong_list]
        elif args.tsne_exp == 'NEGATIVE_CORRECT':
            wrong_list = np.where((fgnetc_issame == False) & (dist > best_threshold2))[0]
            dist = dist[wrong_list]
        elif args.tsne_exp == 'POSITIVE_CORRECT':
            wrong_list = np.where((fgnetc_issame == True) & (dist < best_threshold2))[0]
            dist = dist[wrong_list]
        elif args.tsne_exp == 'POSITIVE_TOTAL':
            wrong_list = np.where(fgnetc_issame == True)[0]
            dist = dist[wrong_list]
        elif args.tsne_exp == 'NEGATIVE_TOTAL':
            wrong_list = np.where(fgnetc_issame == False)[0]
            dist = dist[wrong_list]
        elif args.tsne_exp == 'TOTAL':
            wrong_list = np.arange(len(fgnetc_issame))

        wrong_index_a, wrong_index_b = wrong_list * 2, wrong_list * 2 + 1
        wrong_image_a, wrong_image_b = fgnetc[wrong_index_a], fgnetc[wrong_index_b]

        # identity label
        fgnetc_label_a = [fgnetc_label_temp[number].split(' ')[0].split('A')[0] for number in wrong_list]
        fgnetc_label_b = [fgnetc_label_temp[number].split(' ')[1].split('A')[0] for number in wrong_list]

        # image label
        image_label_a = [fgnetc_label_temp[number].split(' ')[0].split('A')[1][:2] for number in wrong_list]
        image_label_b = [fgnetc_label_temp[number].split(' ')[1].split('A')[1][:2] for number in wrong_list]

        # age difference
        age = [abs(int(image_label_a[index])- int(image_label_b[index])) for index in range(len(image_label_a))]
        age = np.array(age)

        fgnetc_label = fgnetc_label_a + fgnetc_label_b
        image_label = image_label_a + image_label_b
        # tsne
        # index_list = np.arange(len(wrong_list))
        # wrong_list = np.array(index_list.tolist() + index_list.tolist())
        our_features = inference_tsne(wrong_image_a, wrong_image_b, fgnetc_label, learner.model)
    # age analysis
    # output_dir = 'AGE_ANALYSIS'
    # os.makedirs(output_dir, exist_ok=True)
    # plt.scatter(age, dist, s=1.3)
    # plt.ylim((0.0, 2.0))
    # plt.savefig(f'{output_dir}/AGE_DIST_{args.exp}_{args.tsne_exp}.png')
    #
    # import sys
    # sys.exit(0)

    # image analysis
    # output_dir = f'IMAGE_ANALYSIS/{args.exp}_{args.tsne_exp}'
    # os.makedirs(output_dir, exist_ok=True)
    # pbar=tqdm.tqdm(total=len(wrong_list))
    # for i, index in enumerate(wrong_list):
    #     pbar.update(1)
    #     root_path = '/home/nas1_userE/jungsoolee/Face_dataset/FGNET_new_align'
    #     name_a, name_b = fgnetc_label_temp[index].split(' ')[0][:-4], fgnetc_label_temp[index].split(' ')[1][:-4]
    #     image_a, image_b = os.path.join(root_path, fgnetc_label_temp[index].split(' ')[0]), os.path.join(root_path, fgnetc_label_temp[index].split(' ')[1])
    #     image_a, image_b = Image.open(image_a).convert('RGB'), Image.open(image_b).convert('RGB')
    #     image_a, image_b = np.array(image_a), np.array(image_b)
    #     image = np.concatenate((image_a, image_b), axis=1)
    #     image = Image.fromarray(image)
    #     image.save(f'{output_dir}/{best_threshold2:.3f}_{dist[index]:.3f}_{name_a}_{name_b}.png')
    #
    # import sys
    # sys.exit(0)


    # our_features = torch.cat(our_features, dim=0)
    # labels = torch.tensor(labels).squeeze(0)

    # tensor to numpy
    our_features = our_features.cpu().numpy()

    # TSNE
    if 'tsne' in args.exp:
        tsne = TSNE(n_components=2, random_state=0)
        X_2d = tsne.fit_transform(our_features)

    elif 'pca' in args.exp:
        pca = PCA(n_components=2, random_state=0)
        pca.fit(our_features)
        X_2d = pca.transform(our_features)

    today = today = '_'.join(str(datetime.datetime.now()).split(' ')[0].split('-')[1:])

    width_size = 20
    plt.figure(figsize=(width_size, width_size))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'purple']
    # colors = list(mcolors.CSS4_COLORS.values())
    if 'ALL' in args.tsne_exp:
        for i in range(len(X_2d)):
            color_index = int(fgnetc_label[i]) % len(colors)
            if child_list[i] == 0: # adults
                plt.scatter(X_2d[i][0], X_2d[i][1], c= colors[color_index], alpha=0.1)
            else:  # child
                plt.scatter(X_2d[i][0], X_2d[i][1], c= colors[color_index])
            plt.annotate(fgnetc_label[i].item(), xy=(X_2d[i][0], X_2d[i][1]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')


    else:
        for i in range(len(X_2d)):
            # print(i)
            # color_index = int(fgnetc_label[i]) % len(colors)
            color_index = int(wrong_list[i]) % len(colors)
            if i > int(len(fgnetc_label)/2): # adults
                plt.scatter(X_2d[i][0], X_2d[i][1], c= colors[color_index], alpha=0.1)
            else: # child
                plt.scatter(X_2d[i][0], X_2d[i][1], c= colors[color_index])
            # plt.annotate(f'{wrong_list[i]}_{image_label[i]}_{fgnetc_label[i]}', xy=(X_2d[i][0], X_2d[i][1]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
            plt.annotate(f'{fgnetc_label[i]}', xy=(X_2d[i][0], X_2d[i][1]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

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
    plt.savefig(f'TSNE/FGNETC_{conf.exp}_{args.tsne_exp}_{int(len(wrong_list)/2)}_{width_size}.png')
