from config import get_config
import argparse
from Learner import face_learner
from data.data_pipe import get_val_pair
from torchvision import transforms as trans

import torch
from PIL import Image
import numpy as np
import os
import torchvision.transforms as T
import torch.nn.functional as F
from model import Backbone, Arcface, MobileFaceNet, l2_norm
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from torchvision import transforms
from torchvision.utils import save_image
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import datetime

import os
import pdb
import shutil



def balance_dataset():
    source_folder = '/home/nas1_userE/Face_dataset/AgeDB_new_align'
    target_folder = '/home/nas1_temp/jooyeolyun/AgeDB_balanced'

    # CURRENT CODE IS ONLY WRITTEN FOR AGEDB DATASET
    # CHECK 'age = split' PART FOR OTHER DIRECTORIES

    for (dirpath, _, filenames) in os.walk(source_folder):
        name = dirpath.split('/')[-1]
        instance = [os.path.join(dirpath, file) for file in filenames]
        age_count = [0,0,0,0,0,0,0,0]
        for file in instance:
            age = int(file.split('_')[-1].strip('.jpg'))
            age_bin = 7
            if age < 26:
                age_bin = 0 if age < 13 else 1 if age <19 else 2
            elif age < 66:
                age_bin = (age+4)//10
            age_count[age_bin] += 1
        upper_bound = max(age_count)
        os.makedirs(target_folder+'/'+name, exist_ok= True)
        for filedir, filename in zip(instance,filenames):
            age = int(filedir.split('_')[-1].strip('.jpg'))
            age_bin = 7
            if age < 26:
                age_bin = 0 if age < 13 else 1 if age <19 else 2
            elif age < 66:
                age_bin = (age+4)//10
            for i in range(upper_bound//age_count[age_bin]):
                shutil.copy2(filedir, target_folder+'/'+name+'/'+str(i)+filename)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def inference_tsne(learner):
    our_features = []    # save embedding vectors   size: [314, 512], 314: image 개수, 512: embedding vector size
    labels = []          # save labels              size: [1, 314]    8개의 class 이므로 1~8까지 label이 존재
    model = learner.model
    growup = learner.growup
    # AgeDB dataset list
    root_list = ['/home/nas1_userE/jungsoolee/Face_dataset/AgeDB_new_align/BarbraStreisand',
                 '/home/nas1_userE/jungsoolee/Face_dataset/AgeDB_new_align/DickieMoore',
                 '/home/nas1_userE/jungsoolee/Face_dataset/AgeDB_new_align/GladysCooper',
                 '/home/nas1_userE/jungsoolee/Face_dataset/AgeDB_new_align/HelenHayes',
                 '/home/nas1_userE/jungsoolee/Face_dataset/AgeDB_new_align/HelenHunt',
                 '/home/nas1_userE/jungsoolee/Face_dataset/AgeDB_new_align/JaneAsher',
                 '/home/nas1_userE/jungsoolee/Face_dataset/AgeDB_new_align/JhonLenon',
                 '/home/nas1_userE/jungsoolee/Face_dataset/AgeDB_new_align/MickeyRooney']

    # Transform
    t = T.Compose([
        T.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    cls, label = 0, 0
    with torch.no_grad():
        img_tensor = []

        for person_root in root_list:
            img_list = os.listdir(person_root)
            for img_name in img_list:
                img_path = os.path.join(person_root, img_name)
                new_cls = img_path.split("/")[-2]
                pdb.set_trace()

                if cls != new_cls:
                    label += 1
                    cls = new_cls

                img = Image.open(img_path).convert("RGB")

                img = t(img).unsqueeze(0).cuda()
                img_tensor.append(img)

            img_tensor = torch.stack(img_tensor)

        # idx = 0
        # batch_size = 100
        # while idx + batch_size <= len(wrong_images):
        #     print(f'index: {idx}')
        #     batch = torch.tensor(wrong_images[idx:idx + batch_size])
        #     feature = model(batch)
        #     img_tensor.append(feature)
        #     idx += batch_size

        # if idx < len(wrong_images):
        #     batch = torch.tensor(wrong_images[idx:])
        #     feature = model(batch)
        #     img_tensor.append(feature)
        feature = torch.cat(img_tensor[:-1], dim=0)
        feature = torch.cat((feature, img_tensor[-1]), dim=0)
        feature = F.normalize(feature)


        feature = model(img_tensor)
        feature = F.normalize(feature)

        our_features.append(feature)
        labels.append(label)

    return feature, labels

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

    conf = get_config(training=False)
    conf.finetune_model_path = args.finetune_model_path

    # Load model
    learner = face_learner(conf, inference=False)
    learner.load_state(conf, conf.finetune_model_path, model_only=False, from_save_folder=False, analyze=True)  # analyze true == does not load optim.
    print('loaded model from {}'.format(conf.finetue_model_path))

    # Transform
    t = T.Compose([
        T.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
     ])

    our_features, labels = inference_tsne(learner)
    # our_features, labels = inference_tsne(wrong_image_a, wrong_image_b, fgnetc_label, learner.model)
    # our_features = torch.cat(our_features, dim=0)
    # labels = torch.tensor(labels).squeeze(0)

    # tensor to numpy
    our_features = our_features.cpu().numpy()
    labels = np.array(labels)

    # TSNE
    tsne = TSNE(n_components=2, random_state=0)

    X_2d = tsne.fit_transform(our_features)

    today = today = '_'.join(str(datetime.datetime.now()).split(' ')[0].split('-')[1:])
    pass
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

                #
                #
                #
                # # # ''' module for negative pair -> create fake prototypes '''
                # # # if e >= 1:
                # #     # if conf.use_sorted == 'random':
                # # if e >= 1:
                # #     if conf.use_sorted == 'min_first':
                # #         child_labels = torch.tensor(self.child_identity_min).cuda()
                # #     if conf.use_sorted == 'max_first':
                # #         child_labels = torch.tensor(self.child_identity_max).cuda()
                # #     elif conf.use_sorted == 'random':
                # #         child_labels_np = np.array(self.child_identity)
                # #         np.random.shuffle(child_labels_np)
                # #         child_labels_np = child_labels_np[:conf.new_id+1]
                # #         child_labels = torch.tensor(child_labels_np).cuda()
                # #
                # #     child_embeddings = self.child_memory[child_labels].cuda()
                # #
                # #     feature_a, feature_b = child_embeddings[:-1], child_embeddings[1:]
                # #
                # #     mixup_features = (feature_a + feature_b) / 2
                # #     mixup_labels = torch.arange(self.class_num - mixup_features.shape[0], self.class_num).cuda()
                # #     mixup_thetas = self.head(mixup_features, mixup_labels)
                # #
                # #     mixup_loss = ce_loss(mixup_thetas, mixup_labels)
                # #
                # # mixup_total_loss = conf.lambda_mixup * mixup_loss
                # # loss = arcface_loss + mixup_total_loss
                # ''' adding fake prototype loss finished '''
#                 elif conf.lambda_mode == 'decay':
#                     if (e == 0) or (e in self.milestones):
#                         child_lambda = 0.0
#                     elif e < self.milestones[0]:
#                         child_lambda = 1.0
#                     elif e > self.milestones[0] and e < self.milestones[1]:
#                         child_lambda = 0.1
#                     elif e > self.milestones[1] and e < self.milestones[2]:
#                         child_lambda = 0.01
#                     elif e > self.milestones[2]:
#                         child_lambda = 0.001