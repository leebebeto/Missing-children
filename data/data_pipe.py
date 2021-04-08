from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision.datasets import ImageFolder, folder
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import pickle
import torch
from tqdm import tqdm
import glob
import pdb
import os
import random
import bcolz

def de_preprocess(tensor):
    return tensor*0.5 + 0.5

def de_preprocess(tensor):
    return tensor*0.5 + 0.5


def get_train_dataset(imgs_folder):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = ImageFolder(imgs_folder, train_transform)
    class_num = ds[-1][1] + 1
    return ds, class_num

def get_train_loader(conf):
    casia_folder =  '/home/nas1_userE/jungsoolee/C3AE/CASIA_112'
    # vgg_folder = '/home/nas1_userE/jungsoolee/Face_dataset/Vgg_age_label'
    # ms1m_folder = '/home/nas1_userE/jungsoolee/Face_dataset/ms1m-refined-112'
    # emore_folder = '/home/nas1_userE/jungsoolee/Face_dataset/faces_emore'
    # agedb_folder = '/home/nas1_userE/jungsoolee/Face_dataset/AgeDB_new_align'
    # # conf.agedb_balanced_folder = '/home/nas1_userE/Face_dataset/AgeDB_balanced'
    # agedb_balanced_folder = '/home/nas1_temp/jooyeolyun/AgeDB_balanced'
    # insta_folder = '/home/nas1_userD/yonggyu/Instagram_face_preprocessed'

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if conf.data_mode == 'casia':
        ds = CASIADataset(casia_folder, train_transforms=train_transform)
        class_num = ds.class_num
        print('casia loader generated')

    if conf.data_mode in ['ms1m', 'ms1m_vgg_concat']:
        ms1m_ds, ms1m_class_num = get_train_dataset(os.path.join(ms1m_folder, 'imgs'))
        print('ms1m loader generated')
    if conf.data_mode in ['vgg', 'ms1m_vgg_concat']:
        vgg_ds = VGGLabeledDataset(vgg_folder, train_transforms=train_transform)
        vgg_class_num = vgg_ds.class_num
        print('vgg loader generated')     
    if conf.data_mode == 'vgg_agedb':
        ds = VGGAgeDBDataset(vgg_folder, agedb_folder, train_transforms=train_transform)
        class_num = ds.class_num
        print('vgg, agedb loader generated')
    if conf.data_mode == 'vgg_agedb_insta':
        ds = VGGAgeDBInstaDataset(vgg_folder, agedb_folder, insta_folder, train_transforms=train_transform)
        class_num = ds.class_num
        print('vgg, with agedb balaned')
        
    if conf.data_mode == 'vgg':
        ds = vgg_ds
        class_num = vgg_class_num
    elif conf.data_mode == 'ms1m':
        ds = ms1m_ds
        class_num = ms1m_class_num
    elif conf.data_mode == 'ms1m_vgg_concat':
        for i,(url,label) in enumerate(vgg_ds.imgs):
            vgg_ds.imgs[i] = (url, label + ms1m_class_num)
        ds = ConcatDataset([ms1m_ds,vgg_ds])
        class_num = vgg_class_num + ms1m_class_num
    elif conf.data_mode == 'emore':
        ds, class_num = get_train_dataset(os.path.join(conf.emore_folder, 'imgs'))

    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=True, pin_memory=True, num_workers=conf.num_workers)
    return loader, class_num, ds
    
def load_bin(path, rootdir, transform, image_size=[112,112]):
    if not rootdir.exists():
        rootdir.mkdir()
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = transform(img)
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data.shape)
    np.save(str(rootdir)+'_list', np.array(issame_list))
    return data, issame_list

def get_val_pair(path, name):
    '''
    Returns image pairs with labels
        carray: numpy-like array with image pairs
        issame: boolean list of image pair validity
    '''
    carray = bcolz.carray(rootdir = os.path.join(path, name), mode='r')
    issame = np.load(os.path.join(path, '{}_list.npy'.format(name)))    

    return carray, issame

def get_val_data(data_path):
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame

def load_mx_rec(rec_path):
    save_path = os.path.join(rec_path, 'imgs')
    os.makedirs(save_path, exist_ok=True)
    # if not save_path.exists():
    #     save_path.mkdir()
    # imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path/'train.idx'), str(rec_path/'train.rec'), 'r')
    imgrec = mx.recordio.MXIndexedRecordIO(os.path.join(rec_path, 'train.idx'), os.path.join(rec_path, 'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        img = img[:, :, ::-1]
        img = Image.fromarray(img)
        label_path = os.path.join(save_path, str(label))
        # label_path = save_path/str(label)
        os.makedirs(label_path, exist_ok=True)
        img.save(os.path.join(label_path, '{}.jpg'.format(idx)), quality=95)
        # if not label_path.exists():
        #     label_path.mkdir()
        # img.save(label_path/'{}.jpg'.format(idx), quality=95)


class CASIADataset(Dataset):
    '''
    CASIA with pseudo age labels dataset
    directory structure
        root/person_i/{age}_filenum.jpg

    Store image directories at init phase

    Returns image, label, age
    '''

    def __init__(self, imgs_folder, train_transforms):
        self.root_dir = imgs_folder
        self.transform = train_transforms
        self.class_num = len(os.listdir(imgs_folder))

        total_list = glob.glob(self.root_dir + '/*/*')
        self.total_imgs = len(total_list)

        # preprocessing class num list for LDAM loss (once calculated -> may substitute with constants)
        # self.child_num = 0
        # for image in total_list:
        #     if int(image.split('/')[-1].split('_')[-1][:-4]) <=18:
        #         self.child_num += 1
        # self.adult_num = self.total_imgs - self.child_num
        # print(f'child: {self.child_num} || adult: {self.adult_num}')
        # self.class_num_list = [self.child_num, self.adult_num]

        self.class_num_list = [7336, 483287]
        self.total_list = total_list
        print(f'{imgs_folder} length: {self.total_imgs}')

    def __len__(self):
        return self.total_imgs

    def __getitem__(self, index):
        img_path = self.total_list[index]
        img_path_list = img_path.split('/')
        file_name = img_path_list[-1]  # {age}_filenum.jpg

        img = Image.open(img_path)
        label = int(img_path.split('/')[-2])
        age = int(file_name.split('/')[-1].split('_')[-1][:-4])

        if self.transform is not None:
            img = self.transform(img)

        age= 0 if age<= 18 else 1

        return img, label, age


class VGGLabeledDataset(Dataset):
    '''
    VGG with pseudo age labels dataset
    directory structure
        root/person_i/{age}_filenum.jpg
    
    Store image directories at init phase

    Returns image, label, age
    '''
    def __init__(self, imgs_folder, train_transforms):
        self.root_dir = imgs_folder
        self.transform = train_transforms
        self.class_num = len(os.listdir(imgs_folder))

        total_list = []
        for (dirpath, _, filenames) in os.walk(imgs_folder):
            total_list += [os.path.join(dirpath, file) for file in filenames]

        self.total_imgs = len(total_list)
        self.total_list = total_list
        
    def __len__(self):
        return self.total_imgs
    
    def __getitem__(self, index):

        img_path = self.total_list[index]
        img_path_list = img_path.split('/')
        dataset_name = img_path_list[-3]
        file_name = img_path_list[-1] # {age}_filenum.jpg
        folder_name = img_path_list[-2]# label

        img = Image.open(img_path)
        label = int(folder_name)
        age = int(file_name.split('_')[0])

        if self.transform is not None:
            img = self.transform(img)

        return img, label, age



class VGGAgeDBOnlyDataset(Dataset):
    '''
    Joint DB of VGG and AgeDB
    VGG with pseudo age labels dataset
    directory structure
        root/person_i/{age}_filenum.jpg
    AGE DB with actual labels
        root/person_name/filenum_{age}.jpg

    Store image directories at init phase
    Only samples AgeDB dataset for finetuning

    Returns image, label, age
    '''

    def __init__(self, vgg_imgs_folder, agedb_imgs_folder, train_transforms):
        self.vgg_imgs_folder_name = vgg_imgs_folder.split('/')[-1]
        self.agedb_imgs_folder_name = agedb_imgs_folder.split('/')[-1]
        self.transform = train_transforms
        self.vgg_class_num = len(os.listdir(vgg_imgs_folder))
        self.agedb_class_list = os.listdir(agedb_imgs_folder)
        self.agedb_class_num = len(self.agedb_class_list)
        self.class_num = self.vgg_class_num + self.agedb_class_num

        total_list = []
        self.vgg_list, self.agedb_list = [], []
        for (dirpath, _, filenames) in os.walk(vgg_imgs_folder):
            self.vgg_list += [os.path.join(dirpath, file) for file in filenames]

        total_list += self.vgg_list
        for (dirpath, _, filenames) in os.walk(agedb_imgs_folder):
            self.agedb_list += [os.path.join(dirpath, file) for file in filenames]

        total_list += self.agedb_list
        self.total_imgs = len(total_list)
        self.total_list = total_list

    def __len__(self):
        return self.total_imgs

    def __getitem__(self, index):
        index = index % len(self.agedb_list)
        img_path = self.agedb_list[index]
        img_path_list = img_path.split('/')
        dataset_name = img_path_list[-3]
        file_name = img_path_list[-1]  # {age}_filenum.jpg
        folder_name = img_path_list[-2]  # label

        img = Image.open(img_path)
        if dataset_name == self.agedb_imgs_folder_name:
            label = self.agedb_class_list.index(folder_name) + self.vgg_class_num
            age = int(file_name.split('_')[-1].strip('.jpg'))
        elif dataset_name == self.vgg_imgs_folder_name:
            label = int(folder_name)
            age = int(file_name.split('_')[0])  # this is actually meaningless
        else:
            print('Something went wrong... What have you done!')
            assert False

        if self.transform is not None:
            img = self.transform(img)

        return img, label, age


class VGGAgeDBDataset(Dataset):
    '''
    Joint DB of VGG and AgeDB
    VGG with pseudo age labels dataset
    directory structure
        root/person_i/{age}_filenum.jpg
    AGE DB with actual labels
        root/person_name/filenum_{age}.jpg

    Store image directories at init phase

    Returns image, label, age
    '''
    def __init__(self, vgg_imgs_folder, agedb_imgs_folder, train_transforms):
        self.vgg_imgs_folder_name = vgg_imgs_folder.split('/')[-1]
        self.agedb_imgs_folder_name = agedb_imgs_folder.split('/')[-1]
        self.transform = train_transforms
        self.vgg_class_num = len(os.listdir(vgg_imgs_folder))
        self.agedb_class_list = os.listdir(agedb_imgs_folder)
        self.agedb_class_num = len(self.agedb_class_list)
        self.class_num = self.vgg_class_num + self.agedb_class_num

        total_list = []
        for (dirpath, _, filenames) in os.walk(vgg_imgs_folder):
            total_list += [os.path.join(dirpath, file) for file in filenames]
        for (dirpath, _, filenames) in os.walk(agedb_imgs_folder):
            total_list += [os.path.join(dirpath, file) for file in filenames]

        self.total_imgs = len(total_list)
        self.total_list = total_list
        
    def __len__(self):
        return self.total_imgs
    
    def __getitem__(self, index):

        img_path = self.total_list[index]
        img_path_list = img_path.split('/')
        dataset_name = img_path_list[-3]
        file_name = img_path_list[-1] # {age}_filenum.jpg
        folder_name = img_path_list[-2]# label

        img = Image.open(img_path)
        if dataset_name == self.agedb_imgs_folder_name:
            label = self.agedb_class_list.index(folder_name) + self.vgg_class_num
            age = int(file_name.split('_')[-1].strip('.jpg'))
        elif dataset_name == self.vgg_imgs_folder_name:
            label = int(folder_name) 
            age = int(file_name.split('_')[0]) # this is actually meaningless
        else:
            print('Something went wrong... What have you done!')
            assert False

        if self.transform is not None:
            img = self.transform(img)

        return img, label, age

class VGGAgeDBInstaDataset(Dataset):
    '''
    Joint DB of VGG, AgeDB, Insta Dataset

    VGG with pseudo age labels dataset
    directory structure
        root/person_i/{age}_filenum.jpg
    AGE DB with actual labels
    directory structure
        root/person_name/filenum_{age}.jpg
    Insta DB with no labels
    directory structure
        root/person_name/filenum.png

    Store image directories at init phase

    Returns image, label, age
    '''
    def __init__(self, vgg_imgs_folder, agedb_imgs_folder, insta_imgs_folder, train_transforms):

        self.vgg_imgs_folder_name = vgg_imgs_folder.split('/')[-1]
        self.agedb_imgs_folder_name = agedb_imgs_folder.split('/')[-1]
        self.insta_imgs_folder_name = insta_imgs_folder.split('/')[-1]

        self.transform = train_transforms

        self.vgg_class_num = len(os.listdir(vgg_imgs_folder))

        self.agedb_class_list = os.listdir(agedb_imgs_folder)
        self.agedb_class_num = len(self.agedb_class_list)

        # self.insta_class_list = os.listdir(insta_imgs_folder)
        # self.insta_class_num = len(self.insta_class_list)

        # self.class_num = self.vgg_class_num + self.agedb_class_num + self.insta_class_num
        self.class_num = self.vgg_class_num + self.agedb_class_num

        total_list = []
        vgg_list, agedb_list, insta_list = [], [], []
        index_list, identity_list = [], []
        # for (dirpath, _, filenames) in os.walk(vgg_imgs_folder):
        #     vgg_list += [os.path.join(dirpath, file) for file in filenames]
        for index, (dirpath, _, filenames) in enumerate(os.walk(agedb_imgs_folder)):
            for j, file in enumerate(filenames):
                print(index, file)
                index_list.append(index)
                identity_list.append(filenames[0].split('_')[1])
                if j >= 1:
                    print(j)
                    break
        # print(index_list)
        # print(identity_list)
        # index_list = list(index_list)
        # identity_list = list(identity_list)
        import pdb; pdb.set_trace()

        import pandas as pd
        df = pd.DataFrame(list(zip(index_list, identity_list)), columns = ['index', 'identity'])
        df.to_csv('agedb_id.csv')
        import sys
        sys.exit(0)
            # for file in filenames:
            #     print(index, file)
            #     import pdb; pdb.set_trace()
            # agedb_list += [os.path.join(dirpath, file) for file in filenames]

        # for (dirpath, _, filenames) in os.walk(insta_imgs_folder):
        #     insta_list += [os.path.join(dirpath, file) for file in filenames]


        total_list = vgg_list + agedb_list + insta_list
        #
        # total_list = []
        # for (dirpath, _, filenames) in os.walk(vgg_imgs_folder):
        #     total_list += [os.path.join(dirpath, file) for file in filenames]
        # for (dirpath, _, filenames) in os.walk(agedb_imgs_folder):
        #     total_list += [os.path.join(dirpath, file) for file in filenames]
        # for (dirpath, _, filenames) in os.walk(insta_imgs_folder):
        #     total_list += [os.path.join(dirpath, file) for file in filenames]


        self.total_imgs = len(total_list)
        self.total_list = total_list
        
    def __len__(self):
        return self.total_imgs
    
    def __getitem__(self, index):

        img_path = self.total_list[index]
        img_path_list = img_path.split('/')
        dataset_name = img_path_list[-3]
        file_name = img_path_list[-1] # {age}_filenum.jpg
        folder_name = img_path_list[-2]# label

        img = Image.open(img_path)
        if dataset_name == self.agedb_imgs_folder_name:
            label = self.agedb_class_list.index(folder_name) + self.vgg_class_num
            age = int(file_name.split('_')[-1].strip('.jpg'))
        elif dataset_name == self.insta_imgs_folder_name:
            label = self.insta_class_list.index(folder_name) + self.vgg_class_num + self.agedb_class_num
            age = 1 # also meaningless
        elif dataset_name == self.vgg_imgs_folder_name:
            label = int(folder_name) 
            age = int(file_name.split('_')[0]) # this is actually meaningless
        else:
            print('Something went wrong. What have you done!')
            assert False

        if self.transform is not None:
            img = self.transform(img)

        return img, label, age


if __name__ == '__main__':
    # TEST CODE FOR VGGLabeledDateset
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = VGGLabeledDataset('/home/nas1_userE/Face_dataset/Vgg_age_label/', train_transforms=train_transform)
    loader = DataLoader(ds, batch_size=2)
    i, l, a = next(loader)
    print(l)
