from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision.datasets import ImageFolder, folder
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import pickle
import torch
from tqdm import tqdm
import glob
import pdb
import os
import random
# import bcolz


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
    # casia_folder =  './dataset/CASIA_112'
    # casia_folder = os.path.join(conf.home,'dataset/CASIA_REAL_NATIONAL')
    # casia_folder =  '/home/nas1_userD/yonggyu/Face_dataset/casia'

    # casia_folder = './dataset/CASIA_REAL_NATIONAL/CASIA_REAL_NATIONAL'
    # casia_folder = '../bebeto_face_dataset/CASIA_REAL_NATIONAL/CASIA_REAL_NATIONAL'
    ms1m_root = '/home/nas4_user/jungsoolee/Face_dataset/ms1m-refined-112/ms1m'

    casia_folder = '/home/nas4_user/jungsoolee/Face_dataset/CASIA_REAL_NATIONAL'
    print(conf.data_mode)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if conf.data_mode == 'casia':
        ds = CASIADataset(casia_folder, train_transforms=train_transform,  conf=conf)
        class_num = ds.class_num
        child_identity = ds.child_identity
        child_identity_min = ds.child_identity_min
        child_identity_max = ds.child_identity_max
        print('casia loader generated')
    elif conf.data_mode  == 'vgg':
        print(f'vgg folder: {conf.vgg_folder}')
        ds, class_num = get_train_dataset(conf.vgg_folder)
        print('vgg loader generated')
    elif conf.data_mode  == 'ms1m':
        ms1m_root = '/home/nas4_user/jungsoolee/Face_dataset/ms1m-refined-112/ms1m'
        # ms1m_root = './dataset/ms1m'
        # ms1m_root = './dataset/'
        ds = MS1MDataset(ms1m_root, train_transforms=train_transform,  conf=conf)
        class_num = ds.class_num
        child_identity = ds.child_identity
        child_identity_min = ds.child_identity_min
        child_identity_max = ds.child_identity_max
        print('ms1m loader generated')
    elif conf.data_mode == 'vgg_agedb':
        ds = VGGAgeDBDataset(conf.vgg_folder, conf.agedb_folder, train_transforms=train_transform)
        class_num = ds.class_num
        print('vgg_agedb loader generated')
    elif conf.data_mode == 'vgg_insta':
        ds = VGGAgeDBDataset(conf.vgg_folder, conf.insta_folder, train_transforms=train_transform)
        class_num = ds.class_num
        print('vgg_insta loader generated')
    elif conf.data_mode == 'vgg_agedb_insta':
        ds = VGGAgeDBInstaDataset(conf.vgg_folder, conf.agedb_folder, conf.insta_folder,
                                  train_transforms=train_transform)
        class_num = ds.class_num
        print('vgg_agedb_insta loader generated')
    elif conf.data_mode == 'webface':
        webface_root = '/home/nas1_userB/dataset/WebFace42M/img_folder'
        ds = WebFace42M(webface_root, train_transforms=train_transform, conf=conf)
        class_num = ds.class_num
        child_identity = None # Can change to ALL
        child_identity_max, child_identity_min = 0, 0
    elif conf.data_mode == 'casia_cctv':
        cctv_folder = '/home/nas1_temp/jooyeolyun/Datasets/FaceRecog_TestSet/img_renamed'
        ds = CASIACCTVDataset(casia_folder, cctv_folder, train_transforms=train_transform, conf=conf)
        class_num = ds.class_num
        child_identity = ds.child_identity
        child_identity_min = ds.child_identity_min
        child_identity_max = ds.child_identity_max
    elif conf.data_mode == 'ms1m_cctv':
        cctv_folder = '/home/nas1_temp/jooyeolyun/Datasets/FaceRecog_TestSet/img_renamed'
        ds = MS1MCCTVDataset(ms1m_root, cctv_folder, train_transforms=train_transform, conf=conf)
        class_num = ds.class_num
        child_identity = ds.child_identity
        child_identity_min = ds.child_identity_min
        child_identity_max = ds.child_identity_max
    else:
        print('Wrong dataset name')
        raise NotImplementedError

    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=True, pin_memory=True, num_workers=conf.num_workers)

    return loader, class_num, ds, child_identity, child_identity_min, child_identity_max

def get_val_pair(path, name):
    '''
    Returns image pairs with labels
        carray: numpy-like array with image pairs
        issame: boolean list of image pair validity
    '''
    carray = bcolz.carray(rootdir = os.path.join(path, name), mode='r')
    issame = np.load(os.path.join(path, '{}_list.npy'.format(name)))    
    return carray, issame

def get_val_data(data_path, name='cfp_fp'):
    data, data_issame = get_val_pair(data_path, name)
    return data, data_issame

class CASIADataset(Dataset):
    '''
    CASIA with pseudo age labels dataset
    directory structure
        root/person_i/{age}_filenum.jpg

    Store image directories at init phase

    Returns image, label, age
    '''

    def __init__(self, imgs_folder, train_transforms, conf):
        self.conf = conf
        self.root_dir = imgs_folder
        self.transform = train_transforms if conf.low_res is False \
                        else transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.Resize(size=(56,56)),
                            transforms.Resize(size=(112,112)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                        ])
        self.class_num = len(os.listdir(imgs_folder))
        # self.age_file = open('./dataset/casia-webface.txt').readlines()
        # self.age_file = open('../bebeto_face_dataset/casia-webface.txt').readlines()
        self.age_file = open('/home/nas4_user/jungsoolee/Face_dataset/casia-webface.txt').readlines()
        self.id2age = { os.path.join(str(int(line.split(' ')[1].split('/')[1])), str(int(line.split(' ')[1].split('/')[2][:-4]))) : float(line.split(' ')[2]) for line in self.age_file}
        self.child_image2age = { os.path.join(str(int(line.split(' ')[1].split('/')[1])), str(int(line.split(' ')[1].split('/')[2][:-4]))) : float(line.split(' ')[2]) for line in self.age_file if float(line.split(' ')[2]) < 13}
        self.child_image2freq = {id.split('/')[0]: 0 for id in self.child_image2age.keys()}
        for k, v in self.child_image2age.items():
            self.child_image2freq[k.split('/')[0]] += 1

        # sorted in ascending order
        # self.child_identity_freq = {int(k): v for k, v in sorted(self.child_image2freq.items(), key=lambda item: item[1])}
        self.child_identity_freq = {int(k): v for k, v in sorted(self.child_image2freq.items(), key=lambda item: item[1]) if v >= conf.child_filter}
        # if conf.child_filter:
        #     self.child_identity_freq = {int(k): v for k, v in sorted(self.child_image2freq.items(), key=lambda item: item[1]) if v >= 5}
        self.child_identity = list(self.child_identity_freq.keys())
        print(f'child number: {len(self.child_identity)}')
        self.child_identity_min = list(self.child_identity_freq.keys())[:conf.new_id + 1]
        self.child_identity_max = list(self.child_identity_freq.keys())[-(conf.new_id + 1):]

        # oversample
        if self.conf.use_oversample:
            self.child_list, self.adult_list = [], []
            for k, v in tqdm(self.id2age.items()):
                if v > 12:
                    self.adult_list.append(os.path.join(self.root_dir, k) + '.jpg')
                else:
                    self.child_list.append(os.path.join(self.root_dir, k) + '.jpg')

            self.child_list = self.child_list * (int(len(self.adult_list) / len(self.child_list)) + 1)
            if self.conf.oversample_ratio == 1.0:
                self.child_list = self.child_list[: len(self.adult_list)]
            else:
                self.child_list = self.child_list[: int(len(self.child_list) * self.conf.oversample_ratio)]

            print(len(self.child_list), len(self.adult_list))
            total_list = self.child_list + self.adult_list
        else:
            total_list = glob.glob(self.root_dir + '/*/*')

        # total_list = glob.glob(self.root_dir + '/*/*')
        self.total_imgs = len(total_list)

        self.total_list = total_list
        print(f'{imgs_folder} length: {self.total_imgs}')

    def __len__(self):
        return self.total_imgs

    def __getitem__(self, index):
        img_path = self.total_list[index]
        img_path_list = img_path.split('/')
        file_name = img_path_list[-1]  # {age}_filenum.jpg
        id_img = '/'.join((img_path.split('/')[-2], img_path.split('/')[-1].split('_')[0]))
        if 'jpg' in id_img:
            id_img = id_img[:-4]
        try:
            age = self.id2age[id_img]
        except:
            age = 30

        img = Image.open(img_path)
        label = int(img_path.split('/')[-2])
        # age = int(file_name.split('/')[-1].split('_')[-1][:-4])

        if self.transform is not None:
            img = self.transform(img)

        if self.conf.loss == 'DAL':
            if age < 13:
                age = 0
            elif age >= 13 and age < 19:
                age = 1
            elif age >= 19 and age < 26:
                age = 2
            elif age >= 26 and age < 36:
                age = 3
            elif age >= 36 and age < 46:
                age = 4
            elif age >= 46 and age < 56:
                age = 5
            elif age >= 56 and age < 66:
                age = 6
            elif age >= 66:
                age = 7
        elif self.conf.loss == 'OECNN':
            age = int(age)
        else:
            age= 0 if age< 13 else 1

        return img, label, age



class MS1MDataset(Dataset):
    '''
    MS1M with pseudo age labels dataset
    directory structure
        root/person_i/{age}_filenum.jpg

    Store image directories at init phase

    Returns image, label, age
    '''

    def __init__(self, imgs_folder, train_transforms, conf):
        self.conf = conf
        self.root_dir = imgs_folder
        self.transform = train_transforms
        self.class_num = len(os.listdir(imgs_folder))
        # self.age_file = open('./dataset/ms1m.txt').readlines()
        self.age_file = open('/home/nas4_user/jungsoolee/Face_dataset/ms1m.txt').readlines()
        self.id2age = {os.path.join(str(int(line.split(' ')[1].split('/')[1])), str(int(line.split(' ')[1].split('/')[2][:-4]))) : float(line.split(' ')[2]) for line in self.age_file}
        self.child_image2age = { os.path.join(str(int(line.split(' ')[1].split('/')[1])), str(int(line.split(' ')[1].split('/')[2][:-4]))) : float(line.split(' ')[2]) for line in self.age_file if float(line.split(' ')[2]) < 13}
        self.child_image2freq = {id.split('/')[0]: 0 for id in self.child_image2age.keys()}
        for k, v in self.child_image2age.items():
            self.child_image2freq[k.split('/')[0]] += 1

        # sorted in ascending order
        # self.child_identity_freq = {int(k): v for k, v in sorted(self.child_image2freq.items(), key=lambda item: item[1])}
        self.child_identity_freq = {int(k): v for k, v in sorted(self.child_image2freq.items(), key=lambda item: item[1]) if v >= conf.child_filter}
        # if conf.child_filter:
        #     self.child_identity_freq = {int(k): v for k, v in sorted(self.child_image2freq.items(), key=lambda item: item[1]) if v >= 5}
        self.child_identity = list(self.child_identity_freq.keys())
        print(f'child number: {len(self.child_identity)}')
        self.child_identity_min = list(self.child_identity_freq.keys())[:conf.new_id + 1]
        self.child_identity_max = list(self.child_identity_freq.keys())[-(conf.new_id + 1):]

        total_list = glob.glob(self.root_dir + '/*/*.jpg')
        self.total_imgs = len(total_list)

        self.total_list = total_list
        print(f'{imgs_folder} length: {self.total_imgs}')
    def __len__(self):
        return self.total_imgs

    def __getitem__(self, index):
        img_path = self.total_list[index]
        img_path_list = img_path.split('/')
        file_name = img_path_list[-1]  # {age}_filenum.jpg
        id_img = '/'.join((img_path.split('/')[-2], img_path.split('/')[-1].split('_')[0]))
        if 'jpg' in id_img:
            id_img = id_img[:-4]
        try:
            age = self.id2age[id_img]
        except:
            age = 30

        img = Image.open(img_path)
        label = int(img_path.split('/')[-2])
        # age = int(file_name.split('/')[-1].split('_')[-1][:-4])

        if self.transform is not None:
            img = self.transform(img)

        if self.conf.loss == 'DAL':
            if age < 13:
                age = 0
            elif age >= 13 and age < 19:
                age = 1
            elif age >= 19 and age < 26:
                age = 2
            elif age >= 26 and age < 36:
                age = 3
            elif age >= 36 and age < 46:
                age = 4
            elif age >= 46 and age < 56:
                age = 5
            elif age >= 56 and age < 66:
                age = 6
            elif age >= 66:
                age = 7
        elif self.conf.loss == 'OECNN':
            age = int(age)
        else:
            age= 0 if age< 13 else 1

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
        # file_name = img_path_list[-1]  # {age}_filenum.jpg
        folder_name = img_path_list[-2]  # label

        img = Image.open(img_path)
        if dataset_name == self.agedb_imgs_folder_name:
            label = self.agedb_class_list.index(folder_name) + self.vgg_class_num
            # age = int(file_name.split('_')[-1].strip('.jpg'))
        elif dataset_name == self.vgg_imgs_folder_name:
            # folder_name = folder_name.split('n')[1]
            label = int(folder_name)
            # age = int(file_name.split('_')[0])  # this is actually meaningless
        else:
            print('Something went wrong... What have you done!')
            assert False

        if self.transform is not None:
            img = self.transform(img)

        age = 0
        return img, label


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

        self.insta_class_list = os.listdir(insta_imgs_folder)
        self.insta_class_num = len(self.insta_class_list)

        self.class_num = self.vgg_class_num + self.agedb_class_num + self.insta_class_num

        total_list = []
        for (dirpath, _, filenames) in os.walk(vgg_imgs_folder):
            total_list += [os.path.join(dirpath, file) for file in filenames]
        for (dirpath, _, filenames) in os.walk(agedb_imgs_folder):
            total_list += [os.path.join(dirpath, file) for file in filenames]
        for (dirpath, _, filenames) in os.walk(insta_imgs_folder):
            total_list += [os.path.join(dirpath, file) for file in filenames]

        self.total_imgs = len(total_list)
        self.total_list = total_list

    def __len__(self):
        return self.total_imgs

    def __getitem__(self, index):

        img_path = self.total_list[index]
        img_path_list = img_path.split('/')
        dataset_name = img_path_list[-3]
        file_name = img_path_list[-1]  # {age}_filenum.jpg
        folder_name = img_path_list[-2]  # label

        img = Image.open(img_path)
        if dataset_name == self.agedb_imgs_folder_name:
            label = self.agedb_class_list.index(folder_name) + self.vgg_class_num
            _age = int(file_name.split('_')[-1].strip('.jpg'))
            age = 0 if _age < 13 else 1
        elif dataset_name == self.insta_imgs_folder_name:
            label = self.insta_class_list.index(folder_name) + self.vgg_class_num + self.agedb_class_num
            age = 0  # also meaningless
        elif dataset_name == self.vgg_imgs_folder_name:
            label = int(folder_name)
            # age = int(file_name.split('_')[0]) # this is actually meaningless
            age = 1
        else:
            print('Something went wrong. What have you done!')
            assert False

        if self.transform is not None:
            img = self.transform(img)

        return img, label



class WebFace42M(Dataset):
    '''
    WebFace42M dataset
    Returns image, label, age
    '''

    def __init__(self, imgs_folder, train_transforms, conf):
        self.conf = conf
        self.root_dir = imgs_folder
        self.transform = train_transforms

        # CLASS LIST
        if os.path.exists(os.path.join(imgs_folder, 'webface_class_list.pkl')):
            with open(os.path.join(imgs_folder, 'webface_class_list.pkl'), 'rb') as f:
                self.class_num = len(pickle.load(f))
        else:
            class_list = os.listdir(imgs_folder)
            self.class_num = len(os.listdir(imgs_folder))
            with open(os.path.join(imgs_folder, 'webface_class_list.pkl'), 'wb') as f:
                pickle.dump(class_list, f)

        # TOTAL LIST
        if os.path.exists(os.path.join(imgs_folder, 'webface_total_list.pkl')):
            with open(os.path.join(imgs_folder, 'webface_total_list.pkl'), 'rb') as f:
                total_list = pickle.load(f)
        else:
            total_list = glob.glob(self.root_dir + '/*/*')
            with open(os.path.join(imgs_folder, 'webface_total_list.pkl'), 'wb') as f:
                pickle.dump(total_list, f)

        self.total_imgs = len(total_list)

        self.total_list = total_list
        print(f'{imgs_folder} length: {self.total_imgs}')

    def __len__(self):
        return self.total_imgs

    def __getitem__(self, index):
        img_path = self.total_list[index]

        img = Image.open(img_path)
        label = int(img_path.split('/')[-2].split('_')[-1]) # CHECK

        if self.transform is not None:
            img = self.transform(img)

        age = 1

        return img, label, age


class CASIACCTVDataset(Dataset):
    '''
    CASIA with pseudo age labels dataset
    directory structure
        root/person_i/{age}_filenum.jpg

    Not built for Inter-prototype loss

    Returns image, label, age
    '''

    def __init__(self, imgs_folder, cctv_folder, train_transforms, conf):
        self.conf = conf
        self.root_dir = imgs_folder
        self.transform = train_transforms if conf.low_res is False \
                        else transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.Resize(size=(57,57)),
                            transforms.Resize(size=(112,112)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                        ])
        self.casia_class_num = len(os.listdir(imgs_folder))
        self.cctv_class_num =  len(os.listdir(cctv_folder))
        self.class_num = self.casia_class_num + self.cctv_class_num

        self.casia_list = glob.glob(self.root_dir + '/*/*') 
        self.cctv_list = glob.glob(cctv_folder + '/*/*')
        self.total_list = self.casia_list + self.cctv_list

        self.total_imgs = len(self.total_list)

        self.age_file = open('/home/nas4_user/jungsoolee/Face_dataset/casia-webface.txt').readlines()
        self.id2age = { os.path.join(str(int(line.split(' ')[1].split('/')[1])), str(int(line.split(' ')[1].split('/')[2][:-4]))) : float(line.split(' ')[2]) for line in self.age_file}
        self.child_image2age = { os.path.join(str(int(line.split(' ')[1].split('/')[1])), str(int(line.split(' ')[1].split('/')[2][:-4]))) : float(line.split(' ')[2]) for line in self.age_file if float(line.split(' ')[2]) < 13}
        self.child_image2freq = {id.split('/')[0]: 0 for id in self.child_image2age.keys()}
        for k, v in self.child_image2age.items():
            self.child_image2freq[k.split('/')[0]] += 1

        # sorted in ascending order
        # self.child_identity_freq = {int(k): v for k, v in sorted(self.child_image2freq.items(), key=lambda item: item[1])}
        self.child_identity_freq = {int(k): v for k, v in sorted(self.child_image2freq.items(), key=lambda item: item[1]) if v >= conf.child_filter}
        # if conf.child_filter:
        #     self.child_identity_freq = {int(k): v for k, v in sorted(self.child_image2freq.items(), key=lambda item: item[1]) if v >= 5}
        self.child_identity = list(self.child_identity_freq.keys())
        print(f'child number: {len(self.child_identity)}')
        self.child_identity_min = list(self.child_identity_freq.keys())[:conf.new_id + 1]
        self.child_identity_max = list(self.child_identity_freq.keys())[-(conf.new_id + 1):]

        print(f'{imgs_folder} length: {self.total_imgs}')

    def __len__(self):
        return self.total_imgs

    def __getitem__(self, index):
        img_path = self.total_list[index]
        id_img = '/'.join((img_path.split('/')[-2], img_path.split('/')[-1].split('_')[0]))
        if 'jpg' in id_img:
            id_img = id_img[:-4]
        try:
            age = self.id2age[id_img]
        except:
            age = 30

        age= 0 if age< 13 else 1

        img = Image.open(img_path)
        label = int(img_path.split('/')[-2]) if index < len(self.casia_list) \
            else self.casia_class_num + int(img_path.split('/')[-2]) - 1 # class starts from 1
        age = 1 if index < len(self.casia_list) else age

        if self.transform is not None:
            img = self.transform(img)

        return img, label, age



class MS1MCCTVDataset(Dataset):
    '''
    CASIA with pseudo age labels dataset
    directory structure
        root/person_i/{age}_filenum.jpg

    Not built for Inter-prototype loss

    Returns image, label, age
    '''

    def __init__(self, imgs_folder, cctv_folder, train_transforms, conf):
        self.conf = conf
        self.root_dir = imgs_folder
        self.transform = train_transforms
        self.ms1m_class_num = len(os.listdir(imgs_folder))
        self.cctv_class_num =  len(os.listdir(cctv_folder))
        self.class_num = self.ms1m_class_num + self.cctv_class_num

        self.ms1m_list = glob.glob(self.root_dir + '/*/*') 
        self.cctv_list = glob.glob(cctv_folder + '/*/*')
        self.total_list = self.ms1m_list + self.cctv_list

        self.total_imgs = len(self.total_list)

        print(f'{imgs_folder} length: {self.total_imgs}')

        self.age_file = open('/home/nas4_user/jungsoolee/Face_dataset/ms1m.txt').readlines()
        self.id2age = {os.path.join(str(int(line.split(' ')[1].split('/')[1])), str(int(line.split(' ')[1].split('/')[2][:-4]))) : float(line.split(' ')[2]) for line in self.age_file}
        self.child_image2age = { os.path.join(str(int(line.split(' ')[1].split('/')[1])), str(int(line.split(' ')[1].split('/')[2][:-4]))) : float(line.split(' ')[2]) for line in self.age_file if float(line.split(' ')[2]) < 13}
        self.child_image2freq = {id.split('/')[0]: 0 for id in self.child_image2age.keys()}
        for k, v in self.child_image2age.items():
            self.child_image2freq[k.split('/')[0]] += 1

        # sorted in ascending order
        # self.child_identity_freq = {int(k): v for k, v in sorted(self.child_image2freq.items(), key=lambda item: item[1])}
        self.child_identity_freq = {int(k): v for k, v in sorted(self.child_image2freq.items(), key=lambda item: item[1]) if v >= conf.child_filter}
        # if conf.child_filter:
        #     self.child_identity_freq = {int(k): v for k, v in sorted(self.child_image2freq.items(), key=lambda item: item[1]) if v >= 5}
        self.child_identity = list(self.child_identity_freq.keys())
        print(f'child number: {len(self.child_identity)}')
        self.child_identity_min = list(self.child_identity_freq.keys())[:conf.new_id + 1]
        self.child_identity_max = list(self.child_identity_freq.keys())[-(conf.new_id + 1):]

    def __len__(self):
        return self.total_imgs

    def __getitem__(self, index):
        img_path = self.total_list[index]
        id_img = '/'.join((img_path.split('/')[-2], img_path.split('/')[-1].split('_')[0]))
        if 'jpg' in id_img:
            id_img = id_img[:-4]
        try:
            age = self.id2age[id_img]
        except:
            age = 30
        age= 0 if age< 13 else 1

        img = Image.open(img_path)
        label = int(img_path.split('/')[-2]) if index < len(self.ms1m_list) \
            else self.ms1m_class_num + int(img_path.split('/')[-2]) - 1 # class starts from 1
        age = 1 if index < len(self.ms1m_list) else age

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
    # ds = VGGLabeledDataset('./dataset/Vgg_age_label/', train_transforms=train_transform)
    ds = WebFace42M(imgs_folder='/home/nas1_userB/dataset/WebFace42M/img_folder', train_transforms=train_transform, conf=None)
    loader = DataLoader(ds, batch_size=2)
    i, l, a = next(iter(loader))
    pdb.set_trace()
    print(l)
