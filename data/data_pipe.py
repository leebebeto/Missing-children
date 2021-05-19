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
import bcolz


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
    casia_folder = '/home/nas1_userE/jungsoolee/Face_dataset/CASIA_REAL_NATIONAL'
    # casia_folder =  '/home/nas1_userD/yonggyu/Face_dataset/casia'
    if conf.data_mode == 'casia_mixup':
        casia_prettiermonster100_folder = '/home/nas1_userE/jungsoolee/Face_dataset/CASIA_REAL_PrettierMonster100'

    print(casia_folder)
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

    if conf.data_mode == 'casia_vgg':
        ds = CASIAVGGDataset(casia_folder, mode=conf.casia_vgg_mode, train_transforms=train_transform)
        class_num = ds.class_num
        print('casia_vgg loader generated')

    elif conf.data_mode  == 'vgg':
        print(f'vgg folder: {conf.vgg_folder}')
        ds, class_num = get_train_dataset(conf.vgg_folder)
        print('vgg loader generated')

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

    elif conf.data_mode == 'casia_mixup':
        ds = CasiaMixupDataset(casia_folder, casia_prettiermonster100_folder, train_transforms=train_transform, conf=conf)
        class_num = ds.class_num
        child_identity = ds.child_identity
        child_identity_min = ds.child_identity_min
        child_identity_max = ds.child_identity_max

        print('casia, mixup loader generated')


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

def get_val_data(data_path):
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    return lfw, lfw_issame

class CASIAVGGDataset(Dataset):
    '''
    CASIA with pseudo age labels dataset
    directory structure
        root/person_i/{age}_filenum.jpg

    Store image directories at init phase

    Returns image, label, age
    '''

    def __init__(self, imgs_folder, mode, train_transforms):
        self.root_dir = imgs_folder
        self.transform = train_transforms
        self.original_class_num = len(os.listdir(imgs_folder))
        self.mode = mode
        self.casia_id = glob.glob(f'{imgs_folder}/*')
        self.casia_index = { int(id.split('/')[-1]): i for i, id in enumerate(self.casia_id)}

        if self.mode == 'random':
            vgg_list = np.random.choice(8631, 100).tolist()

        elif self.mode == 'max':
            with open('/home/nas1_userE/jungsoolee/Face_dataset/vgg_max_first.pickle', 'rb') as f:
                vgg_list = pickle.load(f)
            vgg_list = sorted(vgg_list)

        elif self.mode == 'min':
            with open('/home/nas1_userE/jungsoolee/Face_dataset/vgg_min_first.pickle', 'rb') as f:
                vgg_list = pickle.load(f)

        elif self.mode == 'insta_similar':
            with open('/home/nas1_userE/jungsoolee/Face_dataset/vgg_insta_similar.pickle', 'rb') as f:
                vgg_list = pickle.load(f)
            vgg_list = sorted(vgg_list)

        self.vgg_dict = {id: index for index, id in enumerate(vgg_list)}
        add_id = len(self.vgg_dict)

        total_list = glob.glob(self.root_dir + '/*/*')
        for i in vgg_list:
            total_list += glob.glob(f'/home/nas1_userD/yonggyu/Face_dataset/vgg/{i}/*')

        self.total_imgs = len(total_list)
        self.class_num = self.original_class_num + add_id
        self.total_list = total_list

        print(f'{imgs_folder} length: {self.total_imgs}')

    def __len__(self):
        return self.total_imgs

    def __getitem__(self, index):
        img_path = self.total_list[index]
        img_path_list = img_path.split('/')
        file_name = img_path_list[-1]  # {age}_filenum.jpg

        img = Image.open(img_path)
        if 'vgg' in img_path:
            # vgg starts with 0
            label = self.original_class_num + self.vgg_dict[int(img_path.split('/')[-2])]
        else:
            label = self.casia_index[int(img_path.split('/')[-2])]

        age = int(file_name.split('/')[-1].split('_')[-1][:-4])

        if self.transform is not None:
            img = self.transform(img)

        # age= 0 if age<= 18 else 1

        return img, label


class CasiaMixupDataset(Dataset):
    '''
    Joint DB of Casia, Mixup(BabyMonster) Dataset
    Casia with no age labels
    directory structure
        root/person_name/{age}_filenum.jpg
    BabyMonster with no age labels
    directory structure
        root/id1-id2/filenum.jpg
    Store image directories at init phase
    Returns image, label, age
    '''

    def __init__(self, casia_imgs_folder, babymonster_imgs_folder, train_transforms, conf):

        self.casia_imgs_folder_name = casia_imgs_folder.split('/')[-1]
        self.babymonster_imgs_folder_name = babymonster_imgs_folder.split('/')[-1]

        self.transform = train_transforms

        self.casia_class_list = os.listdir(casia_imgs_folder)
        self.casia_class_num = len(os.listdir(casia_imgs_folder))

        self.age_file = open('/home/nas1_userE/jungsoolee/Face_dataset/casia-webface.txt').readlines()
        self.id2age = { os.path.join(str(int(line.split(' ')[1].split('/')[1])), str(int(line.split(' ')[1].split('/')[2][:-4]))) : float(line.split(' ')[2]) for line in self.age_file}
        self.child_image2age = { os.path.join(str(int(line.split(' ')[1].split('/')[1])), str(int(line.split(' ')[1].split('/')[2][:-4]))) : float(line.split(' ')[2]) for line in self.age_file if float(line.split(' ')[2]) <= 13}
        self.child_image2freq = {id.split('/')[0]: 0 for id in self.child_image2age.keys()}
        for k, v in self.child_image2age.items():
            self.child_image2freq[k.split('/')[0]] += 1

        # sorted in ascending order
        self.child_identity_freq = {int(k): v for k, v in sorted(self.child_image2freq.items(), key=lambda item: item[1])}
        self.child_identity = list(self.child_identity_freq.keys())
        self.child_identity_min = list(self.child_identity_freq.keys())[:conf.new_id + 1]
        self.child_identity_max = list(self.child_identity_freq.keys())[-(conf.new_id + 1):]

        self.babymonster_class_list = os.listdir(babymonster_imgs_folder)
        self.babymonster_class_num = len(os.listdir(babymonster_imgs_folder))

        self.class_num = self.casia_class_num + self.babymonster_class_num

        total_list = []
        for (dirpath, _, filenames) in os.walk(casia_imgs_folder):
            total_list += [os.path.join(dirpath, file) for file in filenames]
        for (dirpath, _, filenames) in os.walk(babymonster_imgs_folder):
            # NOTE : random suffle and reduce images to n
            id_list = [os.path.join(dirpath, file) for file in filenames]
            if id_list is not []:
                random.shuffle(id_list)
                id_list = id_list[:20]
            total_list += id_list

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
        if dataset_name == self.casia_imgs_folder_name:
            label = self.casia_class_list.index(folder_name)
            # age = int(file_name.split('_')[0]) # this is actually meaningless
            age = 1
        elif dataset_name == self.babymonster_imgs_folder_name:
            label = self.babymonster_class_list.index(folder_name)
            age = 0
        else:
            print('Something went wrong. What have you done!')
            assert False

        if self.transform is not None:
            img = self.transform(img)

        return img, label, age


class CASIADataset(Dataset):
    '''
    CASIA with pseudo age labels dataset
    directory structure
        root/person_i/{age}_filenum.jpg

    Store image directories at init phase

    Returns image, label, age
    '''

    def __init__(self, imgs_folder, train_transforms, conf):
        self.root_dir = imgs_folder
        self.transform = train_transforms
        self.class_num = len(os.listdir(imgs_folder))
        self.age_file = open('/home/nas1_userE/jungsoolee/Face_dataset/casia-webface.txt').readlines()
        self.id2age = { os.path.join(str(int(line.split(' ')[1].split('/')[1])), str(int(line.split(' ')[1].split('/')[2][:-4]))) : float(line.split(' ')[2]) for line in self.age_file}
        self.child_image2age = { os.path.join(str(int(line.split(' ')[1].split('/')[1])), str(int(line.split(' ')[1].split('/')[2][:-4]))) : float(line.split(' ')[2]) for line in self.age_file if float(line.split(' ')[2]) <= 13}
        self.child_image2freq = {id.split('/')[0]: 0 for id in self.child_image2age.keys()}
        for k, v in self.child_image2age.items():
            self.child_image2freq[k.split('/')[0]] += 1

        # sorted in ascending order
        self.child_identity_freq = {int(k): v for k, v in sorted(self.child_image2freq.items(), key=lambda item: item[1])}
        self.child_identity = list(self.child_identity_freq.keys())
        self.child_identity_min = list(self.child_identity_freq.keys())[:conf.new_id + 1]
        self.child_identity_max = list(self.child_identity_freq.keys())[-(conf.new_id + 1):]

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

        # self.class_num_list = [7336, 483287]
        self.total_list = total_list
        print(f'{imgs_folder} length: {self.total_imgs}')

    def __len__(self):
        return self.total_imgs

    def __getitem__(self, index):
        img_path = self.total_list[index]
        img_path_list = img_path.split('/')
        file_name = img_path_list[-1]  # {age}_filenum.jpg
        id_img = '/'.join((img_path.split('/')[-2], img_path.split('/')[-1][:-4]))
        try:
            age = self.id2age[id_img]
        except:
            age = 30

        img = Image.open(img_path)
        label = int(img_path.split('/')[-2])
        # age = int(file_name.split('/')[-1].split('_')[-1][:-4])

        if self.transform is not None:
            img = self.transform(img)

        age= 0 if age<= 13 else 1

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


if __name__ == '__main__':
    # TEST CODE FOR VGGLabeledDateset
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = VGGLabeledDataset('./dataset/Vgg_age_label/', train_transforms=train_transform)
    loader = DataLoader(ds, batch_size=2)
    i, l, a = next(loader)
    print(l)
