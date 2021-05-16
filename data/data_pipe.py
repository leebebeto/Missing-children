from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision.datasets import ImageFolder, folder
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import bcolz
import pickle
import torch
import mxnet as mx
from tqdm import tqdm

import pdb
import os
import glob
import random


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
    '''
    Returns (dataloader, num_class) according to conf.data_mode
    loader for training classifier
    '''
    if conf.data_mode in ['ms1m', 'ms1m_vgg_concat']:
        ms1m_ds, ms1m_class_num = get_train_dataset(os.path.join(conf.ms1m_folder, 'imgs'))
        print('ms1m loader generated')
    if conf.data_mode in ['vgg', 'ms1m_vgg_concat']:
        vgg_ds = VGGLabeledDataset(conf.vgg_folder, train_transforms=conf.train_transform)
        vgg_class_num = vgg_ds.class_num
        print('vgg loader generated')  
    if conf.data_mode == 'casia':
        ds = CasiaDataset(conf.casia_folder, train_transforms=conf.train_transform)
        class_num = ds.class_num
        print('casia generated')   
    if conf.data_mode == 'vgg_agedb':
        ds = VGGAgeDBDataset(conf.vgg_folder, conf.agedb_folder, train_transforms=conf.train_transform)
        class_num = ds.class_num
        print('vgg, agedb loader generated')
    if conf.data_mode == 'vgg_agedb_insta':
        ds = VGGAgeDBInstaDataset(conf.vgg_folder, conf.agedb_folder, conf.insta_folder, train_transforms=conf.train_transform)
        class_num = ds.class_num
    if conf.data_mode == 'vgg_agedb_balanced':
        ds = VGGAgeDBDataset(conf.vgg_folder, conf.agedb_balanced_folder, train_transforms=conf.train_transform)
        class_num = ds.class_num
        print('vgg, with agedb balanced')
    if conf.data_mode == 'casia_agedb_insta':
        ds = CasiaAgeDBInstaDataset(conf.casia_folder, conf.agedb_folder, conf.insta_folder, train_transforms=conf.train_transform)
        class_num = ds.class_num
        print('casia, agedb, insta loader generated')
    if conf.data_mode == 'casia_insta':
        ds = CasiaInstaDataset(conf.casia_folder, conf.insta_folder, train_transforms=conf.train_transform)
        class_num = ds.class_num
        print('casia, insta loader generated')
    if conf.data_mode == 'casia_agedb':
        ds = CasiaAgeDBDataset(conf.casia_folder, conf.agedb_folder, train_transforms=conf.train_transform)
        class_num = ds.class_num
        print('casia, agedb loader generated')
    if conf.data_mode == 'casia_mixup':
        ds = CasiaMixupDataset(conf.casia_folder, conf.casia_prettymonster500_folder, train_transforms=conf.train_transform)
        class_num = ds.class_num
        print('casia, mixup loader generated')
        
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

    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=True, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    return loader, class_num 

def get_train_loader_d(conf):
    '''
    Returns two loaders, child loader and adult loader
    child path and adult path is in conf
    NOTE: no num_class here
    '''
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds_c = ChildDataset(conf.child_folder, train_transform)
    ds_a = AdultDataset(conf.adult_folder, train_transform)

    loader_c = DataLoader(ds_c, batch_size=conf.batch_size_d, shuffle=True, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    loader_d = DataLoader(ds_a, batch_size=conf.batch_size_d, shuffle=True, pin_memory=conf.pin_memory, num_workers=conf.num_workers)

    return loader_c, loader_d

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

    # # for debugging
    # if name == 'lfw':
    #     temp = carray[0][::-1, :, :]
    #     save_image(torch.tensor(temp.copy()), 'asd1.png', normalize=True)

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

class CasiaDataset(Dataset):
    '''
    Joint DB of Casia, Insta Dataset

    Casia with no age labels
    directory structure
        root/person_name/{age}_filenum.jpg
    AGE DB with actual labels
    directory structure
        root/person_name/filenum_{age}.jpg
    Insta DB with no labels
    directory structure
        root/person_name/filenum.png

    Store image directories at init phase

    Returns image, label, age
    '''
    def __init__(self, casia_imgs_folder, train_transforms):

        self.casia_imgs_folder_name = casia_imgs_folder.split('/')[-1]

        self.transform = train_transforms

        self.casia_class_list = os.listdir(casia_imgs_folder)
        self.casia_class_num = len(os.listdir(casia_imgs_folder))

        self.class_num = self.casia_class_num

        total_list = []
        for (dirpath, _, filenames) in os.walk(casia_imgs_folder):
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
        if dataset_name == self.casia_imgs_folder_name:
            label = self.casia_class_list.index(folder_name)
            # age = int(file_name.split('_')[0]) # this is actually meaningless
            age = 1
        else:
            print('Something went wrong. What have you done!')
            assert False

        if self.transform is not None:
            img = self.transform(img)

        return img, label, age

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
    def __init__(self, casia_imgs_folder, babymonster_imgs_folder, train_transforms):

        self.casia_imgs_folder_name = casia_imgs_folder.split('/')[-1]
        self.babymonster_imgs_folder_name = babymonster_imgs_folder.split('/')[-1]

        self.transform = train_transforms

        self.casia_class_list = os.listdir(casia_imgs_folder)
        self.casia_class_num = len(os.listdir(casia_imgs_folder))
        self.babymonster_class_list = os.listdir(babymonster_imgs_folder)
        self.babymonster_class_num = len(os.listdir(babymonster_imgs_folder))

        self.class_num = self.casia_class_num + self.babymonster_class_num

        total_list = []
        for (dirpath, _, filenames) in os.walk(casia_imgs_folder):
            total_list += [os.path.join(dirpath, file) for file in filenames]
        for (dirpath, _, filenames) in os.walk(babymonster_imgs_folder):
            # NOTE : random suffle and reduce images to n
            id_list = [os.path.join(dirpath, file) for file in filenames]
            # if id_list is not []:
            #     random.shuffle(id_list)
            #     id_list = id_list[:20]
            total_list += id_list

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
        file_name = img_path_list[-1] # {age}_filenum.jpg
        folder_name = img_path_list[-2]# label

        img = Image.open(img_path)
        if dataset_name == self.agedb_imgs_folder_name:
            label = self.agedb_class_list.index(folder_name) + self.vgg_class_num
            _age = int(file_name.split('_')[-1].strip('.jpg'))
            age = 0 if _age < 13 else 1
        elif dataset_name == self.insta_imgs_folder_name:
            label = self.insta_class_list.index(folder_name) + self.vgg_class_num + self.agedb_class_num
            age = 0 # also meaningless
        elif dataset_name == self.vgg_imgs_folder_name:
            label = int(folder_name) 
            # age = int(file_name.split('_')[0]) # this is actually meaningless
            age = 1
        else:
            print('Something went wrong. What have you done!')
            assert False

        if self.transform is not None:
            img = self.transform(img)

        return img, label, age

class CasiaAgeDBInstaDataset(Dataset):
    '''
    Joint DB of Casia, AgeDB, Insta Dataset

    Casia with no age labels
    directory structure
        root/person_name/{age}_filenum.jpg
    AGE DB with actual labels
    directory structure
        root/person_name/filenum_{age}.jpg
    Insta DB with no labels
    directory structure
        root/person_name/filenum.png

    Store image directories at init phase

    Returns image, label, age
    '''
    def __init__(self, casia_imgs_folder, agedb_imgs_folder, insta_imgs_folder, train_transforms):

        self.casia_imgs_folder_name = casia_imgs_folder.split('/')[-1]
        self.agedb_imgs_folder_name = agedb_imgs_folder.split('/')[-1]
        self.insta_imgs_folder_name = insta_imgs_folder.split('/')[-1]

        self.transform = train_transforms

        self.casia_class_list = os.listdir(casia_imgs_folder)
        self.casia_class_num = len(os.listdir(casia_imgs_folder))

        self.agedb_class_list = os.listdir(agedb_imgs_folder)
        self.agedb_class_num = len(self.agedb_class_list)

        self.insta_class_list = os.listdir(insta_imgs_folder)
        self.insta_class_num = len(self.insta_class_list)

        self.class_num = self.casia_class_num + self.agedb_class_num + self.insta_class_num

        total_list = []
        for (dirpath, _, filenames) in os.walk(casia_imgs_folder):
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
        file_name = img_path_list[-1] # {age}_filenum.jpg
        folder_name = img_path_list[-2]# label

        img = Image.open(img_path)
        if dataset_name == self.agedb_imgs_folder_name:
            label = self.agedb_class_list.index(folder_name) + self.casia_class_num
            _age = int(file_name.split('_')[-1].strip('.jpg'))
            age = 0 if _age < 13 else 1
        elif dataset_name == self.insta_imgs_folder_name:
            label = self.insta_class_list.index(folder_name) + self.casia_class_num + self.agedb_class_num
            age = 0 # also meaningless
        elif dataset_name == self.casia_imgs_folder_name:
            label = self.casia_class_list.index(folder_name)
            # age = int(file_name.split('_')[0]) # this is actually meaningless
            age = 1
        else:
            print('Something went wrong. What have you done!')
            assert False

        if self.transform is not None:
            img = self.transform(img)

        return img, label, age


class CasiaInstaDataset(Dataset):
    '''
    Joint DB of Casia, Insta Dataset

    Casia with no age labels
    directory structure
        root/person_name/{age}_filenum.jpg
    AGE DB with actual labels
    directory structure
        root/person_name/filenum_{age}.jpg
    Insta DB with no labels
    directory structure
        root/person_name/filenum.png

    Store image directories at init phase

    Returns image, label, age
    '''
    def __init__(self, casia_imgs_folder, insta_imgs_folder, train_transforms):

        self.casia_imgs_folder_name = casia_imgs_folder.split('/')[-1]
        self.insta_imgs_folder_name = insta_imgs_folder.split('/')[-1]

        self.transform = train_transforms

        self.casia_class_list = os.listdir(casia_imgs_folder)
        self.casia_class_num = len(os.listdir(casia_imgs_folder))

        self.insta_class_list = os.listdir(insta_imgs_folder)
        self.insta_class_num = len(self.insta_class_list)

        self.class_num = self.casia_class_num + self.insta_class_num

        total_list = []
        for (dirpath, _, filenames) in os.walk(casia_imgs_folder):
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
        file_name = img_path_list[-1] # {age}_filenum.jpg
        folder_name = img_path_list[-2]# label

        img = Image.open(img_path)
        if dataset_name == self.insta_imgs_folder_name:
            label = self.insta_class_list.index(folder_name) + self.casia_class_num
            age = 0 # also meaningless
        elif dataset_name == self.casia_imgs_folder_name:
            label = self.casia_class_list.index(folder_name)
            # age = int(file_name.split('_')[0]) # this is actually meaningless
            age = 1
        else:
            print('Something went wrong. What have you done!')
            assert False

        if self.transform is not None:
            img = self.transform(img)

        return img, label, age

class CasiaAgeDBDataset(Dataset):
    '''
    Joint DB of Casia, AgeDB, Insta Dataset

    Casia with no age labels
    directory structure
        root/person_name/{age}_filenum.jpg
    AGE DB with actual labels
    directory structure
        root/person_name/filenum_{age}.jpg
    Insta DB with no labels
    directory structure
        root/person_name/filenum.png

    Store image directories at init phase

    Returns image, label, age
    '''
    def __init__(self, casia_imgs_folder, agedb_imgs_folder, train_transforms):

        self.casia_imgs_folder_name = casia_imgs_folder.split('/')[-1]
        self.agedb_imgs_folder_name = agedb_imgs_folder.split('/')[-1]

        self.transform = train_transforms

        self.casia_class_list = os.listdir(casia_imgs_folder)
        self.casia_class_num = len(os.listdir(casia_imgs_folder))

        self.agedb_class_list = os.listdir(agedb_imgs_folder)
        self.agedb_class_num = len(self.agedb_class_list)

        self.class_num = self.casia_class_num + self.agedb_class_num

        total_list = []
        for (dirpath, _, filenames) in os.walk(casia_imgs_folder):
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
            label = self.agedb_class_list.index(folder_name) + self.casia_class_num
            _age = int(file_name.split('_')[-1].strip('.jpg'))
            age = 0 if _age < 13 else 1
        elif dataset_name == self.casia_imgs_folder_name:
            label = self.casia_class_list.index(folder_name)
            # age = int(file_name.split('_')[0]) # this is actually meaningless
            age = 1
        else:
            print('Something went wrong. What have you done!')
            assert False

        if self.transform is not None:
            img = self.transform(img)

        return img, label, age

class ChildDataset(Dataset):
    '''
    Dataset for child images, returns (img, 0)
    root should be child_folder
    '''
    def __init__(self, root, transform):
        self.root_list = [os.path.join(root, img_name) for img_name in os.listdir(root)]
        self.transform = transform
        
    def __len__(self):
        return len(self.root_list)

    def __getitem__(self, item):
        img_path = self.root_list[item]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, 0


class AdultDataset(Dataset):
    '''
    Dataset for adult images, returns (img, 1)
    root should be adult folder
    '''
    def __init__(self, root, transform):
        self.root_list = [os.path.join(root, img_name) for img_name in os.listdir(root)]
        self.transform = transform

    def __len__(self):
        return len(self.root_list)

    def __getitem__(self, item):
        img_path = self.root_list[item]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, 1


class FGNetDataset(Dataset):
    def __init__(self, root, transform):
        super(FGNetDataset).__init__()
        self.img_list = [os.path.join(root, img_name) for img_name in os.listdir(root)]
        self.transform = transform

    def __getitem__(self, item):
        img_path = self.img_list[item]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        age = int(img_path.split("/")[-1].split("A")[-1][:2])
        if age < 13:                                                    # XXX : set child age here
            return img, 0
        else:
            return img, 1

    def __len__(self):
        return len(self.img_list)


class CustomDataset(Dataset):
    '''
    Use multiple datasets with age info at end of filename
    '''
    def __init__(self, root, transform, *args):
        super(CustomDataset).__init__()
        data_list = args[0]
        root_list = []

        self.transform = transform
        self.img_list = []
        self.class_list = []

        for data in data_list:
            root_list.append(os.path.join(root, data))

        for root_path in root_list:
            self.img_list += glob.glob(os.path.join(root_path, "*/**"))

        for root_path in root_list:
            self.class_list += os.listdir(root_path)

        print(len(self.img_list))
        print(len(self.class_list))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_path = self.img_list[item]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        cls = self.class_list.index(img_path.split("/")[-2])
        if int(img_path.split("/")[-1].split("_")[-1].split(".")[0]) < 13: # XXX : Set child age here
            d_cls = 0
        else:
            d_cls = 1
        return img, cls, d_cls


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
