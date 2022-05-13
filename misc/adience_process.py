import os
import glob
import shutil
import tqdm
import pickle
# age distribution
# 0: 0-2 / 1: 4-6 / 2: 8-13 / 3: 15-20 / 4: 25-32 / 5: 38-43 6: 48-53/ 7: 60-

# adience = glob.glob('/home/nas1_userE/jungsoolee/Face_dataset/aligned/100003415@N08/*')
# fold_0 = open('/home/nas1_userE/jungsoolee/Face_dataset/fold_0_data.txt').readlines()
# print(fold_0[1].split('\t')[3])
# import pdb; pdb.set_trace()
# fold_1 = open('/home/nas1_userE/jungsoolee/Face_dataset/fold_1_data.txt')

''' changing our vgg name list to original vgg name list '''
# vgg_identity_list = glob.glob('/home/nas1_userD/yonggyu/Face_dataset/vgg/*')
# pbar = tqdm.tqdm(total = len(vgg_identity_list))
# for index, identity in enumerate(vgg_identity_list):
#     pbar.update(1)
#     identity = identity.split('/')[-1]
#     image_list = glob.glob(f'/home/nas1_userD/yonggyu/Face_dataset/vgg/{identity}/*')
#     os.system(f'mv /home/nas1_userD/yonggyu/Face_dataset/vgg/{identity} /home/nas1_userD/yonggyu/Face_dataset/vgg/{index}')
#
# total_dict = {}
# vgg_identity_list = [i for i in range(8631)]
# pbar = tqdm.tqdm(total = len(vgg_identity_list))
# for identity in vgg_identity_list:
#     pbar.update(1)
#     total_dict[identity] = len(glob.glob(f'/home/nas1_userD/yonggyu/Face_dataset/vgg/{identity}/*'))
#
# min_first_dict = {k: v for k, v in sorted(total_dict.items(), key=lambda item: item[1])}
# max_first_dict = {k: v for k, v in sorted(total_dict.items(), key=lambda item: item[1], reverse=True)}
#
# ''' min first / max first lists '''
# min_first_list, max_first_list = [], []
# min_total, max_total = 0, 0
# for k, v in min_first_dict.items():
#     if min_total > 13000: break
#     min_first_list.append(k)
#     min_total += v
# #
# #
# # for k, v in max_first_dict.items():
# #     if max_total > 30000: break
# #     max_first_list.append(k)
# #     max_total += v
# #
# # with open('/home/nas1_userE/jungsoolee/Face_dataset/vgg_min_first.pickle', 'wb') as f:
# #     pickle.dump(min_first_list, f)
# #
# # with open('/home/nas1_userE/jungsoolee/Face_dataset/vgg_max_first.pickle', 'wb') as f:
# #     pickle.dump(max_first_list, f)
# #
# import pdb; pdb.set_trace()
# with open('/home/nas1_userE/jungsoolee/Face_dataset/vgg_insta_similar.pickle', 'wb') as f:
#     pickle.dump(min_first_list, f)
#

# ''' CASIA '''
#
# total_dict = {}
# vgg_identity_list = [i for i in range(10575)]
# pbar = tqdm.tqdm(total = len(vgg_identity_list))
# for identity in vgg_identity_list:
#     pbar.update(1)
#     total_dict[identity] = len(glob.glob(f'/home/nas1_userE/jungsoolee/Missing-children/dataset/CASIA_112/{identity}/*'))
#
# min_first_dict = {k: v for k, v in sorted(total_dict.items(), key=lambda item: item[1])}
# max_first_dict = {k: v for k, v in sorted(total_dict.items(), key=lambda item: item[1], reverse=True)}
#
# ''' min first / max first lists '''
# min_first_list, max_first_list = [], []
# min_total, max_total = 0, 0
# for k, v in min_first_dict.items():
#     if min_total > 13000: break
#     min_first_list.append(k)
#     min_total += v
#
#
# for k, v in max_first_dict.items():
#     if max_total > 30000: break
#     max_first_list.append(k)
#     max_total += v
#
# with open('/home/nas1_userE/jungsoolee/Face_dataset/vgg_min_first.pickle', 'wb') as f:
#     pickle.dump(min_first_list, f)
#
# with open('/home/nas1_userE/jungsoolee/Face_dataset/vgg_max_first.pickle', 'wb') as f:
#     pickle.dump(max_first_list, f)
#
# import pdb; pdb.set_trace()
# with open('/home/nas1_userE/jungsoolee/Face_dataset/vgg_insta_similar.pickle', 'wb') as f:
#     pickle.dump(min_first_list, f)


''' age distribution '''
casia_file = open('/home/nas1_userE/jungsoolee/Face_dataset/casia-webface.txt').readlines()
age_list = [float(line.split(' ')[-2]) for line in casia_file]
image_list = [int(casia_file[0].split(' ')[-3].split('/')[-1][:-4]) for line in casia_file]
identity_list = [int(casia_file[0].split(' ')[-3].split('/')[-2]) for line in casia_file]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import shutil
# '''plotting'''
x = age_list
n_bins=10
plt.hist(x, bins = n_bins)
plt.xticks(np.arange(0, 100, step=2), rotation='vertical')
plt.savefig(f'age_distribution/CASIA_again_age_histogram_{n_bins}.png')
plt.close()

n_bins=20
plt.hist(x, bins = n_bins)
plt.xticks(np.arange(0, 100, step=2), rotation='vertical')
plt.savefig(f'age_distribution/CASIA_again_age_histogram_{n_bins}.png')
plt.close()

n_bins=50
plt.hist(x, bins = n_bins)
plt.xticks(np.arange(0, 100, step=2), rotation='vertical')
plt.savefig(f'age_distribution/CASIA_again_age_histogram_{n_bins}.png')
plt.close()

n_bins=100
plt.hist(x, bins = n_bins)
plt.xticks(np.arange(0, 100, step=2), rotation='vertical')
plt.savefig(f'age_distribution/CASIA_again_age_histogram_{n_bins}.png')
plt.close()
#
# output_dir = 'CASIA_CHILD'
# os.makedirs(output_dir, exist_ok=True)
# age_list = np.array(age_list)
# child_index = np.where(age_list <= 13.0)[0].tolist()
# pbar = tqdm.tqdm(total = len(child_index))
# for index in child_index:
#     image_path = casia_file[index]
#     pbar.update(1)
#     id, image = int(image_path.split(' ')[1].split('/')[1]), int(image_path.split(' ')[1].split('/')[2][:-4])
#     original_image = glob.glob(f'/home/nas1_userE/jungsoolee/Missing-children/dataset/CASIA_112/{id}/{image}_*')[-1]
#     shutil.copy(original_image, output_dir + '/' + str(id) + '_' + str(image) +'.png'  )
#

