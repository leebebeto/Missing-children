import glob
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


output_dir = 'age_distribution'
os.makedirs(output_dir, exist_ok=True)
dataset = 'CASIA'

# # 1. Simple histogram of age distribution
#
# num_bin = 80
# age_list = glob.glob('./dataset/CASIA_112/*/*')
# age_list = [int(image_path.split('/')[-1].split('_')[-1][:-4]) for image_path in age_list]
#
# # matplotlib histogram
# plt.hist(np.array(age_list), color = 'blue', edgecolor = 'black',
#          bins = num_bin)
#
# # Add labels
# plt.title(f'Histogram of {dataset} age distribution')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.savefig(f'{output_dir}/{dataset}_age_histogram_{num_bin}.png')

############################################################################################################
# # 2. Number of child age bins
# age_list = glob.glob('./dataset/CASIA_112/*/*')
# age_list = [int(image_path.split('/')[-1].split('_')[-1][:-4]) for image_path in age_list if int(image_path.split('/')[-1].split('_')[-1][:-4]) <= 20]
#
# num_bin = np.array(age_list).max()
#
# # matplotlib histogram
# plt.hist(np.array(age_list), color = 'blue', edgecolor = 'black',
#          bins = num_bin)
#
# # Add labels
# plt.title(f'Histogram of {dataset} age distribution')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.savefig(f'{output_dir}/{dataset}_only_child_age_histogram_{num_bin}.png')

############################################################################################################
# 3. Histogram of paired number of child and adults
identity_list = [i for i in range(10572)]
child_identity = set()

domain_dict = {0: 0, 1: 0}
domain_list = []
for identity in identity_list:
    age_list = glob.glob(f'./dataset/CASIA_112/{identity}/*')
    for image_path in age_list:
        age = int(image_path.split('/')[-1].split('_')[-1][:-4])
        if age <=18:
            child_identity.add(identity)

child_identity = list(child_identity)
print(f'# of childs: {len(child_identity)}')

for child in child_identity:
    age_list = glob.glob(f'./dataset/CASIA_112/{child}/*')
    for image_path in age_list:
        age = int(image_path.split('/')[-1].split('_')[-1][:-4])
        if age <=18:
            domain_list.append(0)
            domain_dict[0] += 1
        else:
            domain_list.append(1)
            domain_dict[1] += 1

print(f'# of child images: {domain_dict[0]} || # of paired images: {domain_dict[0] + domain_dict[1]}')
ratio = domain_dict[0] / (domain_dict[0] + domain_dict[1])
# matplotlib histogram
# print(list(domain_dict.values()))
# import pdb; pdb.set_trace()
plt.hist(np.array(domain_list), color = 'blue', edgecolor = 'black',
         bins = 2)

# Add labels
plt.title(f'Histogram of {dataset} age distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig(f'{output_dir}/{dataset}_paired_histogram_{len(child_identity)}_{ratio:.3f}.png')
