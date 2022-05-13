import glob
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


output_dir = 'age_distribution'
os.makedirs(output_dir, exist_ok=True)
dataset = 'CASIA'

from collections import defaultdict
# 1. Simple histogram of age distribution
num_class = 85742
file = open('../bebeto_face_dataset/ms1m.txt').readlines()
age_dict = defaultdict(int)

for item in file:
    age = int(float(item.split(' ')[2]))
    if age < 13:
        age_dict[0] += 1
    elif age >= 13 and age < 19:
        age_dict[1] += 1
    elif age >= 19 and age < 26:
        age_dict[2] += 1
    elif age >= 26 and age < 36:
        age_dict[3] += 1
    elif age >= 36 and age < 46:
        age_dict[4] += 1
    elif age >= 46 and age < 56:
        age_dict[5] += 1
    elif age >= 56 and age < 66:
        age_dict[6] += 1
    elif age >= 66:
        age_dict[7] += 1

age_dict = dict(sorted(age_dict.items(), key = lambda x:x[0]))
import pdb; pdb.set_trace()

# matplotlib histogram
plt.hist(np.array(age_list), color = 'blue', edgecolor = 'black',
         bins = num_bin)

# Add labels
plt.title(f'Histogram of {dataset} age distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig(f'{output_dir}/{dataset}_age_histogram_{num_bin}.png')

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
identity_list = [i for i in range(num_class)]
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
