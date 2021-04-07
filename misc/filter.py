import glob
import os
import shutil
import tqdm

id_list = [i for i in range(10572)]
# id_list = glob.glob('/home/nas1_userE/jungsoolee/Face_dataset/CASIA_REAL_NATIONAL/*')
# new_list = glob.glob('/home/nas1_userE/jungsoolee/C3AE/CASIA_112/*')
original_list = glob.glob('/home/nas1_userE/jungsoolee/Face_dataset/CASIA_REAL_NATIONAL/*/*')
new_list = glob.glob('/home/nas1_userE/jungsoolee/C3AE/CASIA_112/*/*')
print('finished loading....')
print(f'original length: {len(original_list)}')
print(f'new length: {len(new_list)}')

pbar = tqdm.tqdm(total=len(original_list))

new_path_list = [new.split('/')[-2] + '/' + new.split('/')[-1].split('_')[0]  for new in new_list]

for original in original_list:
    pbar.update(1)
    original_path = '/'.join(original.split('/')[-2:])[:-4]
    if original_path not in new_path_list:
        shutil.copy(f'/home/nas1_userE/jungsoolee/Face_dataset/CASIA_REAL_NATIONAL/{original_path}.jpg', f'/home/nas1_userE/jungsoolee/C3AE/CASIA_112/{original_path}_no_30.jpg')
# for i, id in enumerate(id_list):
#     pbar.update(1)
#     original_image_list = glob.glob(f'/home/nas1_userE/jungsoolee/Face_dataset/CASIA_REAL_NATIONAL/{id}/*')
#     original_index_list = [path.split('/')[-1][:-4] for path in original_image_list]
#
#     new_image_list = glob.glob(f'/home/nas1_userE/jungsoolee/Face_dataset/C3AE/CASIA_112/{id}/*')
#     new_index_list = [path.split('/')[-1].split('_')[0][:-4] for path in original_image_list]
#
#     if len(original_index_list) != len(new_index_list):
#         import pdb; pdb.set_trace()
#     # print(len(original_index_list), len(new_index_list))
#
#     for original_index in original_index_list:
#         if original_index not in new_index_list:
#             print('no face detected ...')
#             print(original_index)
#             import pdb; pdb.set_trace()
#             shutil.copy(f'/home/nas1_userE/jungsoolee/Face_dataset/CASIA_REAL_NATIONAL/{id}/{original_index}.jpg', f'/home/nas1_userE/jungsoolee/Face_dataset/C3AE/CASIA_112/{id}/{original_index}_no_30.jpg')




# output_dir = 'CASIA_CHILD'
# os.makedirs(output_dir, exist_ok=True)
# for i, image_path in enumerate(child_list):
#     if i%100 == 0: print(i/len(child_list))
#     age = int(image_path.split('/')[-1].split('_')[-1][:-4])
#     if age <= 18:
#         print(image_path)
#         shutil.copy(image_path, output_dir + "/" + image_path.split('/')[-2] + "_" + image_path.split('/')[-1])
