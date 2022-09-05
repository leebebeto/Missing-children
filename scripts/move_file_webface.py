import os
import glob
import tqdm
import time

os.system('mkdir /home/jungsoolee/webface')

for i in tqdm.tqdm(range(30000)):
    i = '0' * (7-len(str(i))) + str(i)
    os.system(f'cp -r /home/nas1_userB/dataset/WebFace42M/img_folder/0_0_{i} /home/jungsoolee/webface/')
    time.sleep(0.5)

