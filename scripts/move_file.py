import os
import glob
import tqdm
import time

for i in tqdm.tqdm(range(85742)):
    os.system(f'cp -r /home/nas4_user/jungsoolee/Face_dataset/ms1m-refined-112/ms1m/{i} /home/jungsoolee/ms1m-refined-112/ms1m/')
    time.sleep(0.5)

