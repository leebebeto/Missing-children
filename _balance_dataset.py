import os
import pdb
import shutil

source_folder = '/home/nas1_userE/Face_dataset/AgeDB_new_align'
target_folder = '/home/nas1_temp/jooyeolyun/AgeDB_balanced'

# CURRENT CODE IS ONLY WRITTEN FOR AGEDB DATASET
# CHECK 'age = split' PART FOR OTHER DIRECTORIES

for (dirpath, _, filenames) in os.walk(source_folder):
    name = dirpath.split('/')[-1]
    instance = [os.path.join(dirpath, file) for file in filenames]
    age_count = [0,0,0,0,0,0,0,0]
    for file in instance:
        age = int(file.split('_')[-1].strip('.jpg'))
        age_bin = 7
        if age < 26:
            age_bin = 0 if age < 13 else 1 if age <19 else 2
        elif age < 66:
            age_bin = (age+4)//10
        age_count[age_bin] += 1
    upper_bound = max(age_count)
    os.makedirs(target_folder+'/'+name, exist_ok= True)
    for filedir, filename in zip(instance,filenames):
        age = int(filedir.split('_')[-1].strip('.jpg'))
        age_bin = 7
        if age < 26:
            age_bin = 0 if age < 13 else 1 if age <19 else 2
        elif age < 66:
            age_bin = (age+4)//10
        for i in range(upper_bound//age_count[age_bin]):
            shutil.copy2(filedir, target_folder+'/'+name+'/'+str(i)+filename)
