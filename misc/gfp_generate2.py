import glob, os
id_list = glob.glob('/home/nas3_userL/jungsoolee/FaceRecog_TestSet/img/*')
os.makedirs('gfpgan_results', exist_ok=True)

for id in id_list:
  cmd = f'python inference_gfpgan.py -i {id} -o ./gfpgan_results/{id.split("/")[-1]} -v 1.3 -s 2 --bg_upsampler realesrgan'
  os.system(cmd)
  import pdb; pdb.set_trace()