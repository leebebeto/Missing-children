#!/usr/bin/env python
# coding: utf-8

# In[1]:

from __future__ import print_function

from PIL import Image
import cv2
import numpy as np
import os
from skimage import transform as trans

# In[2]:


from tqdm import tqdm

# In[3]:



import numpy as np
import torch
import torch.backends.cudnn as cudnn

from retinaface.data import cfg_mnet
from retinaface.layers.functions.prior_box import PriorBox
from retinaface.loader import load_model
from retinaface.utils.box_utils import decode, decode_landm
from retinaface.utils.nms.py_cpu_nms import py_cpu_nms
import os
from retinaface.detector import detect_faces

from tqdm import tqdm_notebook

cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# device = torch.device('cpu')
model = load_model().to(device)
model.eval()

def making_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def getCropImgSingle(rimg, lmk, src):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i

    img = cv2.warpAffine(rimg, M, (112, 112), borderValue=0.0)
    return img


# In[6]:


# Image preprocessing
def image_processing_single(img_path, save_path, img_margin=0):
    # <--left profile
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)
    src[:, 0] += 8.0
    src = np.expand_dims(src, axis=0)

    img = Image.open(img_path).convert("RGB")
    if img.size[0] > 1500 or img.size[1] > 1500:
        img = img.resize((img.size[0] // 2, img.size[1] // 2))

    img_cv2 = np.array(img)[..., ::-1]
    with torch.no_grad():
        bounding_boxes, landmarks = detect_faces(model, img, device, confidence_threshold=0.95, top_k=5000,
                                                 nms_threshold=0.4, keep_top_k=750, resize=1)
        if len(landmarks) != 0:
            dst = landmarks[0].astype(np.float32)
            facial5points = np.array([[dst[j], dst[j + 5]] for j in range(5)])

            warped = getCropImgSingle(img_cv2, facial5points, src)
            result = Image.fromarray(warped[..., ::-1])
            result.save(save_path)


# In[8]:


root_path = "/home/nas1_userE/jungsoolee/Face_dataset/LAG"
import glob
adult, child = [], []
for person_name in os.listdir(root_path):
    adult_image = glob.glob(os.path.join(root_path, person_name, '*.png'))
    child_image = glob.glob(os.path.join(root_path, person_name, 'y/*'))

    adult += adult_image
    child += child_image

def making_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


save_root = "/home/nas1_userE/jungsoolee/Face_dataset/LAG_new_align"
making_dir(save_root)
#
# for img_path in tqdm(sorted(adult)):
#     making_dir(os.path.join(save_root, img_path.split("/")[-2]))
#     ori_img_path = os.path.join(root_path, img_path)
#     save_img_path = os.path.join(save_root, '/'.join(img_path.split('/')[-2:]))
#     image_processing_single(ori_img_path, save_img_path, img_margin=0)

for img_path in tqdm(sorted(child)):
    making_dir(os.path.join(save_root, '/'.join(img_path.split('/')[-3:-1])))
    ori_img_path = os.path.join(root_path, img_path)
    save_img_path = os.path.join(save_root, '/'.join(img_path.split('/')[-3:]))
    image_processing_single(ori_img_path, save_img_path, img_margin=0)


print("Finish!")

# In[ ]:


# In[ ]:




