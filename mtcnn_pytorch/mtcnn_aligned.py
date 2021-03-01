from src import detect_faces, show_bboxes
from PIL import Image
import cv2
import numpy as np
from src.align_trans import get_reference_facial_points, warp_and_crop_face
from skimage import transform as trans

identity = 'Abel_Pacheco'
filename = f'/home/nas1_userE/Face_dataset/lfw/{identity}/{identity}_0001.jpg'
img = Image.open(filename)
img_cv2 = np.array(img)[..., ::-1]

import pdb; pdb.set_trace()
src = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041]], dtype=np.float32)

src[:, 0] *= (img.size[0] / 96)
src[:, 1] *= (img.size[1] / 112)
print(img.size)
print(src)
import pdb; pdb.set_trace()

bounding_boxes, landmarks = detect_faces(img)
dst = landmarks[0].astype(np.float32)
facial5points = [[dst[j], dst[j + 5]] for j in range(5)]

tform = trans.SimilarityTransform()
tform.estimate(np.array(facial5points), src)
M = tform.params[0:2, :]
print('M', M)
warped = cv2.warpAffine(img_cv2, M, (img.size[0], img.size[1]), borderValue=0.0)
print('warped shaped', warped.shape)
temp = Image.fromarray(warped[..., ::-1])
print('temp', temp.shape)
print('img', img)
