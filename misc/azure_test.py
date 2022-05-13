import asyncio
import io
import glob
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO
# To install this module, run:
# python -m pip install Pillow
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, QualityForRecognition

import asyncio
import io
import glob
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, QualityForRecognition

KEY = "2075e246c42446799a7f433e1c31e8c9"
ENDPOINT = "https://practice-jungsoolee.cognitiveservices.azure.com/"

face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

data_list = glob.glob('/home/nas3_userL/jungsoolee/Face_dataset/FGNET_new_align/001A*.JPG') + \
            glob.glob('/home/nas3_userL/jungsoolee/Face_dataset/FGNET_new_align/002A*.JPG')

for image in data_list:
    gt_age = int(image.split('/')[-1].split('A')[1][:2])
    image= Image.open(image)
    image = image.resize((224, 224))
    image.save('./temp.png')
    image= open('./temp.png', 'r+b')
    faces = face_client.face.detect_with_stream(image, detection_model='detection_01', recognition_model='recognition_04', return_face_attributes=['age'])
    pred_age = faces[0].face_attributes.age
    print(f'gt age: {gt_age}, pred age: {pred_age}')
