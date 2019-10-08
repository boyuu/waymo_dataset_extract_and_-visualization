import os
import tensorflow as tf
import math
import numpy as np
import itertools
import waymo_open_dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


def show_camera_image(camera_image, camera_labels, output, file,no):
  img = np.array(tf.image.decode_jpeg(camera_image.image))
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

  # Draw the camera labels.
  for camera_label in camera_labels:
    # Ignore camera labels that do not correspond to this camera.
    if camera_label.name != camera_image.name:
      continue
    buffer = '{}.jpg '.format(no)
    # Iterate over the individual labels.
    for label in camera_label.labels:
        x = int(float(label.box.center_x))
        y = int(float(label.box.center_y))
        width = int(float(label.box.width))
        length = int(float(label.box.length))
        xl = x-0.5*length
        yl = y-0.5*width
        xd = x+0.5*length
        yd = y+0.5*width
        type = int(float(label.type))
        buffer += '{},{},{},{},{} '.format(xl,yl,xd,yd,type-1)
        # cv2.rectangle(img,(x-0.5*width,y-0.5*length),(x+0.5*width,y+0.5*length),(0,0,255),2)
    buffer += '\n'
  # Show the camera image.
  if int(camera_image.name) == 1:
      path = output+'/'+file.split('.')[0].strip()+'/front/'
      if not os.path.exists(path):
          os.makedirs(path)
      with open(path+'train.txt','a') as writer:
          writer.writelines(buffer)
      cv2.imwrite(path+'{}.jpg'.format(no),img)
  elif int(camera_image.name) == 2:
      path = output+'/'+file.split('.')[0].strip()+'/front_left/'
      if not os.path.exists(path):
          os.makedirs(path)
      with open(path+'train.txt','a') as writer:
          writer.writelines(buffer)
      cv2.imwrite(path+'{}.jpg'.format(no),img)
  elif int(camera_image.name) == 3:
      path = output+'/'+file.split('.')[0].strip()+'/front_right/'
      if not os.path.exists(path):
          os.makedirs(path)
      with open(path+'train.txt','a') as writer:
          writer.writelines(buffer)
      cv2.imwrite(path+'{}.jpg'.format(no),img)
  elif int(camera_image.name) == 4:
      path = output+'/'+file.split('.')[0].strip()+'/side_left/'
      if not os.path.exists(path):
          os.makedirs(path)
      with open(path+'train.txt','a') as writer:
          writer.writelines(buffer)
      cv2.imwrite(path+'{}.jpg'.format(no),img)
  elif int(camera_image.name) == 5:
      path = output+'/'+file.split('.')[0].strip()+'/side_right/'
      if not os.path.exists(path):
          os.makedirs(path)
      with open(path+'train.txt','a') as writer:
          writer.writelines(buffer)
      cv2.imwrite(path+'{}.jpg'.format(no),img)
  # cv2.imshow('img',img)
  # cv2.waitKey(0)

def extract(file, filepath, output,cur_num,total):
    dataset = tf.data.TFRecordDataset(filepath, compression_type='')
    print('currently processing No.{} segment file:{}, ...'.format(cur_num,file))
    no = 1
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        for index, image in enumerate(frame.images):
            show_camera_image(image, frame.camera_labels,output,file,no)
        print('{}/{} {}frames finished'.format(cur_num,total,no))
        no += 1

#waymo dataset segment file location path
filelistpath = '/home/boyu/Downloads'
#output to the specific folder
output = '/home/boyu/Documents/waymo_open_dataset'

filelist_raw = os.popen('cd {}; ls'.format(filelistpath))
filelist_raw = filelist_raw.read()
filelist_raw = filelist_raw.split()
filelist = []
for file in filelist_raw:
    if os.path.exists(output+'/'+file.split('.')[0].strip()):
        continue
    if not file.endswith('.tfrecord'):
        continue
    filelist.append(file)
total = len(filelist)
cur_num = 1
print('totally {} file to extract, processing...'.format(total))
#error segment-15578655130939579324_620_000_640_000_with_camera_labels
#error segment-2739239662326039445_5890_320_5910_320_with_camera_labels
#error segment-3002379261592154728_2256_691_2276_691_with_camera_labels
for file in filelist:
    filepath = filelistpath+'/'+file
    extract(file,filepath,output,cur_num,total)
    cur_num += 1