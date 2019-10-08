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

def extract(file, filepath, output):
    dataset = tf.data.TFRecordDataset(filepath, compression_type='')
    no = 1
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        print(no)
        for index, image in enumerate(frame.images):
            show_camera_image(image, frame.camera_labels,output,file,no)
        no += 1




    # (range_images, camera_projections,
    #  range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
    #     frame)

    # for view in frame.camera_labels:
    #     for label in view.labels:
    #         print('{},{},{},{},{}'.format(label.box.center_x,label.box.center_y,label.box.width,label.box.length,label.type))
#waymo dataset file name
file = 'segment-15533468984793020049_800_000_820_000_with_camera_labels.tfrecord'
#waymo dataset file location path
filepath = '/home/boyu/Downloads/'+file
#output to the specific folder
output = '/home/boyu/Documents/waymo_open_dataset'

extract(file,filepath,output)