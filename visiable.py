import cv2
import numpy as np
import os
# path = '/home/boyu/Documents/fisheye_data/record_videos/Recorded_Videos/'
# fileList = ['11Sep1400_1415.mp4','11Sep1415_1430.mp4','11Sep1430_1445.mp4','11Sep1445_1500.mp4','11Sep1500_1515.mp4'
#             '3Sep1515_1530.mp4','3Sep1530_1545.mp4','3Sep1545_1600.mp4','4Sep1515_1530.mp4','4Sep1530_1545.mp4',
#             '5Sep1400_1415.mp4','5Sep1415_1430.mp4','5Sep1430_1445.mp4','5Sep1445_1500.mp4','11Sep1345_1400.mp4',]
# skip_frame = 8
# des_frame = 10
# video_FourCC = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(path+'test_low.avi', video_FourCC, 2, (720,720))
# for file in fileList:
#     capture = cv2.VideoCapture(path + file)
#     for i in range(des_frame):
#         for q in range(skip_frame-1):
#             _, _ = capture.read()
#         res, frame = capture.read()
#         if res == False:
#             continue
#         frame = cv2.resize(frame,(720,720))
#         # cv2.imshow('img',frame)
#         out.write(frame)
#         # cv2.waitKey(0)
#     capture.release()
# out.release()
path = '/home/boyu/Documents/waymo_open_dataset/segment-15533468984793020049_800_000_820_000_with_camera_labels/front'
with open('{}/train.txt'.format(path)) as reader:
    imgsList = reader.readlines()
    print(len(imgsList))
for img in imgsList:
    details = img.split()
    img_path = details[0].strip()
    img_path = path+'/'+img_path
    picture = cv2.imread(img_path)
    print(img_path)
    for i in range(len(details) - 1):
        bbox = details[i + 1].split(',')
        # x_min = int(bbox[0])
        # y_min = int(bbox[1])
        # x_max = int(bbox[2])
        # y_max = int(bbox[3])
        # tp = int(bbox[4])
        x_min = int(float(bbox[0]))
        y_min = int(float(bbox[1]))
        x_max = int(float(bbox[2]))
        y_max = int(float(bbox[3]))
        tp = int(float(bbox[4]))
        if tp == 0:
            cv2.rectangle(picture,(x_min,y_min),(x_max,y_max),(0,0,255),2)
        elif tp == 1:
            cv2.rectangle(picture,(x_min,y_min),(x_max,y_max),(255,0,0),2)
    scale = 0.6
    pic = cv2.resize(picture,(0,0),fx=scale,fy=scale)
    cv2.imshow('img',pic)
    cv2.waitKey(0)

# with open('/home/boyu/temperory_rotation/train.txt') as reader:
#     imgsList = reader.readlines()
# errer_list = []
# t = 0
# f = 0
# for line in imgsList:
#     path = line.split()[0].strip()
#     judge = os.path.exists(path)
#     if judge == True:
#         t += 1
#     if judge == False:
#         errer_list.append(path)
#         f += 1
# print('true:{}  false:{}'.format(t,f))