#! /usr/bin/env python

import os
import argparse
import json
import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
import numpy as np


def _main_(args):
    config_path  = args.conf
    input_path   = args.input
    output_path_wm  = args.output1
    output_path_nw = args.output2
    txt_path = args.file_path

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    makedirs(output_path_wm)
    makedies(output_path_nw)

    ###############################
    #   Set some parameter
    ###############################       
    # a multiple of 32, the smaller the faster
    net_h = config['pred']['net_h']
    net_w = config['pred']['net_w']
    obj_thresh = config['pred']['obj_thresh']
    nms_thresh = config['pred']['nms_thresh']

    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'])


    ###############################
    #   Predict bounding boxes 
    ###############################

    # do detection on an image or a set of images
    image_paths = []

    if os.path.isdir(input_path):
        for inp_file in os.listdir(input_path):
            image_paths += [input_path + inp_file]
    else:
        image_paths += [input_path]

    image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

    # the main loop
    Total = 0
    count_wm = 0
    for image_path in image_paths:
        Total = Total + 1
        image = cv2.imread(image_path)
        print(image_path)

        # predict the bounding boxes
        boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]

        # draw bounding boxes on the image using labels
        _, flag, labs_str = draw_boxes(image, boxes, config['model']['labels'], obj_thresh)
        if flag:
            count_wm = count_wm + 1
            cv2.imwrite(output_path_wm + image_path.split('/')[-1], np.uint8(image))
            f = open(txt_path, 'a')
            f.write(image_path.split('/')[-1]+'     '+labs_str + '\n')
            f.close()
        # write the image with bounding boxes to file
        if flag == False:
            cv2.imwrite(output_path_nw + image_path.split('/')[-1], np.uint8(image))
            f = open(txt_path,'a')
            f.write(image_path.split('/')[-1]+'     '+labs_str + '\n')
            f.close()
        print("labs_str:"+labs_str)
    print("\n total images number is %d, the watermark image number is %d" % (Total, count_wm))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')    
    argparser.add_argument('-w', '--output1', default='output/', help='path to output directory of watermakr images')
    argparser.add_argument('-n', '--output2', default='non_output/', help = 'path to output directory of non-watermark images')
    argparser.add_argument('-f', '--file_path', default='wm.txt', help='txt file to save images name and their '
                                                                  'corresponding watermark')
    
    args = argparser.parse_args()
    _main_(args)
