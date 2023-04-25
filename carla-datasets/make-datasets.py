import glob
import os
import sys
import math
import numpy as np
import time
import cv2
from tqdm import tqdm


def funOutputImgProperties(img):
    print("properties:shape:{},size:{},dtype:{}".format(img.shape,img.size,img.dtype))


def make_mask(sem_dir, mask_dir):
    os.makedirs(mask_dir, exist_ok=True)
    sem_files = os.listdir(sem_dir)
    sembar = tqdm(sem_files, desc='make mask')
    for sem_file in sembar:
        sem_file_path = os.path.join(sem_dir, sem_file)
        if os.path.exists(sem_file_path) and os.path.isfile(sem_file_path):
            sem_image = cv2.imread(sem_file_path) # BGR
            mask = cv2.inRange(sem_image, (142, 0, 0), (142, 0, 0))
            mask_file_path = os.path.join(mask_dir, sem_file)
            cv2.imwrite(mask_file_path, mask)
        else:
            print('sem image: %s is not exist!!!', sem_file_path)
    print('make mask done!')


def mask_color_texture(texture_dir, exp_rendered_color):
    os.makedirs(texture_dir, exist_ok=True)
    for color in exp_rendered_color:
        exp_color_texture = np.zeros((720, 1080, 3), np.uint8)
        exp_color_texture[:] = color[1]
        texture_file = color[0] + '.png'
        texture_path = os.path.join(texture_dir, texture_file)
        cv2.imwrite(texture_path, exp_color_texture)
    print('make color texture done!')
 

def make_datasets(mask_dir, raw_dir, rendered_dir, texture_dir, ref_dir, ren_dir, exp_dir, exp_rendered_color):
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(ren_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)
    if os.path.exists(mask_dir) and os.path.exists(raw_dir) and os.path.exists(rendered_dir) and os.path.exists(texture_dir):
        mask_files = os.listdir(mask_dir)
        if len(mask_files) == 0:
            print('no mask image in folder: %s', raw_dir)
        else:
            maskbar = tqdm(mask_files, desc='make datasets')
            for mask_file in maskbar:
                mask_file_path = os.path.join(mask_dir, mask_file)
                mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)  # 8-bit one channel
                
                # make ref
                raw_file_path = os.path.join(raw_dir, mask_file)
                if os.path.exists(raw_file_path):
                    raw_image = cv2.imread(raw_file_path) 
                    ref_image = cv2.bitwise_and(raw_image, raw_image, mask=mask)
                    ref_image_path = os.path.join(ref_dir, mask_file)
                    cv2.imwrite(ref_image_path, ref_image)
                else:
                    print('raw image: %s is not exist!!!', raw_file_path)
               
                # make ren and exp
                prop = mask_file.split('-')
                for rendered_color in exp_rendered_color:
                    # make ren
                    rendered_file = prop[0] + '-' + rendered_color[0] + '-' + prop[2]
                    rendered_file_path = os.path.join(rendered_dir, rendered_file)
                    if os.path.exists(rendered_file_path):
                        rendered_image = cv2.imread(rendered_file_path)
                        ren_image = cv2.bitwise_and(rendered_image, rendered_image, mask=mask)
                        ren_image_path = os.path.join(ren_dir, rendered_file)
                        cv2.imwrite(ren_image_path, ren_image)
                    else:
                        print('rendered image: %s is not exist!!!', rendered_file_path)
                    # make exp
                    texture_file = rendered_color[0] + '.png'
                    texture_path = os.path.join(texture_dir, texture_file)
                    if os.path.exists(texture_path):
                        color_texture = cv2.imread(texture_path)
                        exp = cv2.bitwise_and(color_texture, color_texture, mask=mask)
                        exp_path = os.path.join(exp_dir, rendered_file)
                        cv2.imwrite(exp_path, exp)
                    else:
                        print('color texture: %s is not exist!!!', texture_path)
    else:
        print('there is no raw or mask or rendered or texture folder!!!')


# train dataset
train_raw_dir = './output/raw'
train_sem_dir = './output/converted'
train_rendered_dir = './output/rendered'

train_mask_dir = './datasets/train/mask'
train_texture_dir = './datasets/train/texture'

train_ref_dir = './datasets/train/ref'
train_exp_dir = './datasets/train/exp'
train_ren_dir = './datasets/train/ren'


# test dataset
test_raw_dir = './output/raw'
test_sem_dir = './output/converted'
test_rendered_dir = './output/rendered'

test_mask_dir = './datasets/test/mask'
test_texture_dir = './datasets/test/texture'

test_ref_dir = './datasets/test/ref'
test_exp_dir = './datasets/test/exp'
test_ren_dir = './datasets/test/ren'


# ['17,37,103', '75,86,173', '180,42,42', '0,0,0', '137,0,0'] # r,g,b
train_exp_rendered_color = [['001', [103, 37, 17]], 
                            ['002', [173, 86, 75]], 
                            ['003', [42, 42, 180]], 
                            ['004', [0, 0, 0]], 
                            ['005', [0, 0, 137]]] # [index, [b, g, r]]

test_exp_rendered_color = [['001', [225, 225, 225]],
                           ['002', [0, 255, 255]],
                           ['003', [0, 100, 0]],
                           ['004', [19, 69, 139]]]


# # make train dataset
# make_mask(train_sem_dir, train_mask_dir)
# mask_color_texture(train_texture_dir, train_exp_rendered_color)
# make_datasets(train_mask_dir, train_raw_dir, train_rendered_dir, train_texture_dir, train_ref_dir, train_ren_dir, train_exp_dir)

# make test dataset
make_mask(test_sem_dir, test_mask_dir)
mask_color_texture(test_texture_dir, test_exp_rendered_color)
make_datasets(test_mask_dir, test_raw_dir, test_rendered_dir, test_texture_dir, test_ref_dir, test_ren_dir, test_exp_dir, test_exp_rendered_color)




