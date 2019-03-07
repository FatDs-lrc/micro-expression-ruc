import dlib
import numpy as np
import imutils
from imutils import face_utils
import cv2
import os
import math
import matplotlib.pyplot as plt
import copy

path = os.path.abspath(os.path.dirname(__file__))
dlib_dat = os.path.join(path, 'shape_predictor_68_face_landmarks.dat')
cap_list = [17, 21, 22, 26, 36, 39, 27, 42, 45, 31, 35, 50, 52, 48, 54, 58, 56, 37, 38, 41, 40, 43, 44, 46, 47]
# cap_list = list(range(17, 68))
def get_sigma_by_point_num(n):
    pass

def get_dlib_facial_landmark(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlib_dat)
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    h, w, _ = image.shape
    # TODO rects->(0,0,w,h)
    if len(rects) == 0:
        rect = dlib.rectangle(-30, 10, w+10, h-10)
    else:
        rect = rects[0]

    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    # for i, (x, y) in enumerate(shape):
    #     if not any(i == x for x in cap_list):
    #         continue
    #     cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    #     cv2.putText(image, "{}".format(i), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
    # cv2.imshow("Output", image)
    # cv2.waitKey(0)
    # print(image.shape[0:2])
    return shape, image.shape[0:2]

def get_mask_one_points(point, img, sigma, num):
    # img -> ndarrays
    if num in [31, 35, 48, 54]:
        sigma *= 2
    x, y = point
    h, w = img.shape
    mask_x = np.tile(x, (h, w))
    mask_y = np.tile(y, (h, w))

    x1 = np.arange(w)
    x_map = np.tile(x1, (h, 1))

    y1 = np.arange(h)
    y_map = np.tile(y1, (w, 1))
    y_map = np.transpose(y_map)

    Gauss_map = (x_map-mask_x)**2+(y_map-mask_y)**2
    # print(Gauss_map.shape)
    # input()
    img += np.exp(-0.5*Gauss_map/sigma**2)
    return img


def make_mask_using_points(points, shape):
    h, w = shape
    Gauss_map = np.zeros((h, w))
    for i, point in enumerate(points):
        if not i in cap_list:
            continue
        # print(point)
        Gauss_map = get_mask_one_points(point, Gauss_map, 15, num=i)

    # plt.figure()
    # plt.imshow(Gauss_map, plt.cm.gray)
    # plt.show()
    for i in range(Gauss_map.shape[0]):
        for j in range(Gauss_map.shape[1]):
            Gauss_map[i,j] = min(Gauss_map[i,j], 1)
    return Gauss_map

def masked_pic(mask, img):
    image = cv2.imread(img)
    masked_img = copy.deepcopy(image)
    for i in range(3):
        masked_img[:, :, i] = image[:, :, i] * mask
    # print(masked_img.shape, image.shape)
    # plt result
    # plt.figure()
    # plt.subplot(131)
    # plt.imshow(image)
    # plt.subplot(132)
    # plt.imshow(masked_img)
    # plt.subplot(133)
    # plt.imshow(mask)
    # plt.show()
    return masked_img

def get_masked_pic(img, save_path=None):
    points, shape = get_dlib_facial_landmark(img)
    mask = make_mask_using_points(points, shape)
    masked_img = masked_pic(mask, img)
    if save_path is not None:
        cv2.imwrite(save_path, masked_img)
        print(save_path)
    return masked_img

def make_masked_pic_set(src_root, dst_root):
    length = len(src_root)
    print("dst_root:", dst_root)
    count = 0
    for root, _, files in os.walk(src_root):
        for f in files:
            if '.jpg' in f:
                print(count)
                src_path = os.path.join(root, f)
                rpath = src_path[length:]
                dst_path = dst_root + '/' + rpath
                rpath_dir = os.path.dirname(rpath)
                # print('rpath_dir:', rpath_dir)
                # print('dst_root:', dst_root)
                target_dir = dst_root + '/' + rpath_dir
                # print('target_dir:', target_dir)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                get_masked_pic(src_path, dst_path)
                count += 1




#points=get_dlib_facial_landmark('/media/liubin/9400CAB500CA9E1C/torch-project/micro_expression/casme/CASME2_RAW_selected/CASME2_RAW_selected/sub01/EP02_01f/img46.jpg')
# points, shape=get_dlib_facial_landmark('/media/liubin/9400CAB500CA9E1C/torch-project/micro_expression/dataset/raw_imgs/sub01/EP02_01f/reg_img46.jpg')
#
# # points = [[240, 320]]
# mask = make_mask_using_points(points, shape)
# masked_pic(mask, '/media/liubin/9400CAB500CA9E1C/torch-project/micro_expression/dataset/raw_imgs/sub01/EP02_01f/reg_img46.jpg')
dataset_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(os.path.dirname(os.path.dirname(dataset_dir)), 'dataset')
img_root = os.path.join(dataset_dir, 'raw_imgs')
dst_root = os.path.join(dataset_dir, 'mask_imgs_sigma15')
make_masked_pic_set(img_root, dst_root)
