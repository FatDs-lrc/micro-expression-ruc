import os
import numpy as np
import cv2
from glob import glob
import json


Image_prefix = '../../../casme/'

def cal_grey_flow(img_dir, suffix='Cropped'):
    sub_dirs = '/'.join(img_dir.split('/')[-2:])
    grey_dir = os.path.join('../../dataset', suffix, 'grey', sub_dirs)
    flow_dir = os.path.join('../../dataset', suffix, 'optical_flow', sub_dirs)
    for _dir in (grey_dir, flow_dir):
        if not os.path.exists(_dir):
            os.makedirs(_dir)

    # print(os.path.join(Image_prefix, suffix, img_dir, '*.jpg'))
    img_seq = glob(os.path.join(Image_prefix, suffix, img_dir, '*.jpg'))
    img_seq.sort()
    flow = []
    latest = None
    for i, _path in enumerate(img_seq):
        print(_path)
        frame = cv2.imread(_path)
        cur_grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grey_path = os.path.join(grey_dir, _path.split('/')[-1])
        print(grey_path)
        cv2.imwrite(grey_path, cur_grey_frame)
        if i == 0:
            h, w = frame.shape[:2]
            flow = np.zeros((h, w, 2))
            np.save(os.path.join(flow_dir, _path.split('/')[-1].split('.')[0] + '.npy'), flow)

        elif i != 0:
            flow = compute_TVL1(latest, cur_grey_frame)
            # print(flow[..., 0].shape)
            flow_path = os.path.join(flow_dir, _path.split('/')[-1].split('.')[0] + '.npy')
            print(flow_path)
            np.save(os.path.join(flow_dir, _path.split('/')[-1].split('.')[0] + '.npy'), flow)
            # cv2.imwrite(os.path.join(flow_dir, _path.split('/')[-1].split('.')[0] + '_0.jpg'), flow[:, :, 0])
            # cv2.imwrite(os.path.join(flow_dir, _path.split('/')[-1].split('.')[0] + '_1.jpg'), flow[:, :, 1])
        print(f'frame size:{frame.shape}, flow size:{flow.shape}')
        latest = cur_grey_frame 

        # if i != 0:
        #     print(f"i:{i}")
        #     cv2.imshow("latest_grey", latest)
        #     cv2.imshow("cur_grey", cur_grey_frame)
        #     cv2.imshow("opt_flow_0", flow[..., 0])
        #     cv2.imshow("opt_flow_1", flow[..., 1])
        #     cv2.waitKey(0)
    # print(img_seq)
    # print(os.path.abspath(grey_dir))


def process_all_sub(baseinfo):
    data = json.load(open(baseinfo))
    for _d in data:
        img_dir = _d['path']
        cal_grey_flow(img_dir)

# def cal_for_frames(video_path):
#     frames = glob(os.path.join(video_path, '*.jpg'))
#     frames.sort()
#
#     flow = []
#     prev = cv2.imread(frames[0])
#     prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
#     for i, frame_curr in enumerate(frames):
#         curr = cv2.imread(frame_curr)
#         curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
#         tmp_flow = compute_TVL1(prev, curr)
#         flow.append(tmp_flow)
#         prev = curr
#
#     return flow

def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2*bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0
    return flow


# if __name__ =='__main__':
#
#     video_paths="/home/xueqian/bishe/extrat_feature/output"
#     flow_paths="/home/xueqian/bishe/extrat_feature/flow"
#     video_lengths = 109
#
#     extract_flow(video_paths, flow_paths)
process_all_sub('../auxiliary/baseinfo.json')
