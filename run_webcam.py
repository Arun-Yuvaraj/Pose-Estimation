import argparse
import logging
import time
import os
import json
import jsonstreams
import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import pickle

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

start = time.time()


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=str, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--save_video', type=bool, default=False,
                        help='to write output video. dafult name file_name_output.avi')

    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)

    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str(args.tensorrt))
    logger.debug('cam read+')

    cam = cv2.VideoCapture(args.camera)
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # scale = 40
    # new_width = int((width * scale) / 100)
    new_width = 500
    # new_height = int((height * scale) / 100)
    new_height = 500
    frames_per_second = cam.get(cv2.CAP_PROP_FPS)
    output_file = cv2.VideoWriter(
        filename="/content/drive/My Drive/output1.avi",
        # some installation of opencv may not support x264 (due to its license),
        # you can try other format (e.g. MPEG)
        fourcc=cv2.VideoWriter_fourcc(*"mpeg"),
        fps=float(frames_per_second),
        frameSize=(width, height),
        isColor=True,
    )

    ret_val, image = cam.read()

    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    j = 0
    k = 1
    # data = {}
    while True:
        h = {}
        # d = {}
        # data[k] = {}
        # data[k]['Person'] = []

        ret_val, image = cam.read()

        j = j + 1
        frame_start = time.time()
        if image is None:
            j = j - 1
            break
        r_width = width / new_width
        r_height = height / new_height

        image_new = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        image_new = cv2.GaussianBlur(image_new, (3, 3), 0)

        logger.debug('image process+')
        humans = e.inference(image_new, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        logger.debug('postprocess+')
        images = TfPoseEstimator.draw_humans(image_new, humans, imgcopy=False)

        i = 1
        m = len(humans)
        for human in humans:
            # print(type(human))

            d = {}
            # data[k]['Person ' + str(i)] = []
            h['Person ' + str(i)] = []
            try:
                for l in range(14):
                    x = 0
                    y = 0
                    a = human.body_parts[l]
                    # print(a.x)
                    x = int(a.x * images.shape[1] * r_width)
                    y = int(a.y * images.shape[0] * r_height)
                    # print(x)
                    cv2.circle(image, (x, y), 10, (0, 0, 255), -1)
                    if (l == 0):
                        d['x_head'] = x
                        d['y_head'] = y
                    elif (l == 1):
                        d['x_neck'] = x
                        d['y_neck'] = y
                    elif (l == 2):
                        d['x_r_shoulder'] = x
                        d['y_r_shoulder'] = y
                    elif (l == 3):
                        d['x_r_elbow'] = x
                        d['y_r_elbow'] = y
                    elif (l == 4):
                        d['x_r_wrist'] = x
                        d['y_r_wrist'] = y
                    elif (l == 5):
                        d['x_l_shoulder'] = x
                        d['y_l_shoulder'] = y
                    elif (l == 6):
                        d['x_l_elbow'] = x
                        d['y_l_elbow'] = y
                    elif (l == 7):
                        d['x_l_wrist'] = x
                        d['y_l_wrist'] = y
                    elif (l == 8):
                        d['x_r_hip'] = x
                        d['y_r_hip'] = y
                    elif (l == 9):
                        d['x_r_knee'] = x
                        d['y_r_knee'] = y
                    elif (l == 10):
                        d['x_r_foot'] = x
                        d['y_r_foot'] = y
                    elif (l == 11):
                        d['x_l_hip'] = x
                        d['y_l_hip'] = y
                    elif (l == 12):
                        d['x_l_knee'] = x
                        d['y_l_knee'] = y
                    elif (l == 13):
                        d['x_l_foot'] = x
                        d['y_l_foot'] = y

            except:
                pass
            # data[k]['Person ' + str(i)].append(d)
            h['Person ' + str(i)].append(d)
            if (i == m):
                os.chdir('/content/drive/My Drive/Pose-Estimation/json')
                with jsonstreams.Stream(jsonstreams.Type.object, filename='foo' + str(k)) as s:
                    s.write(str(k), h)
                    # print(data)
            i = i + 1

        k = k + 1
        logger.debug('show+')

        frame_end = time.time()
        output_file.write(image)
        print(round(frame_end - frame_start))

        # cv2.imshow('tf-pose-estimation result', image)
        cv2.waitKey(1)
        logger.debug('finished+')

    # with open('/content/drive/My Drive/data.txt', 'w') as outfile:
    #    json.dump(data, outfile, indent=4)

    # print(j)
    # end = time.time()
    # print(round(end - start))

    cv2.destroyAllWindows()
