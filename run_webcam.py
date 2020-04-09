import argparse
import logging
import time
import os

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh



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
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    logger.debug('cam read+')

    cam = cv2.VideoCapture(args.camera)
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #scale = 40
    #new_width = int((width * scale) / 100)
    new_width = 250
    #new_height = int((height * scale) / 100)
    new_height = 250
    frames_per_second = cam.get(cv2.CAP_PROP_FPS)
    output_file = cv2.VideoWriter(
        filename="output1.avi",
        # some installation of opencv may not support x264 (due to its license),
        # you can try other format (e.g. MPEG)
        fourcc=cv2.VideoWriter_fourcc(*"mpeg"),
        fps=float(frames_per_second),
        frameSize=(width, height),
        isColor=True,
    )

    ret_val, image = cam.read()

    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    while True:
        ret_val, image = cam.read()
        frame_start = time.time()
        if image is None:
            break
        r_width = width / new_width
        r_height = height / new_height

        image_new = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        image_new = cv2.GaussianBlur(image_new, (7, 7), 0)

        logger.debug('image process+')
        humans = e.inference(image_new, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        logger.debug('postprocess+')
        images = TfPoseEstimator.draw_humans(image_new, humans, imgcopy=False)
        for human in humans:
            for i in range(len(humans)):
                try:
                    a = human.body_parts[7]  # 7 is for Left wrist
                    x = int(a.x * images.shape[1] * r_width)
                    y = int(a.y * images.shape[0] * r_height)
                    cv2.rectangle(image, (x - 75, y - 75), (x + 75, y + 75), (0, 0, 255), 2)
                    b = human.body_parts[4]  # 4 is for Right wrist
                    x1 = int(b.x * images.shape[1] * r_width)
                    y1 = int(b.y * images.shape[0] * r_height)
                    cv2.rectangle(image, (x1 - 75, y1 - 75), (x1 + 75, y1 + 75), (0, 0, 255), 2)
                except:
                    pass

        logger.debug('show+')

        frame_end = time.time()
        output_file.write(image)
        print(round(frame_end - frame_start))

        # cv2.imshow('tf-pose-estimation result', image)
        cv2.waitKey(1)
        logger.debug('finished+')
    end = time.time()
    print(round(end-start))

    cv2.destroyAllWindows()
