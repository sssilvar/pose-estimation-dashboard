import os
from os.path import join, dirname, realpath, isfile

import uuid
import time
import cv2
import numpy as np

ROOT = dirname(realpath(__file__))
TEMP = join(ROOT, 'tmp')
os.makedirs(TEMP, exist_ok=True)

inWidth = 368
inHeight = 368
threshold = 0.1


def model_files(mode='COCO'):
    """
    Returns the model files for pose estimation depending on the desired mode.
    :param str mode: Model used to estimate pose it can be 'COCO' or 'MPI' (default='MPI')
    :return tuple: A tuple containing the files and params of the model
    """
    if mode is "COCO":
        protoFile = join(ROOT, "models/coco/pose_deploy_linevec.prototxt")
        weightsFile = join(ROOT, "models/coco/pose_iter_440000.caffemodel")
        nPoints = 18
        POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                      [11, 12], [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

    elif mode is "MPI":
        protoFile = join(ROOT, "models/mpi/pose_deploy_linevec_faster_4_stages.prototxt")
        weightsFile = join(ROOT, "models/mpi/pose_iter_160000.caffemodel")
        nPoints = 15
        POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10],
                      [14, 11], [11, 12], [12, 13]]
    else:
        raise KeyError(f'Mode {mode} was not found. Please use "COCO" or "MPI"')
    return protoFile, weightsFile, nPoints, POSE_PAIRS


def estimate_pose_video(input_source, mode='COCO', st_bar=None, frame_holder=None, info_holder=None):
    if hasattr(input_source, 'read'):
        "It means is a file-like object IOBase"
        filename = join(TEMP, f'input.vid')
        with open(filename, 'wb') as f:
            f.write(input_source.getbuffer())
        input_source = filename

    # Load Video
    cap = cv2.VideoCapture(input_source)
    hasFrame, frame = cap.read()
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Load model params
    protoFile, weightsFile, nPoints, POSE_PAIRS = model_files(mode)

    out_file = join(TEMP, f'output.mkv')
    vid_writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 10,
                                 (frame.shape[1], frame.shape[0]))

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    # Output params
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    point_size = int(min(width, height) // 90)

    while cv2.waitKey(1) < 0:
        t = time.time()
        hasFrame, frame = cap.read()
        frameCopy = np.copy(frame)
        if not hasFrame:
            cv2.waitKey()
            break

        progress = cap.get(cv2.CAP_PROP_POS_AVI_RATIO)
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if st_bar:
            st_bar.progress(progress)
        if info_holder:
            info_holder.markdown(f'### Processing frame {current_frame} of {n_frames} {point_size}')

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                        (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()

        H = output.shape[2]
        W = output.shape[3]
        # Empty list to store the detected keypoints
        points = []

        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > threshold:
                cv2.circle(frame, (int(x), int(y)), point_size, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else:
                points.append(None)

        # Draw Skeleton
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), point_size // 2, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[partA], point_size, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], point_size, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8,
                    (255, 50, 0), 2, lineType=cv2.LINE_AA)
        # cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        # cv2.imshow('Output-Keypoints', frameCopy)
        # cv2.imshow('Output-Skeleton', cv2.resize(frame, (0, 0), fx=0.25, fy=0.25))
        if frame_holder:
            frame_holder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

        vid_writer.write(frame)

    vid_writer.release()

