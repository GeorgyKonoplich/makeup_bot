import cv2
import numpy as np
import dlib
from skimage.draw import circle, ellipse
import math


SOURCE_DIR = '/home/georgy.konoplich/workspace/'
PREDICTOR_PATH = SOURCE_DIR + 'Data/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def get_landmarks(im):
    rects = detector(im, 1)
    return np.array([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def get_circle_image(point, r, image):
    img_copy = image.copy() / 1.0
    rr, cc = circle(point[1], point[0], r, image.shape)
    zero = 0.00001
    img_copy[rr, cc, :] = (zero, zero, zero)
    mask = np.all(img_copy == [zero, zero, zero], axis=2).astype(int)
    new_image = image * mask[:, :, np.newaxis]

    rgb_mean = np.mean(image[np.all(img_copy == [zero, zero, zero], axis=2)], axis=0)
    vis_ind = np.argwhere(new_image > 0)
    vis_min = np.min(vis_ind, 0)
    vis_max = np.max(vis_ind, 0)
    edge = (int(vis_max[1] - vis_min[1]), int(vis_max[0] - vis_min[0]))
    image_face_crop = np.copy(new_image[vis_min[0]:(vis_min[0] + edge[1] + 1), vis_min[1]:(vis_min[1] + edge[0] + 1)])
    limit = 0
    image_face_crop[np.all(image_face_crop == [limit, limit, limit], axis=2)] = rgb_mean
    return image_face_crop, mask


def get_ellipse_image(point, r, image):
    img_copy = image.copy() / 1.0
    rr, cc = ellipse(point[1], point[0], int(r/2), r, image.shape)
    zero = 0.00001
    img_copy[rr, cc, :] = (zero, zero, zero)
    mask = np.all(img_copy == [zero, zero, zero], axis=2).astype(int)
    new_image = image * mask[:, :, np.newaxis]

    rgb_mean = np.mean(image[np.all(img_copy == [zero, zero, zero], axis=2)], axis=0)
    vis_ind = np.argwhere(new_image > 0)
    vis_min = np.min(vis_ind, 0)
    vis_max = np.max(vis_ind, 0)
    edge = (int(vis_max[1] - vis_min[1]), int(vis_max[0] - vis_min[0]))
    image_face_crop = np.copy(new_image[vis_min[0]:(vis_min[0] + edge[1]+1), vis_min[1]:(vis_min[1] + edge[0]+1)])
    limit = 0
    image_face_crop[np.all(image_face_crop == [limit, limit, limit], axis=2)] = rgb_mean
    return image_face_crop, mask


def get_left_eye(img_A):
    lands = get_landmarks(img_A.astype(np.uint8))
    p1 = lands[36]
    p2 = lands[39]
    p3 = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    image_eye_A, mask_A = get_circle_image(p3, int(dist + dist / 3), img_A)
    return image_eye_A, mask_A


def get_right_eye(img_A):
    lands = get_landmarks(img_A.astype(np.uint8))
    p1 = lands[42]
    p2 = lands[45]
    p3 = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    image_eye_A, mask_A = get_circle_image(p3, int(dist + dist / 3), img_A)
    return np.flip(image_eye_A, axis=1), mask_A


def get_mouth(img):
    lands = get_landmarks(img)
    p1 = lands[48]
    p2 = lands[54]
    p3 = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    #image_face_crop, mask = get_circle_image(p3, int(dist / 1.5), img)
    image_face_crop, mask = get_ellipse_image(p3, int(dist / 1.5), img)
    return image_face_crop, mask


def clone(mask, img, new_mouth):
    vis_ind = np.argwhere(mask > 0)
    vis_min = np.min(vis_ind, 0)
    vis_max = np.max(vis_ind, 0)
    edge = (int(vis_max[1] - vis_min[1]), int(vis_max[0] - vis_min[0]))
    center = (int((vis_min[1] + vis_max[1]) / 2 + 0.5), int((vis_min[0] + vis_max[0]) / 2 + 0.5))
    img_new = np.zeros(img.shape)
    img_new[(vis_min[0]):(vis_min[0] + edge[1] + 1), (vis_min[1]):(vis_min[1] + edge[0] + 1)] = new_mouth

    img_new = (mask.astype(np.uint8) == 0)[:, :, np.newaxis] * img + mask[:, :, np.newaxis].astype(np.uint8) * img_new
    output = cv2.seamlessClone(img_new.astype(np.uint8), img.astype(np.uint8),
                               (mask * 255).astype(np.uint8), center, cv2.NORMAL_CLONE)
    return output
