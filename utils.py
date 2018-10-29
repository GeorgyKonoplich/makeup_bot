"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import pprint
import scipy.misc
import numpy as np
import copy
try:
    _imread = scipy.misc.imread
except AttributeError:
    from imageio import imread as _imread

from skimage.transform import rotate
from skimage.draw import circle, ellipse
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
)

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for cyclegan
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image

def load_test_data(image_path, fine_size=256):
    img = imread(image_path)
    img = scipy.misc.imresize(img, [fine_size, fine_size])
    img = img/127.5 - 1
    return img


def mouth_aug(p=.5):
    return Compose([
        HorizontalFlip(p=0.5),
        OneOf([
            IAAAdditiveGaussianNoise(scale=(1, 3)),
            GaussNoise(var_limit=(1, 5)),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=.1),
            Blur(blur_limit=3, p=.1),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(alpha=(0.1, 0.2)),
            IAAEmboss(strength=(0.1, 0.3)),
            RandomContrast(limit=0.1),
            RandomBrightness(limit=0.15),
        ], p=0.3)
    ], p=p)


aug_A = HorizontalFlip(p=0.5)#mouth_aug(p=1)
aug_B = HorizontalFlip(p=0.5)


def load_train_data(image_path, load_size=256, fine_size=128, is_testing=False):
    img_A = imread(image_path[0])
    img_B = imread(image_path[1])
    img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
    img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
    img_A_r = rotate(img_A, np.random.randint(-15, 15))
    img_B_r = rotate(img_B, np.random.randint(-15, 15))

    r = int(fine_size/2)-1
    rr, cc = circle(r, r, r+1, img_A.shape)
    mask = np.zeros(img_A.shape[:2])
    mask[rr, cc] = 1

    img_A = mask[:, :, np.newaxis] * (img_A_r*255) + (mask == 0)[:, :, np.newaxis] * img_A
    img_B = mask[:, :, np.newaxis] * (img_B_r*255) + (mask == 0)[:, :, np.newaxis] * img_B

    #img_A = aug_A(image=img_A.astype(np.uint8))['image']
    #img_B = aug_B(image=img_B.astype(np.uint8))['image']

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

def load_train_data_test(image_path, fine_size=128):

    img_A = imread(image_path)
    img_orig = np.copy(img_A)
    img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
    #source_dir = '../Data/women/eyes/without_eyes/'
    #source_dir = '../Data/women/eyes/without_eyes/'
    #img_A_name = '_'.join(image_path.split('/')[-1].split('_')[1:])
    #print('img_name', img_A_name)
    #print(image_path.split('/')[-1][0])
    #if image_path.split('/')[-1][0] =='l':
    #    word = 'leftmask_'
    #else:
    #    word = 'rightmask_'

    #word = 'mouthmask_'
    #print('image_path', source_dir + word + img_A_name)
    #mask_A = imread(source_dir + word + img_A_name, is_grayscale=True) / 255.0
    #vis_ind = np.argwhere(mask_A > 0)
    #vis_min = np.min(vis_ind, 0)
    #vis_max = np.max(vis_ind, 0)
    #edgeA = (int(vis_max[1] - vis_min[1]), int(vis_max[0] - vis_min[0]))


    img_A = img_A/127.5 - 1.

    return img_A, img_orig.shape[:2], img_orig

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return _imread(path, flatten=True).astype(np.float)
    else:
        return _imread(path, mode='RGB').astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.
