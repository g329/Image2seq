from __future__ import print_function
import skimage.io
import numpy as np
import pickle
from scipy.ndimage import zoom
from skimage.transform import resize


# -*- coding: utf-8 -*-

def rgb2bgr(img):
    return img[:, :, (2, 1, 0)]


mean = np.array([103, 116, 123]).astype(np.uint8)


def load_image(filename, color=True):
    """
    Load an image converting from grayscale or alpha as needed.
    Parameters
    ----------
    filename : string
    color : boolean
        flag for color format. True (default) loads as RGB while False
        loads as intensity (if image is already grayscale).
    Returns
    -------
    image : an image with type np.float32 in range [0, 1]
        of size (H x W x 3) in RGB or
        of size (H x W x 1) in grayscale.
    """
    img = skimage.img_as_float(skimage.io.imread(filename, as_grey=not color)).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def resize_image(im, new_dims, interp_order=1):
    """
    Resize an image array with interpolation.
    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    interp_order : interpolation order, default is linear.
    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    if im.shape[-1] == 1 or im.shape[-1] == 3:
        im_min, im_max = im.min(), im.max()
        if im_max > im_min:
            # skimage is fast but only understands {1,3} channel images
            # in [0, 1].
            im_std = (im - im_min) / (im_max - im_min)
            resized_std = resize(im_std, new_dims, order=interp_order)
            resized_im = resized_std * (im_max - im_min) + im_min
        else:
            # the image is a constant -- avoid divide by 0
            ret = np.empty((new_dims[0], new_dims[1], im.shape[-1]),
                           dtype=np.float32)
            ret.fill(im_min)
            return ret
    else:
        # ndimage interpolates anything but more slowly.
        scale = tuple(np.array(new_dims, dtype=float) / np.array(im.shape[:2]))
        resized_im = zoom(im, scale + (1,), order=interp_order)
    return resized_im.astype(np.float32)


class Normalizer(object):
    def __init__(self, size, mean_path):
        # mean sized (3,size_W,size_H) mean image pickle
        # size: H x W
        self.size = size
        if not mean_path is None:
            self.mean = np.load(mean_path)
        else:
            self.mean = None

    def __call__(self, image):
        # H x W x K
        img = image.img
        assert (img.shape[2] == 3)

        img = resize_image(img, self.size)

        img = rgb2bgr(img)
        img = img * 225

        # K x H x W
        img -= np.array([103,116,123]).astype(np.float32)
        img = img.transpose(2, 0, 1)

        # [0, 256)
        return img


class Image(object):
    def __init__(self, path):
        # H x W x K
        img = load_image(path, color=True)
        assert (len(img.shape) == 3)
        assert (img.shape[2] == 3)
        self.img = img


class ImageLoader(object):
    def __init__(self, size, mean):
        self.normalizer = Normalizer(size, mean)

    def load(self, path):
        return self.normalizer(Image(path))


if __name__ == "__main__":
    loader = ImageLoader(size=(227, 227), mean=None)
    image = loader.load("/Users/g329/workspace/deep_Learning/xrd/fetched_data/sm28180763/1p/0128.jpg")
    print(type(image), np.shape(image))
    print(image)
