import openslide
import cv2
from pathlib import Path
import numpy as np
from scipy import ndimage
from probreg import transformation as tf
from numpy.linalg import inv
import math



def get_rotation_matrix(tf_param_b):
    rotation_angle = - math.atan2(tf_param_b[0,1], tf_param_b[0,0]) * 180 / math.pi
    phi = rotation_angle * math.pi / 180
    return np.array([[np.cos(phi), - np.sin(phi), 0],
                     [np.sin(phi), np.cos(phi), 0],
                     [0., 0, 1]])

class SlideContainer:

    def __init__(self, file: Path, down_factor=4, patch_size=512):
        self.slide = openslide.open_slide(str(file))
        self._level = (np.abs(np.array(self.slide.level_downsamples) - down_factor)).argmin()
        self.down_factor = self.slide.level_downsamples[self._level]
        self.patch_size_ds = int((patch_size * self.down_factor) // 100)
        self.grayscale = cv2.cvtColor(np.array(self.slide.get_thumbnail((self.slide.dimensions[0] // 100, self.slide.dimensions[1] // 100))),cv2.COLOR_RGB2GRAY)
        grayscale_cropped = self.grayscale[:self.grayscale.shape[0] - self.patch_size_ds, :self.grayscale.shape[1] - self.patch_size_ds]
        self.white, ret = cv2.threshold(grayscale_cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        detection = ndimage.find_objects(ret == 0)
        self.x_min,self.x_max = 100*detection[0][1].start, 100*detection[0][1].stop
        self.y_min, self.y_max = 100*detection[0][0].start, 100*detection[0][0].stop
        self.patch_size = patch_size

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        self.down_factor = self.slide.level_downsamples[value]
        self._level = value

    @property
    def slide_shape(self):
        return self.slide.level_dimensions[self._level]

    def get_patch(self, x: int = 0, y: int = 0, size=None):
        if size == None:
            size = (self.patch_size, self.patch_size)
        patch = np.array(self.slide.read_region(location=(x, y),level=self._level, size=size))
        patch[patch[:, :, -1] == 0] = [255, 255, 255, 0]
        return patch[:,:,:3]

    def get_registered_patch(self, tf_param_b, tf_param_t, box):
        H = np.identity(3)
        H[:2, :2] = tf_param_b
        H[:2, 2:] = tf_param_t.reshape(2, 1)
        R_inv = inv(get_rotation_matrix(tf_param_b))
        M = H @ R_inv
        mpp_x_scale, mpp_y_scale = M[0][0], M[1][1]
        tf_temp = tf.AffineTransformation(tf_param_b, tf_param_t)
        xc, yc = tf_temp.transform(box[:2])
        w, h = box[2:] * np.array([mpp_x_scale, mpp_y_scale])
        xmin, ymin, xmax, ymax = int(xc - w // 2), int(yc - h // 2), int(xc + w // 2), int(yc + h // 2)
        cropped_B = self.get_patch(xmin, ymin, size=(int(w // self.down_factor), int(h // self.down_factor)))
        w, h = int(w // self.down_factor), int(h // self.down_factor)
        if xmin < 0 or ymin < 0 or xmax > self.slide.dimensions[0] or ymax > self.slide.dimensions[1]:
            raise ValueError("Image tile out of bounds")

        T_c1 = np.vstack((np.array([[1, 0, -int(w // 2)], [0, 1, -int(h // 2)]]), np.array([0, 0, 1])))
        center_t = R_inv @ [int(w // 2), int(h // 2), 1]
        T_c2 = np.vstack((np.array([[1, 0, abs(center_t[0])], [0, 1, abs(center_t[1])]]), np.array([0, 0, 1])))
        final_M = T_c2 @ (R_inv @ T_c1)
        transformed_B = cv2.warpAffine(cropped_B, final_M[0:2, :], (w, h), borderMode=1)
        transformed_B = cv2.resize(transformed_B, (self.patch_size, self.patch_size))
        return transformed_B

    def get_new_train_coordinates(self):
        while (True):
            x, y = np.random.uniform(self.x_min, self.x_max), np.random.uniform(self.x_min, self.x_max)
            xds, yds = int(x//100), int(y//100)
            if np.sum(self.grayscale[yds:yds + self.patch_size_ds, xds:xds + self.patch_size_ds] > self.white) / (self.patch_size_ds ** 2) < 0.5:
                return int(x), int(y)
