from scipy import ndimage
from pathlib import Path
import numpy as np
import openslide
import cv2

parameters = {
    # feature extractor parameters
    "point_extractor": "orb",  # orb , sift
    "maxFeatures": 512,
    "crossCheck": False,
    "flann": False,
    "ratio": 0.6,
    "use_gray": False,

    # QTree parameter
    "homography": True,
    "filter_outliner": False,
    "debug": True,
    "target_depth": 1,
    "run_async": True,
    "num_workers": 2
}

class SlideContainer:

    def __init__(self, file_A: Path,
                 file_B: Path,
                 ds_factor: int = 1,
                 patch_size: int = 256):
        self.slide_A = openslide.open_slide(str(file_A))
        if file_B: 
            from qt_wsi_reg.registration_tree import RegistrationQuadTree
            self.qtree = RegistrationQuadTree(file_A, file_B, **parameters)
            self.slide_B = openslide.open_slide(str(file_B))
        self.patch_size = patch_size
        self.down_factor = ds_factor
        self._level = (np.abs(np.array(self.slide_A.level_downsamples) - ds_factor)).argmin()
        self.patch_size_ds = int((patch_size * self.down_factor) // 100)
        self.grayscale = cv2.cvtColor(np.array(self.slide_A.get_thumbnail((self.slide_A.dimensions[0] // 100, self.slide_A.dimensions[1] // 100))),cv2.COLOR_RGB2GRAY)
        grayscale_cropped = self.grayscale[:self.grayscale.shape[0] - self.patch_size_ds, :self.grayscale.shape[1] - self.patch_size_ds]
        self.white, ret = cv2.threshold(grayscale_cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        detection = ndimage.find_objects(ret == 0)
        self.x_min,self.x_max = 100*detection[0][1].start, 100*detection[0][1].stop
        self.y_min, self.y_max = 100*detection[0][0].start, 100*detection[0][0].stop
        self.patch_size = patch_size

    def get_patch(self, x: int = 0, y: int = 0):
        patch = np.array(self.slide_A.read_region(location=(int(x * self.down_factor), int(y * self.down_factor)),
                                               level=self._level, size=(self.patch_size, self.patch_size)))
        patch[patch[:, :, -1] == 0] = [255, 255, 255, 0]
        return patch[:,:,:3]

    def get_registered_patch(self, x, y):
        box = self.down_factor * np.array([x + self.patch_size // 2, y + self.patch_size // 2, self.patch_size, self.patch_size])
        xc, yc, w, h = self.qtree.transform_boxes([box])[0]
        level_B = (np.abs(np.array(self.slide_B.level_downsamples) - self.down_factor)).argmin()
        xmin, ymin, xmax, ymax = int(xc - w // 2), int(yc - h // 2), int(xc + w // 2), int(yc + h // 2)
        cropped_B = np.array(
            self.slide_B.read_region(location=(xmin, ymin), level=level_B, size=(int(w // self.down_factor), int(h // self.down_factor))))
        cropped_B[cropped_B[:, :, -1] == 0] = [255, 255, 255, 0]
        cropped_B = cropped_B[:,:,:3]
        w, h = int(w // self.down_factor), int(h // self.down_factor)
        if xmin < 0 or ymin < 0 or xmax > self.slide_B.dimensions[0] or ymax > self.slide_B.dimensions[1]:
            raise ValueError("Image tile out of bounds")
        M = self.qtree.get_inv_rotation_matrix
        T_c1 = np.vstack((np.array([[1, 0, -int(w // 2)], [0, 1, -int(h // 2)]]), np.array([0, 0, 1])))
        center_t = M @ [int(w // 2), int(h // 2), 1]
        T_c2 = np.vstack((np.array([[1, 0, abs(center_t[0])], [0, 1, abs(center_t[1])]]), np.array([0, 0, 1])))
        final_M = T_c2 @ (M @ T_c1)
        transformed_B = cv2.warpAffine(cropped_B, final_M[0:2, :], (w, h), borderMode=1)
        transformed_B = cv2.resize(transformed_B, (self.patch_size, self.patch_size))
        return transformed_B

    def get_new_train_coordinates(self):
        while (True):
            x, y = np.random.uniform(self.x_min, self.x_max), np.random.uniform(self.y_min, self.y_max)
            xds, yds = int(x//100), int(y//100)
            if np.sum(self.grayscale[yds:yds + self.patch_size_ds, xds:xds + self.patch_size_ds] > self.white) / (self.patch_size_ds ** 2) < 0.5:
                return int(x//self.down_factor), int(y//self.down_factor)
            
    def get_all_patch_coordinates(self):
        dimensions = self.slide_A.level_dimensions[self._level]
        x_steps, y_steps = np.indices((dimensions[0]//self.patch_size+1, dimensions[1]//self.patch_size+1))*self.patch_size
        return (x_steps.flatten(), y_steps.flatten())
        

