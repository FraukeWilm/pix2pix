import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
from data.slide_container import SlideContainer
import yaml
import numpy as np
from pathlib import Path


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        A, B = opt.name.split("_")
        self.dir_A = os.path.join(opt.dataroot, A, 'SCC')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, B, 'SCC')

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_slides = [SlideContainer(path, down_factor = opt.down_factor, patch_size = opt.crop_size) for path in self.A_paths]
        self.B_slides = [SlideContainer(path, down_factor = opt.down_factor, patch_size = opt.crop_size) for path in self.B_paths]

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.down_factor = opt.down_factor
        self.patch_size = opt.crop_size
        with open('data/registration.yaml', "r", encoding="utf-8") as yaml_file:
            self.registration = yaml.safe_load(yaml_file)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A =  self.A_slides[index]
        B = self.B_slides[index]

        while(True):
            x,y = A.get_new_train_coordinates()
            A_patch = A.get_patch(x, y)
            # apply the same transform to both A and B
            transform_params = get_params(self.opt, A_patch.shape[:2])
            A_box = np.array([x + A.down_factor * A.patch_size // 2,y + A.down_factor * A.patch_size // 2, A.down_factor * A.patch_size, A.down_factor * A.patch_size])
            tf_param = self.registration[Path(self.A_paths[index]).stem[:6]][Path(self.B_paths[index]).stem[7:]]
            try:
                B_patch = B.get_registered_patch(tf_param_b=np.array(tf_param['b']), tf_param_t=np.array(tf_param['t']), box=A_box)
                A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
                B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

                A_patch = A_transform(Image.fromarray(A_patch))
                B_patch = B_transform(Image.fromarray(B_patch))
                return {'A': A_patch, 'B': B_patch, 'A_paths': self.A_paths[index], 'B_paths': self.B_paths[index]}
            except:
                continue

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
