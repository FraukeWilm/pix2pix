from data.base_dataset import BaseDataset, get_transform
from data.slide_container import SlideContainer
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import os


class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        A, B = opt.name.split("_")
        self.dir_A = os.path.join(opt.dataroot, B)
        self.A_paths = sorted(make_dataset(self.dir_A, opt.isTrain, opt.max_dataset_size))[:1]

        self.slides = [SlideContainer(path_A, None, ds_factor = opt.down_factor, patch_size = opt.crop_size) for path_A in self.A_paths]
        self.indices = [slide.get_all_patch_coordinates() for slide in self.slides]
        self.acc_num_patches = np.add.accumulate([len(idx[0]) for idx in self.indices])

        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        # read a image given a random integer index
        slide_idx = np.where(index<self.acc_num_patches)[0][0]
        slide =  self.slides[slide_idx]
        patch_idx = index - self.acc_num_patches[slide_idx - 1] if slide_idx!= 0 else index
        x,y = [idx[patch_idx] for idx in self.indices[slide_idx]]
        A_patch = slide.get_patch(x, y)
        A_patch = self.transform(Image.fromarray(A_patch))
        #return {'A': A_patch, 'A_paths': self.A_paths[slide_idx]}
        return {'A': A_patch, 'A_paths': self.A_paths[slide_idx], 'x': x, 'y': y}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.acc_num_patches[-1]
