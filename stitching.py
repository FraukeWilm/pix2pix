from data.image_folder import make_dataset
from tifffile import memmap
import imageio.v3 as iio
import numpy as np
import openslide
import shutil
import pyvips
import os



def stitch_WSI(opt, dir, slide_path):
    slide = openslide.open_slide(slide_path)
    level = (np.abs(np.array(slide.level_downsamples) - opt.down_factor)).argmin()
    dimensions = slide.level_dimensions[level]
    tiff_file = memmap(os.path.join(opt.results_dir, '{}.tif'.format(dir)), shape=(dimensions[1],dimensions[0], 3), dtype=np.uint8, bigtiff=True)
    for patch in os.listdir(os.path.join(opt.results_dir, dir)):
        image = iio.imread(os.path.join(opt.results_dir, dir, patch))
        x, y = patch.split(".")[0].split("_")[-2:]
        height, width = tiff_file[int(y):int(y)+opt.crop_size,int(x):int(x)+opt.crop_size].shape[:2]
        if height > 0 and width > 0:
            tiff_file[int(y):int(y)+opt.crop_size,int(x):int(x)+opt.crop_size] = image[:height, :width]
    tiff_file.flush()
    pyvips_image = pyvips.Image.new_from_file(os.path.join(opt.results_dir, '{}.tif'.format(dir)), access="sequential")
    pyvips_image.tiffsave(os.path.join(opt.results_dir, '{}.p.tif'.format(dir)), compression="none", tile=True, tile_width=256, tile_height=256, pyramid=True)
    shutil.rmtree(os.path.join(opt.results_dir, dir))
    
    
    
    
def stitch_all(opt):
    A, B = opt.name.split("_")
    slide_dir = os.path.join(opt.dataroot, B)
    slide_paths = sorted(make_dataset(slide_dir, opt.isTrain, opt.max_dataset_size))

    for dir in os.listdir(opt.results_dir):
        if os.path.isdir(os.path.join(opt.results_dir, dir)):
            slide_path = slide_paths[np.where(path.__contains__(dir) for path in slide_paths)[0][0]]
            stitch_WSI(opt, dir, slide_path)

