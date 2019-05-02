import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as F
import logging
from tqdm import tqdm
from typing import *

class ImagesMeanStdFinder(object):
    """  Class for find mean and std from image folder dataset """
    files: list

    def __init__(self, root: str, ext: str="jpg"):
        """
        :param root: base path from your image directory
        :param ext: jpg, pgm, png etc
        """
        self.path: Path = Path(root)
        self.files: list = sorted(list(self.path.glob("*/*."+ext)))
        assert len(self.files)!=0, "Files not found, are you sure your root path or ext is correct?"
        self.images = None
        self._mean = None
        self._std = None

    def _image_to_tensor(self, img_path: str)->torch.Tensor:
        image = Image.open(img_path)
        return F.to_tensor(image)

    def _load_all_image(self)->list:
        images = []
        progress_bar = tqdm(self.files, dynamic_ncols=True)
        for f in progress_bar:
            progress_bar .set_description(f"Load file {f}")
            images.append(self._image_to_tensor(f))
            progress_bar .refresh()
        return images

    def find_mean_std(self, verbose: bool = False)->dict:
        """
        find mean and std from path that has been supplied
        :type verbose: bool
        :return dict: dictionary with key mean and std
        """
        if verbose:
            logging.getLogger().setLevel(logging.INFO)
        logging.info(f"Preapare to load all data from {str(self.path)}, it can take a while...")
        self.images: list = self._load_all_image()

        amean: list = []
        astd: list = []
        pbar = tqdm(self.images, dynamic_ncols=True)
        for tensor_img in pbar:
            pbar.set_description(f"Calculate mean and std")
            pbar.refresh()

            dim = len(tensor_img.size())
            if dim>=3: #check RGB
                chan = tensor_img.size(0)
                amean.append([tensor_img[i].mean().item() for i in range(chan)])
                astd.append([tensor_img[i].std().item() for i in range(chan)])
            elif dim==2:
                amean.append(tensor_img.mean().item())
                astd.append(tensor_img.std().item())
            else:
                raise ValueError("Image dimension is less than 2, it means that this is not matrix of image")

        amean, astd = torch.Tensor(amean), torch.Tensor(astd)
        mean = tuple([amean[:, i].mean().item() for i in range(amean.size(-1))])
        std = tuple([astd[:, i].mean().item() for i in range(astd.size(-1))])
        self._mean = mean
        self._std = std

        logging.info(f"Calculate mean and std finished")

        return {'mean': mean, 'std': std}

    def mean(self)->tuple:
        """
        get mean that has been executed with method find_mean_std
        :return tuple
        """
        return self._mean

    def std(self):
        """
        get std that has been executed with method find_mean_std
        :return tuple
        """
        return self._std






if __name__ == '__main__':
    # logging.Logger.setLevel(20)
    root = '/data/flower_data/valid'
    ims = ImagesMeanStdFinder(root, ext="jpg")
    result = ims.find_mean_std()
    print(result)




