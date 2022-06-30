import os

import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_folders(root):
    """Find classes by folders under a root.

    Args:
        root (string): root directory of folders

    Returns:
        folder_to_idx (dict): the map from folder name to class idx
    """
    folders = [
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ]
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folder_to_idx


def get_samples(rootA,rootB, folder_to_idx, extensions):
    """Make dataset by walking all images under a root.

    Args:
        root (string): root directory of folders
        folder_to_idx (dict): the map from class name to class idx
        extensions (tuple): allowed extensions

    Returns:
        samples (list): a list of tuple where each element is (image, label)
    """
    samples = []
    root = os.path.expanduser(rootA)
    for folder_name in sorted(os.listdir(root)):
        _dir = os.path.join(root, folder_name)
        if not os.path.isdir(_dir):
            continue

        for _, _, fns in sorted(os.walk(_dir)):
            for fn in sorted(fns):
                if has_file_allowed_extension(fn, extensions):
                    path = os.path.join(folder_name, fn)
                    #print('-------------')
                    #print(path)
                    item = (path, folder_to_idx[folder_name]) #(image_path,class_id)
                    samples.append(item)
    return samples


@DATASETS.register_module()
class MS(BaseDataset):
    """`ImageNet <http://www.image-net.org>`_ Dataset.

    This implementation is modified from
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/imagenet.py  # noqa: E501
    """

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    #CLASSES = ['1','2','3','4','5']
    CLASSES = ['0','1','2','3','4','5','6','7','8','9']

    def load_annotations(self):
        #import pdb;pdb.set_trace()
        self.data_EO_prefix = self.data_prefix+'EO/'
        self.data_SAR_prefix = self.data_prefix+'SAR/'


        folder_to_idx = find_folders(self.data_EO_prefix)
        samples = get_samples(
            self.data_EO_prefix,
            self.data_SAR_prefix,
            folder_to_idx,
            extensions=self.IMG_EXTENSIONS)
        if len(samples) == 0:
            raise (RuntimeError('Found 0 files in subfolders of: '
                                f'{self.data_SAR_prefix}. '
                                'Supported extensions are: '
                                f'{",".join(self.IMG_EXTENSIONS)}'))

        self.folder_to_idx = folder_to_idx

        self.samples = samples

        data_infos = []
        for filename_EO, gt_label in self.samples:
            info = {'img_prefix': self.data_prefix}
            class_id = filename_EO.split('/')[0]
            image_id = filename_EO.split('_')[1].split('.')[0]
            extense = filename_EO.split('.')[1]
            info['img_info'] = {'filename_EO': filename_EO,'filename_SAR': class_id + '/'+'SAR_'+image_id+ '.'+extense}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        


        return data_infos
