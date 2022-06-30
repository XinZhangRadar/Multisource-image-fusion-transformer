import copy
from abc import ABCMeta, abstractmethod

import numpy as np
from torch.utils.data import Dataset

from mmcls.models.losses import accuracy
from .pipelines import Compose

from sklearn.metrics import confusion_matrix

import os

from .builder import DATASETS
from cv2 import imread
import scipy.io as scio 
@DATASETS.register_module()
class DFC2013(Dataset, metaclass=ABCMeta):
    """Base dataset.

    Args:
        data_prefix (str): the prefix of data path
        pipeline (list): a list of dict, where each element represents
            a operation defined in `mmcls.datasets.pipelines`
        ann_file (str | None): the annotation file. When ann_file is str,
            the subclass is expected to read from the ann_file. When ann_file
            is None, the subclass is expected to read according to data_prefix
        test_mode (bool): in train mode or test mode
    """



    def __init__(self, data_prefix, pipeline, ann_file=None, test_mode=False):
        super(DFC2013, self).__init__()
        self.CLASSES = {'1':0,'2':1,'3':2,'4':3,'5':4,'6':5,'7':6,'8':7,'9':8,'10':9,'11':10,'12':11,'13':12,'14':13,'15':14}

        #self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        #import pdb;pdb.set_trace()
        if not self.test_mode:
            HSI= scio.loadmat(self.data_prefix + 'HSI_TrSet.mat')['HSI_TrSet']
            LiDAR = scio.loadmat(self.data_prefix + 'LiDAR_TrSet.mat')['LiDAR_TrSet']
            Label = scio.loadmat(self.data_prefix + 'TrLabel.mat')['TrLabel']
        else:
            HSI = scio.loadmat(self.data_prefix + 'HSI_TeSet.mat')['HSI_TeSet']
            LiDAR = scio.loadmat(self.data_prefix + 'LiDAR_TeSet.mat')['LiDAR_TeSet']
            Label = scio.loadmat(self.data_prefix + 'TeLabel.mat')['TeLabel']

        data_infos = []
        for i in range(HSI.shape[0]):
            info = {'img_prefix': self.data_prefix}
            #print(HSI[i,:].reshape(7,7,-1).shape)
            info['img_info'] = {'filename_EO': HSI[i,:].reshape(7,7,-1),'filename_SAR':LiDAR[i,:].reshape(7,7,-1)}
            info['gt_label'] = np.array(self.CLASSES[str(Label[i][0])], dtype=np.int64)
            data_infos.append(info)
            #import pdb;pdb.set_trace()
        print("data_number:")
        print(len(data_infos))

        #self.compute_mean_var(data_infos)
        #print('-________')
        #print(len(data_infos))
        return data_infos
    @property
    def class_to_idx(self):
        """Map mapping class name to class index.

        Returns:
            dict: mapping from class name to class index.
        """

        return {_class: i for i, _class in enumerate(self.CLASSES)}

    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            list[int]: categories for all images.
        """

        gt_labels = np.array([data['gt_label'] for data in self.data_infos])
        return gt_labels

    def get_cat_ids(self, idx):
        """Get category id by index.

        Args:
            idx (int): Index of data.

        Returns:
            int: Image category of specified index.
        """

        return self.data_infos[idx]['gt_label'].astype(np.int)

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        #print(results)
        return self.pipeline(results)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)

    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options={'topk': (1, 5)},
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict: evaluation results
        """
        metric_options={'topk': (1, 2)}
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['accuracy']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

        eval_results = {}
        if metric == 'accuracy':
            topk = metric_options.get('topk')
            results = np.vstack(results)
            gt_labels = self.get_gt_labels()
            #import pdb;pdb.set_trace()
            pred = results.argmax(1)
            cm = confusion_matrix(gt_labels, pred)
            print('Confusion Matrix:')
            print(cm)


            num_imgs = len(results)
            assert len(gt_labels) == num_imgs
            acc = accuracy(results, gt_labels, topk)
            eval_results = {f'top-{k}': a.item() for k, a in zip(topk, acc)}
        return eval_results
    def compute_mean_var(self,data_infos):

 

        R_channel_eo = 0
        G_channel_eo = 0
        B_channel_eo = 0
        num_eo = 0

        R_channel_sar = 0
        G_channel_sar = 0
        B_channel_sar = 0
        num_sar = 0
        for info in data_infos:
            img_eo = imread(os.path.join(info['img_prefix'], info['img_info']['filename_EO']))
            img_sar = imread(os.path.join(info['img_prefix'], info['img_info']['filename_SAR']))

            R_channel_eo = R_channel_eo + np.sum(img_eo[:, :, 0])
            G_channel_eo = G_channel_eo + np.sum(img_eo[:, :, 1])
            B_channel_eo = B_channel_eo + np.sum(img_eo[:, :, 2])
            num_eo = num_eo + img_eo.shape[0]*img_eo.shape[1]


            R_channel_sar = R_channel_sar + np.sum(img_sar[:, :, 0])
            G_channel_sar= G_channel_sar + np.sum(img_sar[:, :, 1])
            B_channel_sar= B_channel_sar + np.sum(img_sar[:, :, 2])
            num_sar = num_sar + img_sar.shape[0]*img_sar.shape[1]
         
        R_mean_sar = R_channel_sar / num_sar  
        G_mean_sar = G_channel_sar / num_sar
        B_mean_sar = B_channel_sar / num_sar

        R_mean_eo = R_channel_eo  / num_eo   # or /255.0
        G_mean_eo  = G_channel_eo  / num_eo 
        B_mean_eo  = B_channel_eo  / num_eo 
         
        R_channel_eo = 0
        G_channel_eo = 0
        B_channel_eo = 0

        R_channel_sar = 0
        G_channel_sar = 0
        B_channel_sar = 0

        for info in data_infos:
            img_eo = imread(os.path.join(info['img_prefix'], info['img_info']['filename_EO']))
            img_sar = imread(os.path.join(info['img_prefix'], info['img_info']['filename_SAR']))

            R_channel_eo = R_channel_eo + np.sum((img_eo[:, :, 0] - R_mean_eo)**2)
            G_channel_eo = G_channel_eo + np.sum((img_eo[:, :, 1] - G_mean_eo)**2)
            B_channel_eo = B_channel_eo + np.sum((img_eo[:, :, 2] - B_mean_eo)**2)


            R_channel_sar = R_channel_sar + np.sum((img_sar[:, :, 0] - R_mean_sar)**2)
            G_channel_sar= G_channel_sar + np.sum((img_sar[:, :, 1] - G_mean_sar)**2)
            B_channel_sar= B_channel_sar + np.sum((img_sar[:, :, 2] - B_mean_sar)**2)


         
        R_var_eo = np.sqrt(R_channel_eo / num_eo)
        G_var_eo = np.sqrt(G_channel_eo / num_eo)
        B_var_eo = np.sqrt(B_channel_eo / num_eo)

        R_var_sar = np.sqrt(R_channel_sar / num_sar)
        G_var_sar = np.sqrt(G_channel_sar / num_sar)
        B_var_sar = np.sqrt(B_channel_sar / num_sar)        
        print("R_mean_eo is %f, G_mean_eo is %f, B_mean_eo is %f" % (R_mean_eo, G_mean_eo, B_mean_eo))
        print("R_var_eo is %f, G_var_eo is %f, B_var_eo is %f" % (R_var_eo, G_var_eo, B_var_eo))

        print("R_mean_sar is %f, G_mean_sar is %f, B_mean_sar is %f" % (R_mean_sar, G_mean_sar, B_mean_sar))
        print("R_var_sar is %f, G_var_sar is %f, B_var_sar is %f" % (R_var_sar, G_var_sar, B_var_sar))                

