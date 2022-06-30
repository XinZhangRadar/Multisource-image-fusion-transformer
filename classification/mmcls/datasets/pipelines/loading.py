import os.path as osp

import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes()`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadMultiImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes()`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename_EO = osp.join(results['img_prefix'],'EO',
                                results['img_info']['filename_EO'])  
            #print(filename_EO)          
            filename_SAR = osp.join(results['img_prefix'],'SAR',
                                results['img_info']['filename_SAR'])
            #print(filename_SAR) 
        else:
            filename = results['img_info']['filename']

        img_bytes_EO = self.file_client.get(filename_EO)
        img_EO = mmcv.imfrombytes(img_bytes_EO, flag=self.color_type)

        img_bytes_SAR = self.file_client.get(filename_SAR)
        img_SAR = mmcv.imfrombytes(img_bytes_SAR, flag=self.color_type)

        if self.to_float32:
            img_EO = img_EO.astype(np.float32)
            img_SAR = img_SAR.astype(np.float32)

        results['filename_EO'] = filename_EO
        results['filename_SAR'] = filename_SAR

        results['img_EO'] = img_EO
        results['img_SAR'] = img_SAR
     

        #print(results['img_EO'])


        results['img_shape_EO'] = img_EO.shape
        results['img_shape_SAR'] = img_SAR.shape

        results['ori_shape_EO'] = img_EO.shape
        results['ori_shape_SAR'] = img_SAR.shape 

        num_channels_EO = 1 if len(img_EO.shape) < 3 else img_EO.shape[2]
        num_channels_SAR = 1 if len(img_SAR.shape) <3 else img_SAR.shape[2]


        results['img_norm_cfg_EO'] = dict(
            mean=np.zeros(num_channels_EO, dtype=np.float32),
            std=np.ones(num_channels_EO, dtype=np.float32),
            to_rgb=False)

        results['img_norm_cfg_SAR'] = dict(
            mean=np.zeros(num_channels_SAR, dtype=np.float32),
            std=np.ones(num_channels_SAR, dtype=np.float32),
            to_rgb=False)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str

@PIPELINES.register_module()
class LoadVAISFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes()`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename_EO = osp.join(results['img_prefix'],
                                results['img_info']['filename_EO'])  
            #print(filename_EO)          
            filename_SAR = osp.join(results['img_prefix'],
                                results['img_info']['filename_SAR'])
            #print(filename_SAR) 
        else:
            filename = results['img_info']['filename']

        img_bytes_EO = self.file_client.get(filename_EO)
        img_EO = mmcv.imfrombytes(img_bytes_EO, flag=self.color_type)

        img_bytes_SAR = self.file_client.get(filename_SAR)
        img_SAR = mmcv.imfrombytes(img_bytes_SAR, flag=self.color_type)

        if self.to_float32:
            img_EO = img_EO.astype(np.float32)
            img_SAR = img_SAR.astype(np.float32)

        results['filename_EO'] = filename_EO
        results['filename_SAR'] = filename_SAR

        results['img_EO'] = img_EO
        results['img_SAR'] = img_SAR
     

        #print(results['img_EO'])


        results['img_shape_EO'] = img_EO.shape
        results['img_shape_SAR'] = img_SAR.shape

        results['ori_shape_EO'] = img_EO.shape
        results['ori_shape_SAR'] = img_SAR.shape 

        num_channels_EO = 1 if len(img_EO.shape) < 3 else img_EO.shape[2]
        num_channels_SAR = 1 if len(img_SAR.shape) <3 else img_SAR.shape[2]


        results['img_norm_cfg_EO'] = dict(
            mean=np.zeros(num_channels_EO, dtype=np.float32),
            std=np.ones(num_channels_EO, dtype=np.float32),
            to_rgb=False)

        results['img_norm_cfg_SAR'] = dict(
            mean=np.zeros(num_channels_SAR, dtype=np.float32),
            std=np.ones(num_channels_SAR, dtype=np.float32),
            to_rgb=False)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadDFC2013FromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes()`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=True,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):

        img_EO = results['img_info']['filename_EO']
        img_SAR = results['img_info']['filename_SAR']

        if self.to_float32:
            img_EO = img_EO.astype(np.float32)
            img_SAR = img_SAR.astype(np.float32)

        #results['filename_EO'] = filename_EO
        #results['filename_SAR'] = filename_SAR

        results['img_EO'] = img_EO
        results['img_SAR'] = img_SAR
     

        #print(results['img_EO'])


        results['img_shape_EO'] = img_EO.shape
        results['img_shape_SAR'] = img_SAR.shape

        results['ori_shape_EO'] = img_EO.shape
        results['ori_shape_SAR'] = img_SAR.shape 

        num_channels_EO = 1 if len(img_EO.shape) < 3 else img_EO.shape[2]
        num_channels_SAR = 1 if len(img_SAR.shape) <3 else img_SAR.shape[2]


        results['img_norm_cfg_EO'] = dict(
            mean=np.zeros(num_channels_EO, dtype=np.float32),
            std=np.ones(num_channels_EO, dtype=np.float32),
            to_rgb=False)

        results['img_norm_cfg_SAR'] = dict(
            mean=np.zeros(num_channels_SAR, dtype=np.float32),
            std=np.ones(num_channels_SAR, dtype=np.float32),
            to_rgb=False)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str