from .cls_head import ClsHead
from .linear_head import LinearClsHead
from .multi_label_head import MultiLabelClsHead
from .multi_label_linear_head import MultiLabelLinearClsHead
from .MS_ClsHead import MS_ClsHead
from .Proto_MS_ClsHead import Proto_MS_ClsHead
__all__ = [
    'ClsHead', 'LinearClsHead', 'MultiLabelClsHead', 'MultiLabelLinearClsHead','MS_ClsHead','Proto_MS_ClsHead'
]
