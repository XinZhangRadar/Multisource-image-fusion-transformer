model = dict(
    type='MultiSourceFusionClassifier',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),
    neck=dict(
        type='FPN_MSFT',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4,
        image_size=56,
        patch_size=4,
        num_classes=6,
        channels=256,
        dim=256,
        depth=8,
        heads=4,
        mlp_dim=1024,
        dropout=0.1,
        loss_out=3),
    head=dict(
        type='MS_ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 2),
        loss_out=3))
dataset_type = 'VAIS'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadVAISFromFile'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadVAISFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=32),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
albu_train_transforms = [
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=1),
            dict(
                type='Cutout',
                num_holes=20,
                max_h_size=10,
                max_w_size=10,
                fill_value=128,
                p=1),
            dict(type='RandomGridShuffle', grid=(3, 3), p=1)
        ],
        p=0.5),
    dict(type='HorizontalFlip', p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='IAAAdditiveGaussianNoise', p=1.0),
            dict(type='GaussNoise', p=1.0)
        ],
        p=0.2)
]
data = dict(
    samples_per_gpu=1,#5,
    workers_per_gpu=2,
    train=dict(
        type='VAIS',
        data_prefix='/home/zhangxin/mmclassification/data/VAIS/',
        ann_file='/home/zhangxin/mmclassification/data/VAIS/annotations.txt',
        pipeline=[
            dict(type='LoadVAISFromFile'),
            dict(type='Resize', size=224),
            dict(
                type='Albu_dualflow',
                transforms=[
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='ShiftScaleRotate',
                                shift_limit=0.0625,
                                scale_limit=0.0,
                                rotate_limit=0,
                                interpolation=1,
                                p=1),
                            dict(
                                type='Cutout',
                                num_holes=20,
                                max_h_size=10,
                                max_w_size=10,
                                fill_value=128,
                                p=1),
                            dict(type='RandomGridShuffle', grid=(3, 3), p=1)
                        ],
                        p=0.5),
                    dict(type='HorizontalFlip', p=0.5),
                    dict(
                        type='RandomBrightnessContrast',
                        brightness_limit=[0.1, 0.3],
                        contrast_limit=[0.1, 0.3],
                        p=0.2),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RGBShift',
                                r_shift_limit=10,
                                g_shift_limit=10,
                                b_shift_limit=10,
                                p=1.0),
                            dict(
                                type='HueSaturationValue',
                                hue_shift_limit=20,
                                sat_shift_limit=30,
                                val_shift_limit=20,
                                p=1.0)
                        ],
                        p=0.1),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Blur', blur_limit=3, p=1.0),
                            dict(type='MedianBlur', blur_limit=3, p=1.0)
                        ],
                        p=0.1),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='IAAAdditiveGaussianNoise', p=1.0),
                            dict(type='GaussNoise', p=1.0)
                        ],
                        p=0.2)
                ],
                keymap=[dict(img_EO='image'),
                        dict(img_SAR='image')],
                update_pad_shape=False),
            dict(
                type='Normalize',
                mean_EO=[123.675, 116.28, 103.53],
                std_EO=[58.395, 57.12, 57.375],
                mean_SAR=[123.675, 116.28, 103.53],
                std_SAR=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='ImageToExpandTensor',
                keys=['img_EO', 'img_SAR'],
                size=224),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img_EO', 'img_SAR', 'gt_label'])
        ]),
    val=dict(
        type='VAIS',
        data_prefix='/home/zhangxin/mmclassification/data/VAIS/',
        ann_file='/home/zhangxin/mmclassification/data/VAIS/annotations.txt',
        pipeline=[
            dict(type='LoadVAISFromFile'),
            dict(type='Resize', size=224),
            dict(
                type='Normalize',
                mean_EO=[123.675, 116.28, 103.53],
                std_EO=[58.395, 57.12, 57.375],
                mean_SAR=[123.675, 116.28, 103.53],
                std_SAR=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='ImageToExpandTensor',
                keys=['img_EO', 'img_SAR'],
                size=224),
            dict(type='Collect', keys=['img_EO', 'img_SAR'])
        ]),
    test=dict(
        type='VAIS',
        data_prefix='/home/zhangxin/mmclassification/data/VAIS/',
        ann_file='/home/zhangxin/mmclassification/data/VAIS/annotations.txt',
        pipeline=[
            dict(type='LoadVAISFromFile'),
            dict(type='Resize', size=224),
            dict(
                type='Normalize',
                mean_EO=[123.675, 116.28, 103.53],
                std_EO=[58.395, 57.12, 57.375],
                mean_SAR=[123.675, 116.28, 103.53],
                std_SAR=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ToTensor', keys=['filename_EO']),
            dict(
                type='ImageToExpandTensor',
                keys=['img_EO', 'img_SAR'],
                size=224),
            dict(type='Collect', keys=['img_EO', 'img_SAR', 'filename_EO'])
        ]))
evaluation = dict(interval=1, metric='accuracy')
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0)
runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(interval=1)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './workdirs/multi-source/VAIS/FPN_MSFT/pre_train'
gpu_ids = range(0, 2)
