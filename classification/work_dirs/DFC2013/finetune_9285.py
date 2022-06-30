model = dict(
    type='MultiSourceFusionClassifier',
    backbone=[
        dict(type='LightNet', input_dim=144),
        dict(type='LightNet', input_dim=21)
    ],
    neck=dict(
        type='FPN_MSFT',
        in_channels=[16, 32, 128, 64],
        out_channels=128,
        num_outs=4,
        image_size=8,
        patch_size=1,
        num_classes=15,
        channels=128,
        dim=128,
        depth=4,
        heads=4,
        mlp_dim=256,
        dropout=0.3,
        loss_out=3),
    head=dict(
        type='MS_ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 2),
        loss_out=3))
dataset_type = 'DFC2013'
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
                p=1)
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
    samples_per_gpu=24,
    workers_per_gpu=2,
    train=dict(
        type='DFC2013',
        data_prefix='/data/zhangxin/DFC/Houston2013/',
        pipeline=[
            dict(type='LoadDFC2013FromFile'),
            dict(
                type='ImageToExpandTensor', keys=['img_EO', 'img_SAR'],
                size=7),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img_EO', 'img_SAR', 'gt_label'])
        ]),
    val=dict(
        type='DFC2013',
        data_prefix='/data/zhangxin/DFC/Houston2013/',
        pipeline=[
            dict(type='LoadDFC2013FromFile'),
            dict(
                type='ImageToExpandTensor', keys=['img_EO', 'img_SAR'],
                size=7),
            dict(type='Collect', keys=['img_EO', 'img_SAR'])
        ]),
    test=dict(
        type='DFC2013',
        data_prefix='/data/zhangxin/DFC/Houston2013/',
        pipeline=[
            dict(type='LoadDFC2013FromFile'),
            #dict(type='ToTensor', keys=['filename_EO']),
            dict(
                type='ImageToExpandTensor', keys=['img_EO', 'img_SAR'],
                size=7),
            dict(type='Collect', keys=['img_EO', 'img_SAR'])
        ]))
evaluation = dict(interval=1, metric='accuracy')
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0)
runner = dict(type='EpochBasedRunner', max_epochs=500)
checkpoint_config = dict(interval=1)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = './workdirs/multi-source/DFC2013/light_Tr/3/finetune/epoch_342_92.85.pth'
workflow = [('train', 1)]
work_dir = './workdirs/multi-source/DFC2013/light_Tr/3/finetune_92.85'
gpu_ids = range(0, 4)
