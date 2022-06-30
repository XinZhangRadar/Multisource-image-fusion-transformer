meta_keys = ('filename_SAR', 'filename_VIS', 'ori_shape_VIS', 'ori_shape_SAR',
             'img_shape_VIS', 'img_shape_SAR', 'pad_shape_VIS',
             'pad_shape_SAR', 'scale_factor', 'flip', 'flip_direction',
             'img_norm_cfg_VIS', 'img_norm_cfg_SAR')
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='MS_EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        in_channels=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        if_trans=False,
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='MSFT',
        image_size=112,
        patch_size=2,
        channels=2048,
        dim=256,
        depth=4,
        heads=4,
        mlp_dim=1024,
        dropout=0.1),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='Mul_SpaceNet6',
        data_root='data/spacenet6/',
        img_dir='SAR',
        ann_dir='labels',
        split='trainval_SAR.txt',
        pipeline=[
            dict(type='LoadMultiImageFromFile', imdecode_backend='tifffile'),
            dict(type='LoadAnnotations', reduce_edge=True),
            dict(type='Resize_Mul', img_scale=(896, 896)),
            dict(
                type='RandomCrop_Mul',
                crop_size=(896, 896),
                cat_max_ratio=0.75),
            dict(type='RandomFlip_Mul', prob=0.5),
            dict(
                type='Normalize_Mul',
                mean_VIS=[46.57639976, 49.26095826, 48.07817453, 61.06509706],
                std_VIS=[58.50214967, 60.02363556, 60.07681265, 71.15278445],
                to_rgb=True,
                mean_SAR=[67.84898429, 81.193908, 77.99933941, 68.5571387],
                std_SAR=[67.44875732, 75.29750527, 73.2490039, 68.11481076]),
            dict(type='Pad_Mul', size=(896, 896), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle_Mul'),
            dict(
                type='Collect',
                keys=['img_SAR', 'img_VIS', 'gt_semantic_seg'],
                meta_keys=('filename_SAR', 'filename_VIS', 'ori_shape_VIS',
                           'ori_shape_SAR', 'img_shape_VIS', 'img_shape_SAR',
                           'pad_shape_VIS', 'pad_shape_SAR', 'scale_factor',
                           'flip', 'flip_direction', 'img_norm_cfg_VIS',
                           'img_norm_cfg_SAR'))
        ]),
    val=dict(
        type='Mul_SpaceNet6',
        data_root='data/spacenet6/',
        img_dir='SAR',
        ann_dir='labels',
        split='trainval_SAR.txt',
        pipeline=[
            dict(type='LoadMultiImageFromFile', imdecode_backend='tifffile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(896, 896),
                flip=False,
                transforms=[
                    dict(type='Resize_Mul', keep_ratio=True),
                    dict(type='RandomFlip_Mul'),
                    dict(
                        type='Normalize_Mul',
                        mean_VIS=[
                            46.57639976, 49.26095826, 48.07817453, 61.06509706
                        ],
                        std_VIS=[
                            58.50214967, 60.02363556, 60.07681265, 71.15278445
                        ],
                        to_rgb=True,
                        mean_SAR=[
                            67.84898429, 81.193908, 77.99933941, 68.5571387
                        ],
                        std_SAR=[
                            67.44875732, 75.29750527, 73.2490039, 68.11481076
                        ]),
                    dict(type='ImageToTensor', keys=['img_SAR', 'img_VIS']),
                    dict(
                        type='Collect',
                        keys=['img_SAR', 'img_VIS'],
                        meta_keys=('filename_SAR', 'filename_VIS',
                                   'ori_shape_VIS', 'ori_shape_SAR',
                                   'img_shape_VIS', 'img_shape_SAR',
                                   'pad_shape_VIS', 'pad_shape_SAR',
                                   'scale_factor', 'flip', 'flip_direction',
                                   'img_norm_cfg_VIS', 'img_norm_cfg_SAR'))
                ])
        ]),
    test=dict(
        type='Mul_SpaceNet6',
        data_root='data/spacenet6/',
        img_dir='SAR',
        ann_dir='labels',
        split='test_SAR.txt',
        pipeline=[
            dict(type='LoadMultiImageFromFile', imdecode_backend='tifffile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(896, 896),
                flip=False,
                transforms=[
                    dict(type='Resize_Mul', keep_ratio=True),
                    dict(type='RandomFlip_Mul'),
                    dict(
                        type='Normalize_Mul',
                        mean_VIS=[
                            46.57639976, 49.26095826, 48.07817453, 61.06509706
                        ],
                        std_VIS=[
                            58.50214967, 60.02363556, 60.07681265, 71.15278445
                        ],
                        to_rgb=True,
                        mean_SAR=[
                            67.84898429, 81.193908, 77.99933941, 68.5571387
                        ],
                        std_SAR=[
                            67.44875732, 75.29750527, 73.2490039, 68.11481076
                        ]),
                    dict(type='ImageToTensor', keys=['img_SAR', 'img_VIS']),
                    dict(
                        type='Collect',
                        keys=['img_SAR', 'img_VIS'],
                        meta_keys=('filename_SAR', 'filename_VIS',
                                   'ori_shape_VIS', 'ori_shape_SAR',
                                   'img_shape_VIS', 'img_shape_SAR',
                                   'pad_shape_VIS', 'pad_shape_SAR',
                                   'scale_factor', 'flip', 'flip_direction',
                                   'img_norm_cfg_VIS', 'img_norm_cfg_SAR'))
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=42800)
checkpoint_config = dict(by_epoch=False, interval=400)
evaluation = dict(interval=400, metric='mIoU')
work_dir = './work_dirs/SAR_MS_building/MsIFT/Deeplabv3plus/'
gpu_ids = range(0, 1)
