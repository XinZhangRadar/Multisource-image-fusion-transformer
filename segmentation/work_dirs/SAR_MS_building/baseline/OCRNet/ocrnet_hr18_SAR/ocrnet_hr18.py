norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='CascadeEncoderDecoder',
    num_stages=2,
    pretrained='open-mmlab://msra/hrnetv2_w18',
    backbone=dict(
        type='HRNet',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        in_channels=4,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(18, 36)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144)))),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[18, 36, 72, 144],
            channels=270,
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            kernel_size=1,
            num_convs=1,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=2,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=[18, 36, 72, 144],
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            channels=512,
            ocr_channels=256,
            dropout_ratio=-1,
            num_classes=2,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='SpaceNet6',
        data_root='data/spacenet6/',
        img_dir='SAR',
        ann_dir='labels',
        split='trainval_SAR.txt',
        pipeline=[
            dict(type='LoadImageFromFile', imdecode_backend='tifffile'),
            dict(type='LoadAnnotations', reduce_edge=True),
            dict(type='Resize', img_scale=(900, 900)),
            dict(type='RandomCrop', crop_size=(900, 900), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Normalize',
                mean=[67.84898429, 81.193908, 77.99933941, 68.5571387],
                std=[67.44875732, 75.29750527, 73.2490039, 68.11481076],
                to_rgb=True),
            dict(type='Pad', size=(900, 900), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='SpaceNet6',
        data_root='data/spacenet6/',
        img_dir='SAR',
        ann_dir='labels',
        split='test_SAR.txt',
        pipeline=[
            dict(type='LoadImageFromFile', imdecode_backend='tifffile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(900, 900),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[67.84898429, 81.193908, 77.99933941, 68.5571387],
                        std=[
                            67.44875732, 75.29750527, 73.2490039, 68.11481076
                        ],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='SpaceNet6',
        data_root='data/spacenet6/',
        img_dir='SAR',
        ann_dir='labels',
        split='test_SAR.txt',
        pipeline=[
            dict(type='LoadImageFromFile', imdecode_backend='tifffile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(900, 900),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[67.84898429, 81.193908, 77.99933941, 68.5571387],
                        std=[
                            67.44875732, 75.29750527, 73.2490039, 68.11481076
                        ],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
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
work_dir = './work_dirs/baseline/OCRNet/ocrnet_hr18_SAR/'
gpu_ids = range(0, 1)
