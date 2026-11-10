_base_ = ['./base_config.py']

custom_imports = dict(imports=['custom_datasets'], allow_failed_imports=False)

# 1. Base scale & bank configuration
_clip_scale = 'b'
_mb_bank = 'coco'
vfm_type = 'radiov3'
mask_generator = 'entityseg'

# 2. Model configuration
model = dict(
    type='CorrCLIPSegmentation',
    name_path='data/classes_rdr2.txt',
    model_type='ViT-B-16-quickgelu',
    clip_type='open_clip_pytorch_model.bin',
    mb_region_set='coco',
    mb_main_encoder='metaclip_b',
    mb_aux_encoder=None,
    aux_model_type=None,
    use_aux_model=False,
    use_support_embedding=True,
    vfm_type=vfm_type,
    mask_generator=mask_generator
)

# 3. Dataset & DataLoader (pure image loading)
dataset_type = 'RDR2AnimalDataset'
data_root = 'data/rdr2_animals'

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='PNG',
            seg_map_path='SemanticSegmentationMasks'
        ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(2048, 336), keep_ratio=True),
            dict(type='LoadAnnotations'),
            dict(type='InvertAndRemapGT'),
            dict(type='PackSegInputs')
        ]
    )
)

# 4. Save visualization predictions
default_hooks = dict(
    visualization=dict(
        type='SegVisualizationHook',
        draw=True,
        interval=1
    )
)

# 5. Sink evaluator that skips GT checks and dumps raw results to disk
test_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU']
)
