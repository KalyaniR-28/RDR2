_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/maskclip/cls_fine_rdr2.txt',
)


# dataset settings
dataset_type = 'RDR2AnimalDatasetFine'
data_root =  'data/rdr2_animals'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 672), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RemapFineGT'),
    dict(type='PackSegInputs')
]

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
            seg_map_path='SemanticSegmentationMasks'),
        pipeline=test_pipeline))

