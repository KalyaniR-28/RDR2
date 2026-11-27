import os
import os.path as osp
import numpy as np
import mmengine.fileio as fileio

from mmcv.transforms import BaseTransform
from mmseg.registry import DATASETS, TRANSFORMS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class RDR2AnimalDatasetFine(BaseSegDataset):

    METAINFO = dict(
    classes=(
        'background','alligator','armadillo','badger','bat','bear','bearblack','beaver','bighornram','bluejay','boar','buck',
        'buffalo','bull','californiacondor','cardinal','carolinaparakeet','cat','cedarwaxwing','chicken',
        'chipmunk','cormorant','cougar','cow','coyote','cranewhooping','crow','deer','dogamericanfoxhound',
        'dogaustraliansheperd','dogbluetickcoonhound','dogcatahoulacur','dogchesbayretriever','dogcollie',
        'doghobo','doghound','doghusky','doglab','dogpoodle','donkey','duck','eagle','egret','elk',
        'fox','frogbull','gilamonster','goat','goosecanada','hawk','heron','horse_americanpaint',
        'horse_americanstandardbred','horse_andalusian','horse_appaloosa','horse_arabian','horse_ardennes',
        'horse_belgian','horse_dutchwarmblood','horse_hungarianhalfbred','horse_kentuckysaddle','horse_missourifoxtrotter',
        'horse_morgan','horse_mustang','horse_nokota','horse_shire','horse_suffolkpunch','horse_tennesseewalker',
        'horse_thoroughbred','horse_turkoman','horsemule','iguana','javelina','loon','moose','muskrat',
        'oriole','owl','ox','panther','parrot','pelican','pheasant','pig','pigeon','possum','prairiechicken',
        'pronghorn','quail','rabbit','raccoon','rat','raven','redfootedbooby','robin','rooster','roseatespoonbill',
        'seagull','sheep','skunk','snake','snakeblacktailrattle','snakeferdelance','snakeredboa','songbird',
        'sparrow','squirrel','toad','turkey','turtlesnapping','vulture','wolf','woodpecker'
    ),

        palette=[
    [0, 0, 0],        # background
    [63, 0, 0],       # alligator
    [255, 255, 63],   # armadillo
    [255, 191, 63],   # badger
    [127, 127, 255],  # bat
    [191, 255, 127],  # bear
    [255, 127, 191],  # bearblack
    [255, 0, 255],    # beaver
    [0, 0, 127],      # bighornram
    [127, 191, 191],  # bluejay
    [191, 191, 63],   # boar
    [127, 255, 255],  # buck
    [127, 0, 0],      # buffalo
    [191, 255, 63],   # bull
    [63, 127, 0],     # californiacondor
    [127, 0, 63],     # cardinal
    [127, 191, 255],  # carolinaparakeet
    [191, 255, 255],  # cat
    [191, 63, 0],     # cedarwaxwing
    [63, 191, 0],     # chicken
    [127, 127, 63],   # chipmunk
    [191, 191, 0],   # cormorant
    [0, 63, 255],     # cougar
    [127, 127, 0],   # cow
    [0, 191, 127],    # coyote
    [0, 0, 255],      # cranewhooping
    [127, 63, 127],   # crow
    [191, 0, 127],    # deer
    [0, 255, 255],    # dogamericanfoxhound
    [191, 255, 0],    # dogaustraliansheperd
    [191, 191, 127],  # dogbluetickcoonhound
    [63, 0, 255],     # dogcatahoulacur
    [191, 0, 63],     # dogchesbayretriever
    [63, 0, 191],     # dogcollie
    [191, 63, 191],   # doghobo
    [0, 127, 63],     # doghound
    [255, 0, 0],      # doghusky
    [63, 127, 63],    # doglab
    [127, 255, 63],   # dogpoodle
    [0, 255, 0],      # donkey
    [255, 127, 0],    # duck
    [255, 127, 127],  # eagle
    [0, 63, 63],      # egret
    [255, 63, 191],   # elk
    [191, 127, 255],  # fox
    [191, 127, 63],   # frogbull
    [0, 255, 191],    # gilamonster
    [255, 63, 0],     # goat
    [255, 191, 255],  # goosecanada
    [255, 63, 63],    # hawk
    [127, 127, 191],  # heron
    [63, 191, 255],   # horse_americanpaint
    [127, 255, 0],    # horse_americanstandardbred
    [0, 0, 63],       # horse_andalusian
    [255, 63, 255],   # horse_appaloosa
    [63, 255, 127],   # horse_arabian
    [0, 191, 63],     # horse_ardennes
    [127, 191, 63],   # horse_belgian
    [127, 127, 127],  # horse_dutchwarmblood
    [191, 63, 127],   # horse_hungarianhalfbred
    [0, 191, 0],      # horse_kentuckysaddle
    [191, 127, 191],  # horse_missourifoxtrotter
    [127, 63, 255],   # horse_morgan
    [0, 191, 191],    # horse_mustang
    [191, 0, 0],      # horse_nokota
    [191, 191, 255],  # horse_shire
    [63, 255, 0],     # horse_suffolkpunch
    [255, 255, 127],  # horse_tennesseewalker
    [191, 63, 255],   # horse_thoroughbred
    [0, 63, 127],     # horse_turkoman
    [255, 127, 255],  # horsemule
    [63, 255, 255],   # iguana
    [191, 0, 191],    # javelina
    [63, 191, 63],    # loon
    [63, 63, 63],     # moose
    [0, 255, 127],    # muskrat
    [0, 127, 127],    # oriole
    [63, 127, 191],   # owl
    [255, 0, 63],     # ox
    [63, 0, 63],      # panther
    [0, 255, 63],     # parrot
    [0, 63, 0],       # pelican
    [63, 63, 255],    # pheasant
    [63, 63, 127],    # pig
    [0, 191, 255],    # pigeon
    [127, 0, 191],    # possum
    [63, 127, 127],   # prairiechicken
    [255, 191, 0],    # pronghorn
    [127, 255, 127],  # quail
    [63, 127, 255],   # rabbit
    [0, 127, 191],    # raccoon
    [63, 255, 63],    # rat
    [127, 191, 127],  # raven
    [0, 127, 0],      # redfootedbooby
    [127, 63, 191],   # robin
    [127, 191, 0],    # rooster
    [191, 0, 255],    # roseatespoonbill
    [255, 255, 0],    # seagull
    [63, 191, 127],   # sheep
    [127, 0, 255],    # skunk
    [255, 0, 127],    # snake
    [191, 127, 127],  # snakeblacktailrattle
    [127, 63, 63],    # snakeferdelance
    [63, 63, 191],    # snakeredboa
    [63, 191, 191],   # songbird
    [0, 0, 191],      # sparrow
    [191, 191, 191],  # squirrel
    [191, 63, 63],    # toad
    [63, 0, 127],     # turkey
    [191, 255, 191],  # turtlesnapping
    [255, 0, 191],    # vulture
    [127, 255, 191],  # wolf
    [255, 191, 191]   # woodpecker
]
)

    def __init__(self, reduce_zero_label=False, **kwargs):
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=reduce_zero_label,
            **kwargs
        )
    def load_data_list(self):

        data_list = []

        img_dir = self.data_prefix['img_path']
        ann_dir = self.data_prefix['seg_map_path']

        img_names = [
            os.path.basename(p)
            for p in os.listdir(img_dir)
            if p.endswith('.png')
        ]

        for img_name in img_names:

            parts = img_name.split('_')
            scene_id = parts[1] if len(parts) > 1 else parts[0]

            # Change this suffix to match your actual coarse GT filename
            ann_name = f"mSemSegmentation_{scene_id}_fine.png"

            data_list.append(dict(
                img_path=os.path.join(img_dir, img_name),
                seg_map_path=os.path.join(ann_dir, ann_name),
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                seg_fields=[],
            ))

        return data_list

@TRANSFORMS.register_module()
class RemapFineGT(BaseTransform):

    def transform(self, results):

        mask = results['gt_seg_map']

        new_mask = np.zeros_like(mask, dtype=np.uint8)
        new_mask[mask == 255] = 0
        animal_mask = (mask >= 0) & (mask <= 111)
        new_mask[animal_mask] = mask[animal_mask] + 1

        results['gt_seg_map'] = new_mask

        return results
        
    
@DATASETS.register_module()
class PascalVOC20Dataset(BaseSegDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    METAINFO = dict(
        classes=('aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor'),
        palette=[[128, 0, 0], [0, 128, 0], [0, 0, 192],
                 [128, 128, 0], [128, 0, 128], [0, 128, 128], [192, 128, 64],
                 [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                 [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                 [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                 [0, 64, 128]])

    def __init__(self,
                 ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ann_file=ann_file,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'],
                             self.backend_args) and osp.isfile(self.ann_file)

@DATASETS.register_module()
class COCOObjectDataset(BaseSegDataset):
    """
    Implementation borrowed from TCL (https://github.com/kakaobrain/tcl) and GroupViT (https://github.com/NVlabs/GroupViT)
    COCO-Object dataset.
    1 bg class + first 80 classes from the COCO-Stuff dataset.
    """

    METAINFO = dict(
    
    classes = ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
               'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
               'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
               'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
               'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'),

    palette = [[0, 0, 0], [0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192], [0, 64, 64], [0, 192, 224],
               [0, 192, 192], [128, 192, 64], [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224], [0, 0, 64],
               [0, 160, 192], [128, 0, 96], [128, 0, 192], [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
               [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128], [64, 128, 32], [0, 160, 0], [0, 0, 0],
               [192, 128, 160], [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0], [0, 128, 0], [192, 128, 32],
               [128, 96, 128], [0, 0, 128], [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160], [0, 96, 128],
               [128, 128, 128], [64, 0, 160], [128, 224, 128], [128, 128, 64], [192, 0, 32],
               [128, 96, 0], [128, 0, 192], [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160], [64, 96, 0],
               [0, 128, 192], [0, 128, 160], [192, 224, 0], [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
               [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160], [64, 32, 128], [128, 192, 192], [0, 0, 160],
               [192, 160, 128], [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128], [64, 128, 96], [64, 160, 0],
               [0, 64, 0], [192, 128, 224], [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0]])

    def __init__(self, **kwargs):
        super(COCOObjectDataset, self).__init__(img_suffix='.jpg', seg_map_suffix='_instanceTrainIds.png', **kwargs)

@DATASETS.register_module()
class PascalContext60Dataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'aeroplane', 'bag', 'bed', 'bedclothes',
                 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle',
                 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling',
                 'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog',
                 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground',
                 'horse', 'keyboard', 'light', 'motorbike', 'mountain',
                 'mouse', 'person', 'plate', 'platform', 'pottedplant', 'road',
                 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow',
                 'sofa', 'table', 'track', 'train', 'tree', 'truck',
                 'tvmonitor', 'wall', 'water', 'window', 'wood'),
        palette=[[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]])

    def __init__(self,
                 ann_file: str,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class PascalContext59Dataset(BaseSegDataset):
    METAINFO = dict(
        classes=('aeroplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle',
                 'bird', 'boat', 'book', 'bottle', 'building', 'bus',
                 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth',
                 'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence',
                 'floor', 'flower', 'food', 'grass', 'ground', 'horse',
                 'keyboard', 'light', 'motorbike', 'mountain', 'mouse',
                 'person', 'plate', 'platform', 'pottedplant', 'road', 'rock',
                 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa',
                 'table', 'track', 'train', 'tree', 'truck', 'tvmonitor',
                 'wall', 'water', 'window', 'wood'),
        palette=[[180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
                 [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]])

    def __init__(self,
                 ann_file: str,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            reduce_zero_label=reduce_zero_label,
            **kwargs)