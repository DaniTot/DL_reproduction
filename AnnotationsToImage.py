import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def load_annotations(path):
    roi_rec = {}

    classes_map =  ['__background__',  # always index 0
                    'n02691156', 'n02419796', 'n02131653', 'n02834778',
                    'n01503061', 'n02924116', 'n02958343', 'n02402425',
                    'n02084071', 'n02121808', 'n02503517', 'n02118333',
                    'n02510455', 'n02342885', 'n02374451', 'n02129165',
                    'n01674464', 'n02484322', 'n03790512', 'n02324045',
                    'n02509815', 'n02411705', 'n01726692', 'n02355227',
                    'n02129604', 'n04468005', 'n01662784', 'n04530566',
                    'n02062744', 'n02391049']

    num_classes = len(classes_map)

    tree = ET.parse(path)
    size = tree.find('size')

    roi_rec['height'] = float(size.find('height').text)
    roi_rec['width'] = float(size.find('width').text)


    objs = tree.findall('object')
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)

    valid_objs = np.zeros((num_objs), dtype=np.bool)

    class_to_index = dict(zip(classes_map, range(num_classes)))

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = np.maximum(float(bbox.find('xmin').text), 0)
        y1 = np.maximum(float(bbox.find('ymin').text), 0)
        x2 = np.minimum(float(bbox.find('xmax').text), roi_rec['width'] - 1)
        y2 = np.minimum(float(bbox.find('ymax').text), roi_rec['height'] - 1)
        # if not class_to_index.has_key(obj.find('name').text):
        #     continue
        valid_objs[ix] = True
        cls = class_to_index[obj.find('name').text.lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = cls

    boxes = boxes[valid_objs, :]
    gt_classes = gt_classes[valid_objs]

    assert (boxes[:, 2] >= boxes[:, 0]).all()
    if gt_classes.size > 0:
        gt_inds = np.where(gt_classes != 0)[0]
        gt_boxes = np.empty((boxes.shape[0], 5), dtype=np.float32)
        gt_boxes[:,:4] = boxes[gt_inds, :]
        gt_boxes[:,4] = gt_classes[gt_inds]
    else:
        gt_boxes = np.empty((0,5), dtype=np.float32)

    return gt_boxes

def Picture_Annotation(path, gt_boxes):
    im = Image.open(path)

    if gt_boxes.size == 0:
        print('Borders are not defined!')
        return

    print(im)
    fig, ax = plt.subplots()
    ax.imshow(im)

    x_start = gt_boxes[0, 0]
    y_start = gt_boxes[0, 1]
    x_len = gt_boxes[0, 2] - gt_boxes[0, 0]
    y_len = gt_boxes[0, 3] - gt_boxes[0, 1]

    print(x_len)

    rect = patches.Rectangle((x_start, y_start), x_len, y_len, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    plt.show()

def MakePath():
    os.path.expanduser('~/ILSVRC2015/')

    return AnnotationPath, PicturePath
## Path --> needs to be changed.
shared_path = "/Users/ruben/Downloads/ILSVRC2015/"
AnnotationPath = shared_path + "Annotations/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00002000/000093.xml"
PicturePath = shared_path + "Data/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00002000/000093.JPEG"

gt_boxes = load_annotations(AnnotationPath)
Picture_Annotation(PicturePath, gt_boxes)

classes = ['airplane', 'antelope', 'bear', 'bicycle',
           'bird', 'bus', 'car', 'cattle',
           'dog', 'domestic_cat', 'elephant', 'fox',
           'giant_panda', 'hamster', 'horse', 'lion',
           'lizard', 'monkey', 'motorcycle', 'rabbit',
           'red_panda', 'sheep', 'snake', 'squirrel',
           'tiger', 'train', 'turtle', 'watercraft',
           'whale', 'zebra']


#print(classes[np.int(gt_boxes[0, 4] - 1)])

