import json
import multiprocessing
import os
import math
import itertools as it
from mindspore import FileWriter
import xml.etree.ElementTree as et
import cv2
import mindspore.dataset as de
import numpy as np

from models.SSD.src.model_utils.get_data_config import get_config

f = open("./config.txt")
path = f.readline().splitlines()[0]
if "ssdmobilenetv1.yaml" in path:
    config_path = r"./network/cv/SSD/config/ssd_mobilenet_v1_300_config_gpu.yaml"

elif "ssdmobilenetv1fpn.yaml" in path:
    config_path = r"./network/cv/SSD/config/ssd_mobilenet_v1_fpn_config.yaml"

elif "ssdvgg16.yaml" in path:
    config_path = r"./network/cv/SSD/config/ssd_vgg16_config.yaml"

elif "ssdresnet50fpn.yaml" in path:
    config_path = r"./network/cv/SSD/config/ssd_resnet50_fpn_config.yaml"

elif "ssdmobilenetv2.yaml" in path:
    config_path = r"./network/cv/SSD/config/ssd300_config.yaml"

config = get_config(config_path)
f.close()


class GridAnchorGenerator:
    """
    Anchor Generator
    """

    def __init__(self, image_shape, scale, scales_per_octave, aspect_ratios):
        super(GridAnchorGenerator, self).__init__()
        self.scale = scale
        self.scales_per_octave = scales_per_octave
        self.aspect_ratios = aspect_ratios
        self.image_shape = image_shape

    def generate(self, step):
        scales = np.array([2 ** (float(scale) / self.scales_per_octave)
                           for scale in range(self.scales_per_octave)]).astype(np.float32)
        aspects = np.array(list(self.aspect_ratios)).astype(np.float32)

        scales_grid, aspect_ratios_grid = np.meshgrid(scales, aspects)
        scales_grid = scales_grid.reshape([-1])
        aspect_ratios_grid = aspect_ratios_grid.reshape([-1])

        feature_size = [self.image_shape[0] / step, self.image_shape[1] / step]
        grid_height, grid_width = feature_size

        base_size = np.array([self.scale * step, self.scale * step]).astype(np.float32)
        anchor_offset = step / 2.0

        ratio_sqrt = np.sqrt(aspect_ratios_grid)
        heights = scales_grid / ratio_sqrt * base_size[0]
        widths = scales_grid * ratio_sqrt * base_size[1]

        y_centers = np.arange(grid_height).astype(np.float32)
        y_centers = y_centers * step + anchor_offset
        x_centers = np.arange(grid_width).astype(np.float32)
        x_centers = x_centers * step + anchor_offset
        x_centers, y_centers = np.meshgrid(x_centers, y_centers)

        x_centers_shape = x_centers.shape
        y_centers_shape = y_centers.shape

        widths_grid, x_centers_grid = np.meshgrid(widths, x_centers.reshape([-1]))
        heights_grid, y_centers_grid = np.meshgrid(heights, y_centers.reshape([-1]))

        x_centers_grid = x_centers_grid.reshape(*x_centers_shape, -1)
        y_centers_grid = y_centers_grid.reshape(*y_centers_shape, -1)
        widths_grid = widths_grid.reshape(-1, *x_centers_shape)
        heights_grid = heights_grid.reshape(-1, *y_centers_shape)

        bbox_centers = np.stack([y_centers_grid, x_centers_grid], axis=3)
        bbox_sizes = np.stack([heights_grid, widths_grid], axis=3)
        bbox_centers = bbox_centers.reshape([-1, 2])
        bbox_sizes = bbox_sizes.reshape([-1, 2])
        bbox_corners = np.concatenate([bbox_centers - 0.5 * bbox_sizes, bbox_centers + 0.5 * bbox_sizes], axis=1)
        self.bbox_corners = bbox_corners / np.array([*self.image_shape, *self.image_shape]).astype(np.float32)
        self.bbox_centers = np.concatenate([bbox_centers, bbox_sizes], axis=1)
        self.bbox_centers = self.bbox_centers / np.array([*self.image_shape, *self.image_shape]).astype(np.float32)

        print(self.bbox_centers.shape)
        return self.bbox_centers, self.bbox_corners

    def generate_multi_levels(self, steps):
        bbox_centers_list = []
        bbox_corners_list = []
        for step in steps:
            bbox_centers, bbox_corners = self.generate(step)
            bbox_centers_list.append(bbox_centers)
            bbox_corners_list.append(bbox_corners)

        self.bbox_centers = np.concatenate(bbox_centers_list, axis=0)
        self.bbox_corners = np.concatenate(bbox_corners_list, axis=0)
        return self.bbox_centers, self.bbox_corners


class GeneratDefaultBoxes:
    """
    Generate Default boxes for SSD, follows the order of (W, H, archor_sizes).
    `self.default_boxes` has a shape of [archor_sizes, H, W, 4], the last dimension is [y, x, h, w].
    `self.default_boxes_tlbr` has a shape as `self.default_boxes`, the last dimension is [y1, x1, y2, x2].
    """

    def __init__(self):
        fk = config.img_shape[0] / np.array(config.steps)
        scale_rate = (config.max_scale - config.min_scale) / (len(config.num_default) - 1)
        scales = [config.min_scale + scale_rate * i for i in range(len(config.num_default))] + [1.0]
        self.default_boxes = []
        for idex, feature_size in enumerate(config.feature_size):
            sk1 = scales[idex]
            sk2 = scales[idex + 1]
            sk3 = math.sqrt(sk1 * sk2)
            if idex == 0 and not config.aspect_ratios[idex]:
                w, h = sk1 * math.sqrt(2), sk1 / math.sqrt(2)
                all_sizes = [(0.1, 0.1), (w, h), (h, w)]
            else:
                all_sizes = [(sk1, sk1)]
                for aspect_ratio in config.aspect_ratios[idex]:
                    w, h = sk1 * math.sqrt(aspect_ratio), sk1 / math.sqrt(aspect_ratio)
                    all_sizes.append((w, h))
                    all_sizes.append((h, w))
                all_sizes.append((sk3, sk3))

            assert len(all_sizes) == config.num_default[idex]

            for i, j in it.product(range(feature_size), repeat=2):
                for w, h in all_sizes:
                    cx, cy = (j + 0.5) / fk[idex], (i + 0.5) / fk[idex]
                    self.default_boxes.append([cy, cx, h, w])

        def to_tlbr(cy, cx, h, w):
            return cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2

        # For IoU calculation
        self.default_boxes_tlbr = np.array(tuple(to_tlbr(*i) for i in self.default_boxes), dtype='float32')
        self.default_boxes = np.array(self.default_boxes, dtype='float32')


if hasattr(config, 'use_anchor_generator') and config.use_anchor_generator:
    generator = GridAnchorGenerator(config.img_shape, 4, 2, [1.0, 2.0, 0.5])
    default_boxes, default_boxes_tlbr = generator.generate_multi_levels(config.steps)
else:
    default_boxes_tlbr = GeneratDefaultBoxes().default_boxes_tlbr
    default_boxes = GeneratDefaultBoxes().default_boxes
y1, x1, y2, x2 = np.split(default_boxes_tlbr[:, :4], 4, axis=-1)
vol_anchors = (x2 - x1) * (y2 - y1)
matching_threshold = config.match_threshold


def ssd_bboxes_encode(boxes):
    """
    Labels anchors with ground truth inputs.

    Args:
        boxex: ground truth with shape [N, 5], for each row, it stores [y, x, h, w, cls].

    Returns:
        gt_loc: location ground truth with shape [num_anchors, 4].
        gt_label: class ground truth with shape [num_anchors, 1].
        num_matched_boxes: number of positives in an image.
    """

    def jaccard_with_anchors(bbox):
        """Compute jaccard score a box and the anchors."""
        # Intersection bbox and volume.
        ymin = np.maximum(y1, bbox[0])
        xmin = np.maximum(x1, bbox[1])
        ymax = np.minimum(y2, bbox[2])
        xmax = np.minimum(x2, bbox[3])
        w = np.maximum(xmax - xmin, 0.)
        h = np.maximum(ymax - ymin, 0.)

        # Volumes.
        inter_vol = h * w
        union_vol = vol_anchors + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) - inter_vol
        jaccard = inter_vol / union_vol
        return np.squeeze(jaccard)

    pre_scores = np.zeros((config.num_ssd_boxes), dtype=np.float32)
    t_boxes = np.zeros((config.num_ssd_boxes, 4), dtype=np.float32)
    t_label = np.zeros((config.num_ssd_boxes), dtype=np.int64)
    for bbox in boxes:
        label = int(bbox[4])
        scores = jaccard_with_anchors(bbox)
        idx = np.argmax(scores)
        scores[idx] = 2.0
        mask = (scores > matching_threshold)
        mask = mask & (scores > pre_scores)
        pre_scores = np.maximum(pre_scores, scores * mask)
        t_label = mask * label + (1 - mask) * t_label
        for i in range(4):
            t_boxes[:, i] = mask * bbox[i] + (1 - mask) * t_boxes[:, i]

    index = np.nonzero(t_label)

    # Transform to tlbr.
    bboxes = np.zeros((config.num_ssd_boxes, 4), dtype=np.float32)
    bboxes[:, [0, 1]] = (t_boxes[:, [0, 1]] + t_boxes[:, [2, 3]]) / 2
    bboxes[:, [2, 3]] = t_boxes[:, [2, 3]] - t_boxes[:, [0, 1]]

    # Encode features.
    bboxes_t = bboxes[index]
    default_boxes_t = default_boxes[index]
    bboxes_t[:, :2] = (bboxes_t[:, :2] - default_boxes_t[:, :2]) / (default_boxes_t[:, 2:] * config.prior_scaling[0])
    tmp = np.maximum(bboxes_t[:, 2:4] / default_boxes_t[:, 2:4], 0.000001)
    bboxes_t[:, 2:4] = np.log(tmp) / config.prior_scaling[1]
    bboxes[index] = bboxes_t

    num_match = np.array([len(np.nonzero(t_label)[0])], dtype=np.int32)
    return bboxes, t_label.astype(np.int32), num_match


def intersect(box_a, box_b):
    """Compute the intersect of two sets of boxes."""
    max_yx = np.minimum(box_a[:, 2:4], box_b[2:4])
    min_yx = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_yx - min_yx), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes."""
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union


def _rand(a=0., b=1.):
    """Generate random."""
    return np.random.rand() * (b - a) + a


def get_imageId_from_fileName(filename, id_iter):
    """Get imageID from fileName if fileName is int, else return id_iter."""
    filename = os.path.splitext(filename)[0]
    if filename.isdigit():
        return int(filename)
    return id_iter


def random_sample_crop(image, boxes):
    """Random Crop the image and boxes"""
    height, width, _ = image.shape
    min_iou = np.random.choice([None, 0.1, 0.3, 0.5, 0.7, 0.9])

    if min_iou is None:
        return image, boxes

    # max trails (50)
    for _ in range(50):
        image_t = image

        w = _rand(0.3, 1.0) * width
        h = _rand(0.3, 1.0) * height

        # aspect ratio constraint b/t .5 & 2
        if h / w < 0.5 or h / w > 2:
            continue

        left = _rand() * (width - w)
        top = _rand() * (height - h)

        rect = np.array([int(top), int(left), int(top + h), int(left + w)])
        overlap = jaccard_numpy(boxes, rect)

        # dropout some boxes
        drop_mask = overlap > 0
        if not drop_mask.any():
            continue

        if overlap[drop_mask].min() < min_iou and overlap[drop_mask].max() > (min_iou + 0.2):
            continue

        image_t = image_t[rect[0]:rect[2], rect[1]:rect[3], :]

        centers = (boxes[:, :2] + boxes[:, 2:4]) / 2.0

        m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
        m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

        # mask in that both m1 and m2 are true
        mask = m1 * m2 * drop_mask

        # have any valid boxes? try again if not
        if not mask.any():
            continue

        # take only matching gt boxes
        boxes_t = boxes[mask, :].copy()

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], rect[:2])
        boxes_t[:, :2] -= rect[:2]
        boxes_t[:, 2:4] = np.minimum(boxes_t[:, 2:4], rect[2:4])
        boxes_t[:, 2:4] -= rect[:2]

        return image_t, boxes_t
    return image, boxes


def preprocess_fn(img_id, image, box, is_training):
    """Preprocess function for dataset."""
    cv2.setNumThreads(2)

    def _infer_data(image, input_shape):
        img_h, img_w, _ = image.shape
        input_h, input_w = input_shape

        image = cv2.resize(image, (input_w, input_h))

        # When the channels of image is 1
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.concatenate([image, image, image], axis=-1)

        return img_id, image, np.array((img_h, img_w), np.float32)

    def _data_aug(image, box, is_training, image_size=(300, 300)):
        """Data augmentation function."""
        ih, iw, _ = image.shape
        h, w = image_size

        if not is_training:
            return _infer_data(image, image_size)

        # Random crop
        box = box.astype(np.float32)
        image, box = random_sample_crop(image, box)
        ih, iw, _ = image.shape

        # Resize image
        image = cv2.resize(image, (w, h))

        # Flip image or not
        flip = _rand() < .5
        if flip:
            image = cv2.flip(image, 1, dst=None)

        # When the channels of image is 1
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.concatenate([image, image, image], axis=-1)

        box[:, [0, 2]] = box[:, [0, 2]] / ih
        box[:, [1, 3]] = box[:, [1, 3]] / iw

        if flip:
            box[:, [1, 3]] = 1 - box[:, [3, 1]]

        box, label, num_match = ssd_bboxes_encode(box)
        return image, box, label, num_match

    return _data_aug(image, box, is_training, image_size=config.img_shape)


def create_voc_label(is_training):
    """Get image path and annotation from VOC."""
    voc_root = config.voc_root
    cls_map = {name: i for i, name in enumerate(config.classes)}
    sub_dir = 'train' if is_training else 'eval'
    voc_dir = os.path.join(voc_root, sub_dir)
    if not os.path.isdir(voc_dir):
        raise ValueError(f'Cannot find {sub_dir} dataset path.')

    image_dir = anno_dir = voc_dir
    if os.path.isdir(os.path.join(voc_dir, 'Images')):
        image_dir = os.path.join(voc_dir, 'Images')
    if os.path.isdir(os.path.join(voc_dir, 'Annotations')):
        anno_dir = os.path.join(voc_dir, 'Annotations')

    if not is_training:
        json_file = os.path.join(config.voc_root, config.voc_json)
        file_dir = os.path.split(json_file)[0]
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)
        json_dict = {"images": [], "type": "instances", "annotations": [],
                     "categories": []}
        bnd_id = 1

    image_files_dict = {}
    image_anno_dict = {}
    images = []
    id_iter = 0
    for anno_file in os.listdir(anno_dir):
        print(anno_file)
        if not anno_file.endswith('xml'):
            continue
        tree = et.parse(os.path.join(anno_dir, anno_file))
        root_node = tree.getroot()
        file_name = root_node.find('filename').text
        img_id = get_imageId_from_fileName(file_name, id_iter)
        id_iter += 1
        image_path = os.path.join(image_dir, file_name)
        print(image_path)
        if not os.path.isfile(image_path):
            print(f'Cannot find image {file_name} according to annotations.')
            continue

        labels = []
        for obj in root_node.iter('object'):
            cls_name = obj.find('name').text
            if cls_name not in cls_map:
                print(f'Label "{cls_name}" not in "{config.classes}"')
                continue
            bnd_box = obj.find('bndbox')
            x_min = int(float(bnd_box.find('xmin').text)) - 1
            y_min = int(float(bnd_box.find('ymin').text)) - 1
            x_max = int(float(bnd_box.find('xmax').text)) - 1
            y_max = int(float(bnd_box.find('ymax').text)) - 1
            labels.append([y_min, x_min, y_max, x_max, cls_map[cls_name]])

            if not is_training:
                o_width = abs(x_max - x_min)
                o_height = abs(y_max - y_min)
                ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': \
                    img_id, 'bbox': [x_min, y_min, o_width, o_height], \
                       'category_id': cls_map[cls_name], 'id': bnd_id, \
                       'ignore': 0, \
                       'segmentation': []}
                json_dict['annotations'].append(ann)
                bnd_id = bnd_id + 1

        if labels:
            images.append(img_id)
            image_files_dict[img_id] = image_path
            image_anno_dict[img_id] = np.array(labels)

        if not is_training:
            size = root_node.find("size")
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            image = {'file_name': file_name, 'height': height, 'width': width,
                     'id': img_id}
            json_dict['images'].append(image)

    if not is_training:
        for cls_name, cid in cls_map.items():
            cat = {'supercategory': 'none', 'id': cid, 'name': cls_name}
            json_dict['categories'].append(cat)
        json_fp = open(json_file, 'w')
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
        json_fp.close()

    return images, image_files_dict, image_anno_dict


def create_coco_label(is_training):
    """Get image path and annotation from COCO."""
    from pycocotools.coco import COCO

    coco_root = os.path.join(config.data_path, config.coco_root)
    data_type = config.val_data_type
    if is_training:
        data_type = config.train_data_type

    # Classes need to train or test.
    train_cls = config.classes
    train_cls_dict = {}
    for i, cls in enumerate(train_cls):
        train_cls_dict[cls] = i

    anno_json = os.path.join(coco_root, config.instances_set.format(data_type))

    coco = COCO(anno_json)
    classs_dict = {}
    cat_ids = coco.loadCats(coco.getCatIds())
    for cat in cat_ids:
        classs_dict[cat["id"]] = cat["name"]

    image_ids = coco.getImgIds()
    images = []
    image_path_dict = {}
    image_anno_dict = {}
    for img_id in image_ids:
        image_info = coco.loadImgs(img_id)
        file_name = image_info[0]["file_name"]
        anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = coco.loadAnns(anno_ids)
        image_path = os.path.join(coco_root, data_type, file_name)
        annos = []
        iscrowd = False
        for label in anno:
            bbox = label["bbox"]
            class_name = classs_dict[label["category_id"]]
            iscrowd = iscrowd or label["iscrowd"]
            if class_name in train_cls:
                x_min, x_max = bbox[0], bbox[0] + bbox[2]
                y_min, y_max = bbox[1], bbox[1] + bbox[3]
                annos.append(list(map(round, [y_min, x_min, y_max, x_max])) + [train_cls_dict[class_name]])

        if not is_training and iscrowd:
            continue
        if len(annos) >= 1:
            images.append(img_id)
            image_path_dict[img_id] = image_path
            image_anno_dict[img_id] = np.array(annos)

    return images, image_path_dict, image_anno_dict


def voc_data_to_mindrecord(mindrecord_dir, is_training, prefix="ssd.mindrecord", file_num=8):
    """Create MindRecord file by image_dir and anno_path."""
    mindrecord_path = os.path.join(mindrecord_dir, prefix)
    writer = FileWriter(mindrecord_path, file_num)
    images, image_path_dict, image_anno_dict = create_voc_label(is_training)

    ssd_json = {
        "img_id": {"type": "int32", "shape": [1]},
        "image": {"type": "bytes"},
        "annotation": {"type": "int32", "shape": [-1, 5]},
    }
    writer.add_schema(ssd_json, "ssd_json")

    for img_id in images:
        image_path = image_path_dict[img_id]
        with open(image_path, 'rb') as f:
            img = f.read()
        annos = np.array(image_anno_dict[img_id], dtype=np.int32)
        img_id = np.array([img_id], dtype=np.int32)
        row = {"img_id": img_id, "image": img, "annotation": annos}
        writer.write_raw_data([row])
    writer.commit()


def anno_parser(annos_str):
    """Parse annotation from string to list."""
    annos = []
    for anno_str in annos_str:
        anno = list(map(int, anno_str.strip().split(',')))
        annos.append(anno)
    return annos


def filter_valid_data(image_dir, anno_path):
    """Filter valid image file, which both in image_dir and anno_path."""
    images = []
    image_path_dict = {}
    image_anno_dict = {}
    if not os.path.isdir(image_dir):
        raise RuntimeError("Path given is not valid.")
    if not os.path.isfile(anno_path):
        raise RuntimeError("Annotation file is not valid.")

    with open(anno_path, "rb") as f:
        lines = f.readlines()
    for img_id, line in enumerate(lines):
        line_str = line.decode("utf-8").strip()
        line_split = str(line_str).split(' ')
        file_name = line_split[0]
        image_path = os.path.join(image_dir, file_name)
        if os.path.isfile(image_path):
            images.append(img_id)
            image_path_dict[img_id] = image_path
            image_anno_dict[img_id] = anno_parser(line_split[1:])

    return images, image_path_dict, image_anno_dict


def data_to_mindrecord_byte_image(dataset="coco", is_training=True, prefix="ssd.mindrecord", file_num=8):
    """Create MindRecord file."""
    mindrecord_path = os.path.join(config.data_path, config.mindrecord_dir, prefix)
    writer = FileWriter(mindrecord_path, file_num)
    if dataset == "coco":
        images, image_path_dict, image_anno_dict = create_coco_label(is_training)
    else:
        images, image_path_dict, image_anno_dict = filter_valid_data(config.image_dir, config.anno_path)

    ssd_json = {
        "img_id": {"type": "int32", "shape": [1]},
        "image": {"type": "bytes"},
        "annotation": {"type": "int32", "shape": [-1, 5]},
    }
    writer.add_schema(ssd_json, "ssd_json")

    for img_id in images:
        image_path = image_path_dict[img_id]
        with open(image_path, 'rb') as f:
            img = f.read()
        annos = np.array(image_anno_dict[img_id], dtype=np.int32)
        img_id = np.array([img_id], dtype=np.int32)
        row = {"img_id": img_id, "image": img, "annotation": annos}
        writer.write_raw_data([row])
    writer.commit()


def create_mindrecord(datadir, prefix="ssd.mindrecord", is_training=True):
    print("Start create dataset!")

    # It will generate mindrecord file in config.mindrecord_dir,
    # and the file name is ssd.mindrecord0, 1, ... file_num.
    dataset = "coco"
    mindrecord_dir = datadir
    # mindrecord_dir = os.path.join(config.data_path, config.mindrecord_dir)
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if dataset == "coco":
            coco_root = os.path.join(config.data_path, config.coco_root)
            if os.path.isdir(coco_root):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("coco", is_training, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("coco_root not exits.")
        elif dataset == "voc":
            if os.path.isdir(config.voc_root):
                print("Create Mindrecord.")
                voc_data_to_mindrecord(mindrecord_dir, is_training, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("voc_root not exits.")
        else:
            if os.path.isdir(config.image_dir) and os.path.exists(config.anno_path):
                print("Create Mindrecord.")
                data_to_mindrecord_byte_image("other", is_training, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("image_dir or anno_path not exits.")
    return mindrecord_file


def create_ssd_dataset(mindrecord_file, batch_size=32,
                       is_training=True):
    """Create SSD dataset with MindDataset."""
    cores = multiprocessing.cpu_count()
    rank = 0
    device_num = 1
    num_parallel_workers = 1
    use_multiprocessing = False
    if cores < num_parallel_workers:
        print("The num_parallel_workers {} is set too large, now set it {}".format(num_parallel_workers, cores))
        num_parallel_workers = cores
    ds = de.MindDataset(mindrecord_file, columns_list=["img_id", "image", "annotation"], num_shards=device_num,
                        shard_id=rank, num_parallel_workers=num_parallel_workers, shuffle=is_training)
    decode = de.vision.Decode()
    ds = ds.map(operations=decode, input_columns=["image"])
    change_swap_op = de.vision.HWC2CHW()
    # Computed from random subset of ImageNet training images
    normalize_op = de.vision.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                       std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    color_adjust_op = de.vision.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4)
    compose_map_func = (lambda img_id, image, annotation: preprocess_fn(img_id, image, annotation, is_training))
    if is_training:
        output_columns = ["image", "box", "label", "num_match"]
        trans = [color_adjust_op, normalize_op, change_swap_op]
    else:
        output_columns = ["img_id", "image", "image_shape"]
        trans = [normalize_op, change_swap_op]
    ds = ds.map(operations=compose_map_func, input_columns=["img_id", "image", "annotation"],
                output_columns=output_columns,
                python_multiprocessing=use_multiprocessing,
                num_parallel_workers=num_parallel_workers)
    ds = ds.map(operations=trans, input_columns=["image"], python_multiprocessing=use_multiprocessing,
                num_parallel_workers=num_parallel_workers)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds


if __name__ == '__main__':
    datadir = "/data1/pzy/minddb/ssd/datamind"
    batch_size = 8
    config.device_target = "CPU"
    # mindrecord_file = create_mindrecord(datadir, "ssd.mindrecord", True)

    mindrecord_file = os.path.join(datadir, "ssd.mindrecord" + "0")

    use_multiprocessing = (config.device_target != "CPU")
    train_set = create_ssd_dataset(mindrecord_file, batch_size=batch_size,
                                   is_training=True)
    test_set = create_ssd_dataset(mindrecord_file, batch_size=batch_size,
                                  is_training=False)

    epochs = 1
    train_iter = train_set.create_dict_iterator(output_numpy=True, num_epochs=epochs)
    test_iter = test_set.create_dict_iterator(output_numpy=True, num_epochs=epochs)

    dataset_size = train_set.get_dataset_size()
    print(f"Create dataset done! dataset size is {dataset_size}")
    for data in train_iter:
        # print(data)
        # print(len(data))
        # print(type(data))
        print("=#=###=#=###=#=###=#=###")
        print(data['image'].shape)
        print(data['box'].shape)
        print(data['label'].shape)
        print(data['num_match'].shape)
        print("=#=#=#=#=#=#=#=#=#=#=#=#")
