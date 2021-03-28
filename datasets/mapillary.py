"""
Mapillary Dataset Loader
"""
import logging
import json
import os
import numpy as np
from PIL import Image, ImageCms
from skimage import color

from torch.utils import data
import torch
import torchvision.transforms as transforms
import datasets.uniform as uniform
import datasets.cityscapes_labels as cityscapes_labels
import transforms.transforms as extended_transforms
import copy

from config import cfg

# Convert this dataset to have labels from cityscapes
num_classes = 19 #65
ignore_label = 255 #65
root = cfg.DATASET.MAPILLARY_DIR
config_fn = os.path.join(root, 'config.json')
color_mapping = []
id_to_trainid = {}
id_to_ignore_or_group = {}


def gen_id_to_ignore():
    global id_to_ignore_or_group
    for i in range(66):
        id_to_ignore_or_group[i] = ignore_label

    ### Convert each class to cityscapes one
    ### Road
    # Road
    id_to_ignore_or_group[13] = 0
    # Lane Marking - General
    id_to_ignore_or_group[24] = 0
    # Manhole
    id_to_ignore_or_group[41] = 0

    ### Sidewalk
    # Curb
    id_to_ignore_or_group[2] = 1
    # Sidewalk
    id_to_ignore_or_group[15] = 1

    ### Building
    # Building
    id_to_ignore_or_group[17] = 2

    ### Wall
    # Wall
    id_to_ignore_or_group[6] = 3

    ### Fence
    # Fence
    id_to_ignore_or_group[3] = 4

    ### Pole
    # Pole
    id_to_ignore_or_group[45] = 5
    # Utility Pole
    id_to_ignore_or_group[47] = 5

    ### Traffic Light
    # Traffic Light
    id_to_ignore_or_group[48] = 6

    ### Traffic Sign
    # Traffic Sign
    id_to_ignore_or_group[50] = 7

    ### Vegetation
    # Vegitation
    id_to_ignore_or_group[30] = 8

    ### Terrain
    # Terrain
    id_to_ignore_or_group[29] = 9

    ### Sky
    # Sky
    id_to_ignore_or_group[27] = 10

    ### Person
    # Person
    id_to_ignore_or_group[19] = 11

    ### Rider
    # Bicyclist
    id_to_ignore_or_group[20] = 12
    # Motorcyclist
    id_to_ignore_or_group[21] = 12
    # Other Rider
    id_to_ignore_or_group[22] = 12

    ### Car
    # Car
    id_to_ignore_or_group[55] = 13

    ### Truck
    # Truck
    id_to_ignore_or_group[61] = 14

    ### Bus
    # Bus
    id_to_ignore_or_group[54] = 15

    ### Train
    # On Rails
    id_to_ignore_or_group[58] = 16

    ### Motorcycle
    # Motorcycle
    id_to_ignore_or_group[57] = 17

    ### Bicycle
    # Bicycle
    id_to_ignore_or_group[52] = 18


def colorize_mask(image_array):
    """
    Colorize a segmentation mask
    """
    new_mask = Image.fromarray(image_array.astype(np.uint8)).convert('P')
    new_mask.putpalette(color_mapping)
    return new_mask


def make_dataset(quality, mode):
    """
    Create File List
    """
    assert (quality == 'semantic' and mode in ['train', 'val'])
    img_dir_name = None
    if quality == 'semantic':
        if mode == 'train':
            img_dir_name = 'training'
        if mode == 'val':
            img_dir_name = 'validation'
        mask_path = os.path.join(root, img_dir_name, 'labels')
    else:
        raise BaseException("Instance Segmentation Not support")

    img_path = os.path.join(root, img_dir_name, 'images')
    print(img_path)
    if quality != 'video':
        imgs = sorted([os.path.splitext(f)[0] for f in os.listdir(img_path)])
        msks = sorted([os.path.splitext(f)[0] for f in os.listdir(mask_path)])
        assert imgs == msks

    items = []
    c_items = os.listdir(img_path)
    if '.DS_Store' in c_items:
        c_items.remove('.DS_Store')

    for it in c_items:
        if quality == 'video':
            item = (os.path.join(img_path, it), os.path.join(img_path, it))
        else:
            item = (os.path.join(img_path, it),
                    os.path.join(mask_path, it.replace(".jpg", ".png")))
        items.append(item)
    return items


def gen_colormap():
    """
    Get Color Map from file
    """
    global color_mapping

    # load mapillary config
    with open(config_fn) as config_file:
        config = json.load(config_file)
    config_labels = config['labels']

    # calculate label color mapping
    colormap = []
    id2name = {}
    for i in range(0, len(config_labels)):
        colormap = colormap + config_labels[i]['color']
        id2name[i] = config_labels[i]['readable']
    color_mapping = colormap
    return id2name


class Mapillary(data.Dataset):
    def __init__(self, quality, mode, joint_transform_list=None,
                 transform=None, target_transform=None, target_aux_transform=None,
                 image_in=False, dump_images=False, class_uniform_pct=0,
                 class_uniform_tile=768, test=False):
        """
        class_uniform_pct = Percent of class uniform samples. 1.0 means fully uniform.
                            0.0 means fully random.
        class_uniform_tile_size = Class uniform tile size
        """
        gen_id_to_ignore()
        self.quality = quality
        self.mode = mode
        self.joint_transform_list = joint_transform_list
        self.transform = transform
        self.target_transform = target_transform
        self.image_in = image_in
        self.target_aux_transform = target_aux_transform
        self.dump_images = dump_images
        self.class_uniform_pct = class_uniform_pct
        self.class_uniform_tile = class_uniform_tile
        self.id2name = gen_colormap()
        self.imgs_uniform = None


        # find all images
        self.imgs = make_dataset(quality, mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        if test:
            np.random.shuffle(self.imgs)
            self.imgs = self.imgs[:200]

        if self.class_uniform_pct:
            json_fn = 'mapillary_tile{}.json'.format(self.class_uniform_tile)
            if os.path.isfile(json_fn):
                with open(json_fn, 'r') as json_data:
                    centroids = json.load(json_data)
                self.centroids = {int(idx): centroids[idx] for idx in centroids}
            else:
                # centroids is a dict (indexed by class) of lists of centroids
                self.centroids = uniform.class_centroids_all(
                    self.imgs,
                    num_classes,
                    id2trainid=None,
                    tile_size=self.class_uniform_tile)
                with open(json_fn, 'w') as outfile:
                    json.dump(self.centroids, outfile, indent=4)
        else:
            self.centroids = []
        self.build_epoch()

    def build_epoch(self):
        if self.class_uniform_pct != 0:
            self.imgs_uniform = uniform.build_epoch(self.imgs,
                                                    self.centroids,
                                                    num_classes,
                                                    self.class_uniform_pct)
        else:
            self.imgs_uniform = self.imgs

    def __getitem__(self, index):
        if len(self.imgs_uniform[index]) == 2:
            img_path, mask_path = self.imgs_uniform[index]
            centroid = None
            class_id = None
        else:
            img_path, mask_path, centroid, class_id = self.imgs_uniform[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in id_to_ignore_or_group.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        # Image Transformations
        if self.joint_transform_list is not None:
            for idx, xform in enumerate(self.joint_transform_list):
                if idx == 0 and centroid is not None:
                    # HACK! Assume the first transform accepts a centroid
                    img, mask = xform(img, mask, centroid)
                else:
                    img, mask = xform(img, mask)

        if self.dump_images:
            outdir = 'dump_imgs_{}'.format(self.mode)
            os.makedirs(outdir, exist_ok=True)
            if centroid is not None:
                dump_img_name = self.id2name[class_id] + '_' + img_name
            else:
                dump_img_name = img_name
            out_img_fn = os.path.join(outdir, dump_img_name + '.png')
            out_msk_fn = os.path.join(outdir, dump_img_name + '_mask.png')
            mask_img = colorize_mask(np.array(mask))
            img.save(out_img_fn)
            mask_img.save(out_msk_fn)

        if self.transform is not None:
            img = self.transform(img)

        rgb_mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img_gt = transforms.Normalize(*rgb_mean_std)(img)
        if self.image_in:
            eps = 1e-5
            rgb_mean_std = ([torch.mean(img[0]), torch.mean(img[1]), torch.mean(img[2])],
                    [torch.std(img[0])+eps, torch.std(img[1])+eps, torch.std(img[2])+eps])
        img = transforms.Normalize(*rgb_mean_std)(img)

        if self.target_aux_transform is not None:
            mask_aux = self.target_aux_transform(mask)
        else:
            mask_aux = torch.tensor([0])
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        mask = extended_transforms.MaskToTensor()(mask)
        return img, mask, img_name, mask_aux

    def __len__(self):
        return len(self.imgs_uniform)

    def calculate_weights(self):
        raise BaseException("not supported yet")
