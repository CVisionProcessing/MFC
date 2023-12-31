import os
import cv2
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import functional.utils.custom_transforms as tr
from functional.utils.mask_io import read_mask
video_names = None

def dataloader_davis(filepath):
    global video_names
    video_txt = filepath + '/ImageSets/2017/val.txt'
    video_names = sorted(open(video_txt).readlines())
    annotation_all = []
    jpeg_all = []
    video_all = []
    root_label_path = os.path.join(filepath, 'Annotations/480p/')
    root_img_path = os.path.join(filepath, 'JPEGImages/480p/')
    annotation_index_all = []
    for video in video_names:
        video_all.append(video.strip())
        annotation_index_all.append([0])
        anno_path = os.path.join(filepath, 'Annotations/480p/' + video.strip())
        cat_annos = sorted(os.listdir(anno_path))
        annotation_all.append(cat_annos)
        jpeg_path = os.path.join(filepath, 'JPEGImages/480p/' + video.strip())
        cat_jpegs = sorted(os.listdir(jpeg_path))
        jpeg_all.append(cat_jpegs)
    return root_label_path, root_img_path, annotation_all, jpeg_all, video_all, annotation_index_all

def dataloader_custom(filepath):
    global video_names
    video_names = sorted(os.listdir(os.path.join(filepath, 'valid_demo/JPEGImages')))
    annotation_all = []
    jpeg_all = []
    video_all = []
    annotation_index_all = []
    root_img_path = os.path.join(filepath, 'valid_demo/JPEGImages')
    root_label_path = os.path.join(filepath, 'valid_demo/Annotations')
    for idx, video in enumerate(video_names):
        video_all.append(video)
        images = sorted(np.unique(os.listdir(os.path.join(root_img_path, video))))
        labels = sorted(np.unique(os.listdir(os.path.join(root_label_path, video))))
        first_frame_idx = images.index(labels[0].replace('png', 'jpg'))
        images = images[first_frame_idx:]
        annotation_all.append(labels)
        jpeg_all.append(images)
        anno_idx = []
        for anno in labels:
            anno = anno.replace('png', 'jpg')
            anno_idx.append(images.index(anno))
        annotation_index_all.append(anno_idx)
    return root_label_path, root_img_path, annotation_all, jpeg_all, video_all, annotation_index_all

def frame_read(path):
    image = cv2.imread(path)
    image = np.float32(image) / 255.0
    return image
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img

def annotation_read(path):
    anno = read_mask(path)
    anno = np.expand_dims(anno, 0)
    return torch.Tensor(anno).contiguous().long()

class test_image_folder_davis(data.Dataset):
    def __init__(self, train_data, training=False):
        root_annos, root_imgs, annos, jpegs, videos, annos_index = train_data
        self.root_annos = root_annos
        self.root_imgs = root_imgs
        self.annos = annos
        self.jpegs = jpegs
        self.videos = videos
        self.annos_index = annos_index
        self.training = training
        self.composed_transforms = transforms.Compose([
            tr.ConvertToLAB(),
            tr.ToTensor(),
            tr.Normalize([50, 0, 0], [50, 127, 127])])

    def __getitem__(self, index):
        annos = self.annos[index]
        jpegs = self.jpegs[index]
        video_name = self.videos[index]
        annos_index = self.annos_index[index]
        annotations = [annotation_read(os.path.join(self.root_annos, video_name, anno)) for anno in annos]
        images = [frame_read(os.path.join(self.root_imgs, video_name, jpeg)) for jpeg in jpegs]
        lab_images = self.composed_transforms(images)
        rgb_images = [load_image(os.path.join(self.root_imgs, video_name, jpeg)) for jpeg in jpegs]
        _, height, width = annotations[0].shape
        meta = {"video_name": video_name, "annotation_index": annos_index, "frame_names": [x.replace('jpg', 'png') for x in jpegs],
                "height": height, "width": width, "abs_frame_path": [os.path.join(self.root_imgs, video_name, x) for x in jpegs]}
        return lab_images, rgb_images, annotations, meta

    def __len__(self):
        return len(self.annos)

class test_image_folder_custom(data.Dataset):
    def __init__(self, train_data, training=False):
        root_annos, root_imgs, annos, jpegs, videos, annos_index = train_data
        self.root_annos = root_annos
        self.root_imgs = root_imgs
        self.annos = annos
        self.jpegs = jpegs
        self.videos = videos
        self.annos_index = annos_index
        self.training = training

    def __getitem__(self, index):
        annos = self.annos[index]
        jpegs = self.jpegs[index]
        video_name = self.videos[index]
        annos_index = self.annos_index[index]
        annotations = [annotation_read(os.path.join(self.root_annos, video_name, anno)) for anno in annos]
        images_rgb = [frame_read(os.path.join(self.root_imgs, video_name, jpeg)) for jpeg in jpegs]
        _, height, width = annotations[0].shape
        meta = {"video_name": video_name, "annotation_index": annos_index, "frame_names": [x.replace('jpg', 'png') for x in jpegs],
                "height": height, "width": width, "abs_frame_path": [os.path.join(self.root_imgs, video_name, x) for x in jpegs]}
        return images_rgb, annotations, meta

    def __len__(self):
        return len(self.annos)
