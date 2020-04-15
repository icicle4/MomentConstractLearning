
import argparse
import os
import cv2
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('-m', "--mot_root", default='', type=str)
parser.add_argument('-i', "--imagenet_root", default='', type=str)
parser.add_argument('-c', "--conf_thresh", default=0.4, type=float)
parser.add_argument('-r', "--train_ratio", default=0.8, type=float)
args = parser.parse_args()


def read_annotation(annotation_file, train_ratio):
    with open(annotation_file, 'r') as f:
        contents = f.readlines()
        res = [
            content.strip().split(',') for content in contents
        ]

    people_annotations = defaultdict(list)

    for frame, people_id, xmin, ymin, width, height, conf, _, _, _ in res:
        people_annotations[people_id].append(
            [int(frame), float(xmin), float(ymin), float(width), float(height), float(conf)]
        )

    people_train_val_annotations = dict()

    for people_id, annotation in people_annotations.items():
        sorted_annotations = sorted(annotation, key = lambda x:x[0])
        train_annotations = sorted_annotations[
            :int(train_ratio * len(sorted_annotations))
        ]

        val_annotations = sorted_annotations[
            int(train_ratio * len(sorted_annotations)):
        ]

        people_train_val_annotations[people_id] = (train_annotations, val_annotations)
    return people_train_val_annotations


def round_int(x):
    return int(round(x))


def clip_imgs(annotations, imgs, out_img_dir, video_name, people_id, conf_thresh):
    for annotation in annotations:
        frame, xmin, ymin, width, height, conf = annotation

        if conf > conf_thresh:
            img = cv2.imread(imgs[frame])
            H, W = img.shape[:2]
            clip = img[max(round_int(ymin), 0): min(round_int(ymin+ height), H - 1),
                        max(round_int(xmin), 0): min(round_int(xmin + width), W - 1)]

            people_imgs_dir = os.path.join(out_img_dir, f'{video_name}_{people_id}')
            if not os.path.exists(people_imgs_dir):
                os.makedirs(people_imgs_dir, exist_ok=True)

            cv2.imwrite(
                os.path.join(
                    people_imgs_dir, f'{frame}.png'
                ), clip
            )


def clip_peoples(annotations, imgs, conf_thresh, video_name, out_img_dir):

    for people_id, annotation in annotations.items():
        train_annotations, val_annotations = annotation

        train_out_img_dir = os.path.join(out_img_dir, 'train')
        val_out_img_dir = os.path.join(out_img_dir, 'val')

        clip_imgs(train_annotations, imgs, train_out_img_dir, video_name, people_id, conf_thresh)
        clip_imgs(val_annotations, imgs, val_out_img_dir, video_name, people_id, conf_thresh)


def convert(mot_dataset_root, imagenet_dataset_root, conf_thresh, train_ratio):
    mot_train_dataset = os.path.join(mot_dataset_root, 'train')
    gt_imagenet_root = os.path.join(imagenet_dataset_root, 'gt')
    #det_imagenet_root = os.path.join(imagenet_dataset_root, 'det')

    sub_datasets = os.listdir(mot_train_dataset)

    for i, sub_dataset in enumerate(sub_datasets):
        if not sub_dataset.startswith('.'):
            gt_annotations = read_annotation(os.path.join(
                mot_train_dataset, sub_dataset, 'gt', 'gt.txt'
            ), train_ratio)

            # det_annotation = read_annotation(os.path.join(
            #     mot_train_dataset, sub_dataset, 'det.txt'
            # ), train_ratio)

            imgs = {int(img_name.split('.')[0]):
                        os.path.join(mot_train_dataset,
                                     sub_dataset,
                                     'img1',
                                     img_name
                                     )
                         for img_name in os.listdir(os.path.join(mot_train_dataset,
                                                                sub_dataset,
                                                                'img1'))}

            clip_peoples(
                gt_annotations, imgs, conf_thresh, sub_dataset, gt_imagenet_root
            )

            # clip_peoples(
            #     det_annotation, imgs, conf_thresh, sub_dataset, det_imagenet_root
            # )


if __name__ == '__main__':
    mot_dataset_root = args.mot_root
    imagenet_dataset_root = args.imagenet_root
    conf_thresh = args.conf_thresh
    train_ratio = args.train_ratio

    convert(mot_dataset_root, imagenet_dataset_root, conf_thresh, train_ratio)















