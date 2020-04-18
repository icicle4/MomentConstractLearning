
import os
from collections import defaultdict
import cv2

def load_annotation(det_file):
    with open(det_file, 'r') as f:
        contents = f.readlines()
        res = [
            content.strip().split(',') for content in contents
        ]

    frame_annotations = defaultdict(list)
    for frame, people_id, xmin, ymin, width, height, conf, _, _, _ in res:
        frame_annotations[int(frame)].append(
            [float(xmin), float(ymin), float(width), float(height)]
        )

    return frame_annotations

def clip_peoples(frame, annotations):
    clips = list()
    H, W = frame.shape[:2]
    for annotation in annotations:
        xmin, ymin, width, height = annotation
        clips.append(
            [frame[max(int(ymin), 0): min(int(ymin + height), H - 1),
                  max(int(xmin), 0): min(int(xmin + width), W - 1)],
                [xmin, ymin, width, height]
            ]
            )
    return frame, clips


def load_frame_clips(img_dir, det_file):
    img_names = {int(img.split('.')[0]): os.path.join(img_dir, img) for img in os.listdir(img_dir)}
    frame_annotations = load_annotation(det_file)
    return (clip_peoples(cv2.imread(img_names[img_frame]), frame_annotations[img_frame])
                        for img_frame in sorted(img_names.keys()))

