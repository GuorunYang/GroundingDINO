import os
import json
import shutil
import numpy as np
from tqdm import tqdm


def nms(bounding_boxes, confidence_score, threshold = 0.2):
    """
    Non-max Suppression Algorithm

    @param list  Object candidate bounding boxes
    @param list  Confidence score of bounding boxes
    @param float IoU threshold

    @return Rest boxes after nms operation
    """
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score


def collect_images():
    '''
        This function is used to collect the images 
        from distinct case directories.
    '''
    case_dir = "/home/guorun.yang/data/corner_case/full_case"
    collect_dir = "/home/guorun.yang/data/corner_case/collect_images"
    os.makedirs(collect_dir, exist_ok=True)
    seq_list = os.listdir(case_dir)
    for i, seq_name in enumerate(tqdm(seq_list)):
        seq_dir = os.path.join(case_dir, seq_name)
        image_list = os.listdir(seq_dir)
        for image_fn in image_list:
            if not image_fn.endswith(".jpg"):
                continue
            src_image_pth = os.path.join(seq_dir, image_fn)
            des_image_pth = os.path.join(collect_dir, seq_name + "_" + image_fn)
            shutil.copyfile(src_image_pth, des_image_pth)


def collect_list_images():
    '''
        This function is used to collect the images 
        from distinct case directories.
    '''
    list_pth = "/home/guorun.yang/data/cornercase/list/A001_20231107_161603_arrow.txt"
    src_image_dir = "/data01/autra_core/data/records/A001_20231107_161603/_apollo_sensor_camera_upmiddle_left_30h_image_compressed"
    collect_dir = "/home/guorun.yang/data/cornercase/cases/A001_20231107_161603"
    os.makedirs(collect_dir, exist_ok=True)
    with open(list_pth, 'r') as f:
        list_lines = f.readlines()
        for i, ln in enumerate(list_lines):
            ln = ln.strip()
            src_image_pth = os.path.join(src_image_dir, ln)
            des_image_pth = os.path.join(collect_dir, ln)
            shutil.copyfile(src_image_pth, des_image_pth)


def collect_dig_images():
    '''
        This function is used to collect digged images
    '''
    dig_result_dir = "/home/guorun.yang/GroundingDINO/data/result/A001_20231107_161603_threshold_0.40/"
    # src_image_dir = "/data01/autra_core/data/records/A001_20231107_161603/_apollo_sensor_camera_upmiddle_left_30h_image_compressed"
    src_image_dir = "/home/guorun.yang/GroundingDINO/data/output/A001_20231107_161603_threshold_0.40"
    dig_image_dir = "/home/guorun.yang/GroundingDINO/data/dig/A001_20231107_161603_threshold_0.40"
    os.makedirs(dig_image_dir, exist_ok=True)
    dig_list = sorted(os.listdir(dig_result_dir))
    for fn in dig_list:
        image_fn = fn.rsplit('.', 1)[0]
        src_image_path = os.path.join(src_image_dir, image_fn)
        des_image_path = os.path.join(dig_image_dir, image_fn)
        shutil.copyfile(src_image_path, des_image_path)


def exclude_image():
    '''
        This function is used to extract the difficult images 
        excluded from json label 
    '''
    src_image_dir = "/home/guorun.yang/data/corner_case/collect_images"
    new_image_dir = "/home/guorun.yang/data/corner_case/difficult_images"
    os.makedirs(new_image_dir, exist_ok=True)
    json_pth = "/home/guorun.yang/data/corner_case/trainval.json"
    exclude_frames = set()
    with open(json_pth, 'r') as f:
        json_label = json.load(f)
        image_dict = {}
        for i, image_info in enumerate(json_label["images"]):
            image_id = image_info['id']
            image_fn = image_info['file_name']
            image_dict[image_id] = image_fn
        
        for k, anno_info in enumerate(json_label["annotations"]):
            anno_image_id = anno_info['image_id']
            anno_image_fn = image_dict[anno_image_id]
            exclude_frames.add(anno_image_fn)
    src_image_list = sorted(os.listdir(src_image_dir))
    for i, image_fn in enumerate(src_image_list):
        # frame_fn = src_image_fn.split('.')[0]
        if image_fn in exclude_frames:
            continue
        else:
            src_image_pth = os.path.join(src_image_dir, image_fn)
            des_image_pth = os.path.join(new_image_dir, image_fn)
            shutil.copyfile(src_image_pth, des_image_pth)


def collect_labels_v2():
    label_dict = {
        "images" : [],
        "annotations": [],
        "categories" : [
            {
                "id": 1,
                "name": "a dropped object"
            }
        ]
    }
    result_dir = "/home/guorun.yang/autra.tech/GroundingDINO/results/difficult_results"
    select_list_pth = "/home/guorun.yang/data/cornercase/select_image_list/difficult_dropped_object_images.txt"
    json_pth = "/home/guorun.yang/data/cornercase/annotations/test.json"
    select_frames = []
    with open(select_list_pth, 'r') as f:
        select_lines = f.readlines()
        select_frames = [ln.strip().split('.')[0] for ln in select_lines]
    anno_id = 0
    image_id = 0
    for i, frame_name in enumerate(select_frames):
        # Step 0: extract the image info
        image_id += 1
        result_pth = os.path.join(result_dir, frame_name + ".jpg.txt")
        frame_image_info = {
            "height" : 1080,
            "width" : 1920,
            "id" : image_id,
            "file_name" : frame_name + ".jpg"
        }
        label_dict["images"].append(frame_image_info)

        # Step 1: get the bboxes and scores
        frame_obj_bboxes = []
        frame_obj_scores = []
        if os.path.exists(result_pth):
            # bbox : [xmin, ymin, width, height]
            with open(result_pth, 'r') as f:
                result_lines = f.readlines()
                for ln in result_lines:
                    if ln.startswith("a dropped object"):
                        obj_info = ln.strip().split("object ")[-1]
                        obj_values = obj_info.split()
                        center_x, center_y, width, height, score = \
                            float(obj_values[0]), float(obj_values[1]), float(obj_values[2]), float(obj_values[3]), float(obj_values[4])
                        xmin = center_x - width / 2.0
                        ymin = center_y - height / 2.0
                        xmax = center_x + width / 2.0
                        ymax = center_y + height / 2.0
                        frame_obj_bboxes.append([xmin, ymin, xmax, ymax])
                        frame_obj_scores.append(score)

        # Step 2: use nms to filter the images
        picked_bboxes, picked_scores = nms(frame_obj_bboxes, frame_obj_scores)
        for bbox in picked_bboxes:
            # print("Picked bbox: ", bbox)
            bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox[0], bbox[1], bbox[2], bbox[3]
            bbox_width = (bbox_xmax - bbox_xmin)
            bbox_height = (bbox_ymax - bbox_ymin)
            # print("Width: {}, Height: {}".format(bbox_width, bbox_height))
            select_bbox = [bbox_xmin, bbox_ymin, bbox_width, bbox_height]
            select_area = bbox_width * bbox_height
            anno_id += 1
            frame_anno_info = {
                "iscrowd": 0,
                "category_id": 1,
                "id" : anno_id,
                "image_id" : image_id,
                "bbox" : select_bbox,
                "area" : select_area,
                "segmentation" : [[]]
            }
            label_dict["annotations"].append(frame_anno_info)
    with open(json_pth, 'w') as f:
        json.dump(label_dict, f, indent=4)


def collect_labels_v1():
    label_dict = {
        "images" : [],
        "annotations": [],
        "categories" : [
            {
                "id": 1,
                "name": "a dropped object"
            }
        ]
    }
    result_dir = "/home/guorun.yang/autra.tech/GroundingDINO/results/collect_results"
    select_list_pth = "/home/guorun.yang/data/corner_case/drop_object_frames.txt"
    json_pth = "/home/guorun.yang/data/corner_case/drop_object.json"
    select_frames = []
    with open(select_list_pth, 'r') as f:
        select_lines = f.readlines()
        select_frames = [ln.strip().split('.')[0] for ln in select_lines]
    anno_id = 0
    image_id = 0
    for i, frame_name in enumerate(select_frames):
        image_id += 1
        result_pth = os.path.join(result_dir, frame_name + ".jpg.txt")
        frame_image_info = {
            "height" : 1080,
            "width" : 1920,
            "id" : image_id,
            "file_name" : frame_name + ".jpg"
        }
        label_dict["images"].append(frame_image_info)
        if os.path.exists(result_pth):
            # bbox : [xmin, ymin, width, height]
            with open(result_pth, 'r') as f:
                result_lines = f.readlines()
                min_object_area = 1e6
                select_bbox = []
                select_area = 0
                for ln in result_lines:
                    if ln.startswith("a dropped object"):
                        obj_info = ln.strip().split("object ")[-1]
                        obj_values = obj_info.split()
                        center_x, center_y, width, height, score = \
                            float(obj_values[0]), float(obj_values[1]), float(obj_values[2]), float(obj_values[3]), float(obj_values[4])
                        xmin = center_x - width / 2.0
                        ymin = center_y - height / 2.0
                        obj_box = [xmin, ymin, width, height]
                        obj_area = width * height
                        if obj_area < min_object_area:
                            select_bbox = obj_box
                            select_area = obj_area
                anno_id += 1
                frame_anno_info = {
                    "iscrowd": 0,
                    "category_id": 1,
                    "id" : anno_id,
                    "image_id" : image_id,
                    "bbox" : select_bbox,
                    "area" : select_area,
                    "segmentation" : [[]]
                }
                label_dict["annotations"].append(frame_anno_info)
    with open(json_pth, 'w') as f:
        json.dump(label_dict, f, indent=4)


if __name__ == "__main__":
    # collect_labels()
    # collect_dig_images()
    # exclude_image()
    collect_labels_v2()