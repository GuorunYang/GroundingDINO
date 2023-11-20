import os
import json
import numpy as np
from tqdm import tqdm
import cv2
import PIL
from PIL import Image, ImageDraw, ImageFont
import fiftyone as fo
# from fiftyone.utils.coco import COCODetectionDatasetExporter

def check_image_size(image_dir):
    image_list = sorted(os.listdir(image_dir))
    for image_fn in image_list:
        image_path = os.path.join(image_dir, image_fn)
        image_pil = Image.open(image_path).convert("RGB")
        image_width, image_height = image_pil.size
        if image_width != 1920 or image_height != 1080:
            print("Image pth : {} width: {}, height: {}".format(
                image_path, image_width, image_height
            ))


def resize_image(src_dir, des_dir):
    image_list = sorted(os.listdir(src_dir))
    os.makedirs(des_dir, exist_ok=True)
    for i, image_fn in enumerate(tqdm(image_list, desc="Resize images")):
        # Robin_20230802
        # Robin_20230816
        # Messi_20230802
        src_pth = os.path.join(src_dir, image_fn)
        des_pth = os.path.join(des_dir, image_fn)
        src_image = cv2.imread(src_pth)
        image_width, image_height = src_image.shape[1], src_image.shape[0]
        if image_width == 3840 and image_height == 2160:
            des_size = (int(image_width / 2), int(image_height / 2))
            des_image = cv2.resize(src_image, des_size, interpolation = cv2.INTER_LANCZOS4)
            print()
            cv2.imwrite(des_pth, des_image)


def resize_results(src_dir, des_dir):
    result_list = sorted(os.listdir(src_dir))
    os.makedirs(des_dir, exist_ok=True)
    for i, result_fn in enumerate(tqdm(result_list, desc="Resize results")):
        if result_fn.startswith("Robin_20230816") \
            or result_fn.startswith("Robin_20230802") \
            or result_fn.startswith("Messi_20230802"):
            src_result_pth = os.path.join(src_dir, result_fn)
            des_result_pth = os.path.join(des_dir, result_fn)
            src_lines = []
            des_lines = []
            with open(src_result_pth, 'r') as f:
                src_lines = f.readlines()
                for ln in src_lines:
                    ln = ln.strip().rsplit(' ', 5)
                    obj_cls = ln[0]
                    cx, cy, w, h, score = float(ln[1]), float(ln[2]), float(ln[3]), float(ln[4]), float(ln[5])
                    cx /= 2.0
                    cy /= 2.0
                    w /= 2.0
                    h /= 2.0
                    new_ln = "{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
                        obj_cls, cx, cy, w, h, score
                    )
                    des_lines.append(new_ln)
            with open(des_result_pth, 'w') as f:
                f.writelines(des_lines)


def get_unique_list(list_pth):
    raw_list_lines = []
    line_set = set()
    if os.path.exists(list_pth):
        with open(list_pth, 'r') as f:
            raw_list_lines = f.readlines()
            for ln in raw_list_lines:
                ln = ln.strip()
                line_set.add(ln)
    unique_list_lines = []
    for ln in line_set:
        unique_list_lines.append(ln + '\n')
    unique_list_lines = sorted(unique_list_lines)
    des_list_pth = list_pth.rsplit('.', 1)[0] + "_unique.txt"
    print("Lines of {}: {}->{}".format(list_pth, len(raw_list_lines), len(unique_list_lines)))
    with open(des_list_pth, 'w') as f:
        f.writelines(unique_list_lines)
    

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def trans_bbox(src_box):
    des_box = {}
    center_x, center_y, width, height = \
        float(src_box[0]), float(src_box[1]), float(src_box[2]), float(src_box[3])
    des_box['x1'] = center_x - width / 2.0
    des_box['y1'] = center_y - height / 2.0
    des_box['x2'] = center_x + width / 2.0
    des_box['y2'] = center_y + height / 2.0
    return des_box

def trans_bbox2fiftyone(src_box, image_width, image_height):
    src_box = src_box['bbox']
    center_x, center_y, box_width, box_height = \
        float(src_box[0]), float(src_box[1]), float(src_box[2]), float(src_box[3])
    top_left_x = center_x - box_width / 2.0
    top_left_y = center_y - box_height / 2.0
    if image_width > 0 and image_height > 0:
        top_left_x /= image_width
        top_left_y /= image_height
        box_width /= image_width
        box_height /= image_height
        return [top_left_x, top_left_y, box_width, box_height]
    else:
        print("Invalid image width and height")
        return [top_left_x, top_left_y, box_width, box_height]


def cluster_objects(obj_annos):
    obj_num = len(obj_annos)
    iou_matrix = np.zeros((obj_num, obj_num), dtype=np.float32)
    for i in range(obj_num):
        iou_matrix[i, i] = 1.0
        for j in range(i+1, obj_num):
            obj1 = trans_bbox(obj_annos[i]['bbox'])
            obj2 = trans_bbox(obj_annos[j]['bbox'])
            iou = get_iou(obj1, obj2)
            iou_matrix[i, j] = iou_matrix[j, i] = iou
    
    # Cluster the overlap objects
    group_clusters = []
    scan_objs = set()
    for i in range(obj_num):
        if i in scan_objs:
            # If the object is scanned, skip
            continue
        cls_i = obj_annos[i]['cls']
        obj_group = set()
        obj_group.add(i)
        scan_objs.add(i)
        for j in range(i+1, obj_num):
            if j in scan_objs:
                continue
            cls_j = obj_annos[j]['cls']
            iou_ij = iou_matrix[i, j]
            if (cls_i == "a dropped object") and (cls_j == "a dropped object"):
                if iou_ij >= 0.25:
                    obj_group.add(j)
                    scan_objs.add(j)
            else:
                if iou_ij >= 0.75:
                    obj_group.add(j)
                    scan_objs.add(j)
        group_clusters.append(obj_group)
    return group_clusters


def load_detections(result_pth):
    result_dict = {}
    result_list = []
    if os.path.isdir(result_pth):
        result_dir = result_pth
        result_list = [os.path.join(result_dir, fn) for fn in sorted(os.listdir(result_dir))]
    elif os.path.isfile(result_pth):
        result_list.append(result_pth)
    else:
        raise TypeError
    
    for i, result_name in enumerate(result_list):
        # Robin_20230802
        # Robin_20230816
        # Messi_20230802
        # seq_name = result_name.split('/')[-1]
        # if not (seq_name.startswith("Robin_20230802") \
        #     or seq_name.startswith("Robin_20230816") \
        #     or seq_name.startswith("Messi_20230802")):
        #     print("Ignore the frame: ", seq_name)
        #     continue
        frame_name = result_name.rsplit('/')[-1].split('.')[0]
        frame_results = []
        with open(result_name, 'r') as f:
            result_lines = f.readlines()
            for ln in result_lines:
                ln = ln.strip().rsplit(' ', 5)
                obj_cls = ln[0]
                cx, cy, w, h, score = float(ln[1]), float(ln[2]), float(ln[3]), float(ln[4]), float(ln[5])
                obj_anno = {
                    "cls": obj_cls,
                    "bbox" : [cx, cy, w, h, score],
                }
                frame_results.append(obj_anno)
        result_dict[frame_name] = frame_results
    return result_dict


def extract_ground_objs(frame_dets):
    frame_clusters = cluster_objects(frame_dets)
    frame_ground_objects = []
    for obj_group in frame_clusters:
        drop_obj_id = -1
        for obj_id in obj_group:
            if frame_dets[obj_id]["cls"] == "a dropped object":
                drop_obj_id = obj_id
        if drop_obj_id != -1:
            ground_obj = frame_dets[drop_obj_id]
            frame_ground_objects.append(ground_obj)
            continue
        # else:
        #     print("No arrow object in the cluster")
    return frame_ground_objects


def extract_lane_objs(frame_dets):
    frame_clusters = cluster_objects(frame_dets)
    frame_lane_objects = []
    for obj_group in frame_clusters:
        arrow_obj_id = -1
        painting_obj_id = -1
        drop_obj_id = -1
        for obj_id in obj_group:
            if frame_dets[obj_id]["cls"] == "a white painting":
                painting_obj_id = obj_id
            if frame_dets[obj_id]["cls"] == "a dropped object":
                drop_obj_id = obj_id
            if frame_dets[obj_id]["cls"] == "an arrow":
                arrow_obj_id = obj_id
        if painting_obj_id != -1:
            arrow_obj = frame_dets[painting_obj_id]
            frame_lane_objects.append(arrow_obj)
            continue
        elif drop_obj_id != -1:
            arrow_obj = frame_dets[drop_obj_id]
            frame_lane_objects.append(arrow_obj)
            continue
        elif arrow_obj_id != -1:
            arrow_obj = frame_dets[arrow_obj_id]
            frame_lane_objects.append(arrow_obj)
            continue
        # else:
        #     print("No arrow object in the cluster")
    return frame_lane_objects


def extract_arrow_objs(frame_dets):
    frame_clusters = cluster_objects(frame_dets)
    frame_arrow_objects = []
    for obj_group in frame_clusters:
        arrow_obj_id = -1
        painting_obj_id = -1
        drop_obj_id = -1
        for obj_id in obj_group:
            if frame_dets[obj_id]["cls"] == "an arrow":
                arrow_obj_id = obj_id
            if frame_dets[obj_id]["cls"] == "a white painting":
                painting_obj_id = obj_id
            if frame_dets[obj_id]["cls"] == "a dropped object":
                drop_obj_id = obj_id
        if arrow_obj_id != -1:
            arrow_obj = frame_dets[arrow_obj_id]
            frame_arrow_objects.append(arrow_obj)
            continue
        elif painting_obj_id != -1:
            arrow_obj = frame_dets[painting_obj_id]
            frame_arrow_objects.append(arrow_obj)
            continue
        elif drop_obj_id != -1:
            arrow_obj = frame_dets[drop_obj_id]
            frame_arrow_objects.append(arrow_obj)
            continue
        # else:
        #     print("No arrow object in the cluster")
    return frame_arrow_objects


def extract_drop_objs(frame_dets):
    frame_clusters = cluster_objects(frame_dets)
    frame_drop_objects = []
    for obj_group in frame_clusters:
        min_area = 1e6
        min_obj_id = -1
        for obj_id in obj_group:
            if frame_dets[obj_id]["cls"] == "a dropped object":
                obj_box = frame_dets[obj_id]["bbox"]
                obj_width, obj_height = obj_box[2], obj_box[3]
                obj_area = obj_width * obj_height
                if obj_area < min_area:
                    min_obj_id = obj_id
                    min_area = obj_area
        if min_obj_id != -1:
            drop_obj = frame_dets[min_obj_id]
            frame_drop_objects.append(drop_obj)
    return frame_drop_objects


def collect_drop_annos(drop_list_pth, image_dir, drop_detections):
    frame_drop_annos = {}
    with open(drop_list_pth, 'r') as f:
        list_lines = f.readlines()
        for i, ln in enumerate(tqdm(list_lines, desc="Collect drop annotation")):
            # ln = ln.strip()
            # seq_name = ln.rsplit('_', 1)[0]
            image_name = ln.strip()
            frame_name = image_name.split('.')[0]
            image_pth = os.path.join(image_dir, image_name)
            if (os.path.exists(image_pth)) and (frame_name in drop_detections):
                frame_drop_annos[image_pth] = {}
                frame_drop_annos[image_pth] = extract_drop_objs(drop_detections[frame_name])
                if len(frame_drop_annos[image_pth]) == 0:
                    print("Frame {} has no drop detections".format(image_name))
            else:
                print("Frame {} does not exist or has no detections".format(image_name))
    return frame_drop_annos


def collect_arrow_annos(arrow_list_pth, image_dir, arrow_detections):
    frame_arrow_annos = {}
    with open(arrow_list_pth, 'r') as f:
        list_lines = f.readlines()
        for i, ln in enumerate(tqdm(list_lines, desc="Collect arrow annotation")):
            image_name = ln.strip()
            frame_name = image_name.split('.')[0]
            image_pth = os.path.join(image_dir, image_name)
            if (os.path.exists(image_pth)) and (frame_name in arrow_detections):
                frame_arrow_annos[image_pth] = {}
                frame_arrow_annos[image_pth] = extract_arrow_objs(arrow_detections[frame_name])
                if len(frame_arrow_annos[image_pth]) == 0:
                    print("Frame {} has no arrow detections".format(image_name))
            else:
                print("Frame {} does not exist or has no detections".format(image_name))
    return frame_arrow_annos


def collect_lane_annos(lane_list_pth, image_dir, lane_detections):
    frame_lane_annos = {}
    with open(lane_list_pth, 'r') as f:
        list_lines = f.readlines()
        for i, ln in enumerate(tqdm(list_lines, desc="Collect lane annotation")):
            image_name = ln.strip()
            frame_name = image_name.split('.')[0]
            image_pth = os.path.join(image_dir, image_name)
            if (os.path.exists(image_pth)) and (frame_name in lane_detections):
                frame_lane_annos[image_pth] = {}
                frame_lane_annos[image_pth] = extract_lane_objs(lane_detections[frame_name])
                if len(frame_lane_annos[image_pth]) == 0:
                    print("Frame {} has no lane detections".format(image_name))
            else:
                print("Frame {} does not exist or has no detections".format(image_name))
    return frame_lane_annos

def collect_ground_annos(ground_list_pth, image_dir, ground_detections):
    frame_ground_annos = {}
    with open(ground_list_pth, 'r') as f:
        list_lines = f.readlines()
        for i, ln in enumerate(tqdm(list_lines, desc="Collect ground-pair annotation")):
            image_name = ln.strip()
            frame_name = image_name.split('.')[0]
            image_pth = os.path.join(image_dir, image_name)
            if (os.path.exists(image_pth)) and (frame_name in ground_detections):
                frame_ground_annos[image_pth] = {}
                frame_ground_annos[image_pth] = extract_ground_objs(ground_detections[frame_name])
                if len(frame_ground_annos[image_pth]) == 0:
                    print("Frame {} has no ground detections".format(image_name))
            else:
                print("Frame {} does not exist or has no detections".format(image_name))
    return frame_ground_annos


def main():
    image_dir = "/home/guorun.yang/data/cornercase/images"
    base_dir = "/home/guorun.yang/autra.tech/GroundingDINO/data/round3"
    export_dir = "/home/guorun.yang/data/cornercase/annotations"
    os.makedirs(export_dir, exist_ok=True)
    fo_dataset = fo.Dataset(name="cornercase")
    # fo_dataset = fo.Dataset.from_images_dir(image_dir)
    # print("Fo dataset: ", fo_dataset)
    # check_image_size(image_dir)

    # drop_result_dir = "/home/guorun.yang/autra.tech/GroundingDINO/data/round3/result_drop"
    # all_result_dir = "/home/guorun.yang/autra.tech/GroundingDINO/data/round3/result_all"
    drop_result_dir = "/home/guorun.yang/data/cornercase/round3_all_images/result_drop"
    all_result_dir = "/home/guorun.yang/data/cornercase/round3_all_images/result_all"
    drop_detections = load_detections(drop_result_dir)
    all_detections = load_detections(all_result_dir)

    drop_list_name = "frame_drop_unique.txt"
    arrow_list_name = "frame_arrow_unique.txt"
    ground_list_name = "frame_ground_unique.txt"
    lane_list_name = "frame_lane_unique.txt"
    drop_list_pth = os.path.join(base_dir, drop_list_name)
    arrow_list_pth = os.path.join(base_dir, arrow_list_name)
    ground_list_pth = os.path.join(base_dir, ground_list_name)
    lane_list_pth = os.path.join(base_dir, lane_list_name)

    image_width = 1920
    image_height = 1080

    ## resize images and results
    # resize_image_dir = "/home/guorun.yang/data/cornercase/images_resize"
    # resize_image(resize_image_dir, resize_dir)
    # drop_resize_dir = "/home/guorun.yang/data/cornercase/round3_all_images/result_drop_resize"
    # all_resize_dir = "/home/guorun.yang/data/cornercase/round3_all_images/result_all_resize"
    # resize_results(drop_result_dir, drop_resize_dir)
    # resize_results(all_result_dir, all_resize_dir)

    # Collect drop annotations
    drop_annos = collect_drop_annos(drop_list_pth, image_dir, all_detections)
    for image_pth, frame_drop_objs in drop_annos.items():
        image_name = image_pth.split('/')[-1]
        # if image_name == "Robin_20230802_104644_1690947616_1690947646_00031994.jpg":
        #     print(frame_drop_objs)
        if len(frame_drop_objs) > 0:
            frame_annos = []
            for drop_obj in frame_drop_objs:
                drop_fiftyone_obj = trans_bbox2fiftyone(drop_obj, image_width, image_height)
                frame_annos.append(
                    fo.Detection(
                        label="a dropped object", 
                        bounding_box=drop_fiftyone_obj
                    )
                )
            frame_sample = fo.Sample(filepath=image_pth)
            frame_sample["ground_truth"] = fo.Detections(detections=frame_annos)
            fo_dataset.add_sample(frame_sample)
        else:
            print("Frame {} has no drop object".format(frame_drop_objs))

    # Collect arrow annotations
    arrow_annos = collect_arrow_annos(arrow_list_pth, image_dir, all_detections)
    for image_pth, frame_arrow_objs in arrow_annos.items():
        if len(frame_arrow_objs) > 0:
            frame_annos = []
            for arrow_obj in frame_arrow_objs:
                arrow_fiftyone_obj = trans_bbox2fiftyone(arrow_obj, image_width, image_height)
                frame_annos.append(
                    fo.Detection(
                        label="an arrow", 
                        bounding_box=arrow_fiftyone_obj
                    )
                )
            frame_sample = fo.Sample(filepath=image_pth)
            frame_sample["ground_truth"] = fo.Detections(detections=frame_annos)
            fo_dataset.add_sample(frame_sample)
        else:
            print("Frame {} has no arrow object".format(frame_arrow_objs))
    
    # Collect lane annotations
    lane_annos = collect_lane_annos(lane_list_pth, image_dir, all_detections)
    for image_pth, frame_lane_objs in lane_annos.items():
        if len(frame_lane_objs) > 0:
            frame_annos = []
            for lane_obj in frame_lane_objs:
                lane_fiftyone_obj = trans_bbox2fiftyone(lane_obj, image_width, image_height)
                frame_annos.append(
                    fo.Detection(
                        label="a white painting", 
                        bounding_box=lane_fiftyone_obj
                    )
                )
            frame_sample = fo.Sample(filepath=image_pth)
            frame_sample["ground_truth"] = fo.Detections(detections=frame_annos)
            fo_dataset.add_sample(frame_sample)
        else:
            print("Frame {} has no lane object".format(frame_lane_objs))

    # Collect lane annotations
    ground_annos = collect_ground_annos(ground_list_pth, image_dir, all_detections)
    for image_pth, frame_ground_objs in ground_annos.items():
        if len(frame_ground_objs) > 0:
            frame_annos = []
            for ground_obj in frame_ground_objs:
                lane_fiftyone_obj = trans_bbox2fiftyone(ground_obj, image_width, image_height)
                frame_annos.append(
                    fo.Detection(
                        label="a ground repair", 
                        bounding_box=lane_fiftyone_obj
                    )
                )
            frame_sample = fo.Sample(filepath=image_pth)
            frame_sample["ground_truth"] = fo.Detections(detections=frame_annos)
            fo_dataset.add_sample(frame_sample)
        else:
            print("Frame {} has no lane object".format(frame_lane_objs))


    # Export annotations
    # export_pth = os.path.join(export_dir, "drop_annotations.json")
    export_pth = os.path.join(export_dir, "annotations_4cls.json")
    # fo_dataset.persistent = True
    fo_dataset.export(
        # export_dir=export_dir,
        dataset_type=fo.types.COCODetectionDataset,
        labels_path = export_pth,
        label_field = "ground_truth",
    )
    # pretty print
    json_labels = {}
    with open(export_pth, 'r') as f:
        json_labels = json.load(f)
    with open(export_pth, 'w') as f:
        json.dump(json_labels, f, indent=4)
    
    # view_dataset = True
    # if view_dataset:
    #     session = fo.launch_app(fo_dataset)
    #     session.wait()


if __name__ == '__main__':
    main()
