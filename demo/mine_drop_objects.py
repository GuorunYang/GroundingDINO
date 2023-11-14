import argparse
import os
import sys
import pickle as pkl

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

from tqdm import tqdm

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


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

def get_multiple_grounding_output(model, image, caption_list, box_threshold_list, 
                                  text_threshold=None, with_logits=True, cpu_only=False):
    assert len(caption_list) == len(box_threshold_list), "len of caption list != len of box threshold list"

    # Transform the model and image to GPU if available
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    tokenlizer = model.tokenizer

    multiple_logits = []
    multiple_boxes = []
    multiple_tokenized = []
    for caption in caption_list:
        # Handle captions
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)
        tokenized = tokenlizer(caption)
        multiple_logits.append(logits.cpu().clone())
        multiple_boxes.append(boxes.cpu().clone())
        multiple_tokenized.append(tokenized)

    multiple_phrases = []
    multiple_boxes_filt = []
    # filter output
    for i in range(len(multiple_logits)):
        logits_filt = multiple_logits[i]
        boxes_filt = multiple_boxes[i]
        box_threshold = box_threshold_list[i]
        tokenized = multiple_tokenized[i]
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
        multiple_phrases.append(pred_phrases)
        multiple_boxes_filt.append(boxes_filt)

    return multiple_boxes_filt, multiple_phrases


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases


    return boxes_filt, pred_phrases


def trans_bbox(src_box):
    des_box = {}
    center_x, center_y, width, height = \
        float(src_box[0]), float(src_box[1]), float(src_box[2]), float(src_box[3])
    des_box['x1'] = center_x - width / 2.0
    des_box['y1'] = center_y - height / 2.0
    des_box['x2'] = center_x + width / 2.0
    des_box['y2'] = center_y + height / 2.0
    return des_box

def filter_drop_objects(object_bboxes, object_cls, 
                        cross_cls_iou=0.75, inter_cls_iou=0.25):
    # Calculate the IoU matrix 
    object_num = len(object_bboxes)
    iou_matrix = np.zeros((object_num, object_num), dtype=np.float32)
    for i in range(object_num):
        iou_matrix[i, i] = 1.0
        for j in range(i+1, object_num):
            obj1 = trans_bbox(object_bboxes[i])
            obj2 = trans_bbox(object_bboxes[j])
            iou = get_iou(obj1, obj2)
            iou_matrix[i, j] = iou_matrix[j, i] = iou
    # print("IoU Matrix: ", iou_matrix)

    # Cluster the overlap objects
    group_clusters = []
    scan_objs = set()
    for i in range(object_num):
        cls_i = object_cls[i]
        if cls_i == "dropped object":
            if i in scan_objs:
                # If the object is scanned, skip
                continue
            obj_group = set()
            obj_group.add(i)
            scan_objs.add(i)
            for j in range(i+1, object_num):
                cls_j = object_cls[j]
                iou_ij = iou_matrix[i, j]
                if cls_j == "dropped object":
                    if iou_ij >= inter_cls_iou:
                        obj_group.add(j)
                        scan_objs.add(j)
                else:
                    if iou_ij >= cross_cls_iou:
                        obj_group.add(j)
                        scan_objs.add(j)
            group_clusters.append(obj_group)
    # print("Group cluster: ", group_clusters)


    # filter the objects which overlap with dropped objects
    reserve_indices = []
    for cluster_set in group_clusters:
        obj_list = list(cluster_set)
        # Select the dropped objects in cluster
        min_area = 1e6
        min_obj_id = -1
        valid_flag = True
        for obj_id in obj_list:
            if object_cls[obj_id] == "dropped object":
                obj_box = object_bboxes[obj_id]
                obj_width, obj_height = obj_box[2], obj_box[3]
                obj_area = obj_width * obj_height
                if obj_area < min_area:
                    min_obj_id = obj_id
                    min_area = obj_area
        if min_obj_id != -1:
            for obj_id in obj_list:
                obj_cls = object_cls[obj_id]
                if obj_cls == "dropped object":
                    continue
                if obj_cls == "arrow" or obj_cls == "cone" or obj_cls == "painting":
                    valid_flag = False
        if valid_flag:
            reserve_indices.append(min_obj_id)
    return reserve_indices


def save_results(pred_results, save_pth, save_label = None):
    H, W = pred_results["size"]
    boxes = pred_results["boxes"]
    labels = pred_results["labels"]
    box_lines = []
    max_score = -1.0
    for i, lbl in enumerate(labels):
        lbl_cls = lbl.split("(")[0]
        lbl_score = float(lbl.split("(")[1][:-1])
        # print("lbl_cls: ", lbl_cls)
        if save_label is not None:
            if save_label not in lbl:
                continue
        lbl_box = boxes[i]
        box_center_x = lbl_box[0] * W
        box_center_y = lbl_box[1] * H
        box_width = lbl_box[2] * W
        box_height = lbl_box[3] * H
        if save_label is None:
            box_ln = "{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
                lbl_cls, box_center_x, box_center_y, box_width, box_height, lbl_score
            )
        else:
            box_ln = "{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
                save_label, box_center_x, box_center_y, box_width, box_height, lbl_score
            )
        box_lines.append(box_ln)
    with open(save_pth, 'w') as f:
        f.writelines(box_lines)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, 
                        default="groundingdino/config/GroundingDINO_SwinB_cfg.py",
                        help="path to config file")
    parser.add_argument("--checkpoint_path", "-p", type=str,
                        default="./weights/groundingdino_swinb_cogcoor.pth",
                        help="path to checkpoint file")
    parser.add_argument("--image_dir", "-i", type=str, required=True, help="path to image directory")

    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    # parser.add_argument("--box_threshold", type=float, default=0.35, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    image_dir = args.image_dir
    text_threshold = args.text_threshold

    output_plot_drop_dir = os.path.join(args.output_dir, "plot_drop")
    output_plot_all_dir = os.path.join(args.output_dir, "plot_all")
    output_result_dir = os.path.join(args.output_dir, "result")
    os.makedirs(output_plot_drop_dir, exist_ok=True)
    os.makedirs(output_plot_all_dir, exist_ok=True)
    os.makedirs(output_result_dir, exist_ok=True)

    # define the text prompt
    text_prompt_dict = {
        "there is a dropped object on the road" : 0.40,
        "there is an arrow on the road" : 0.45,
        "there is a cone on the road" : 0.50,
        "there is a white painting on the road" : 0.35,
    }
    text_prompt_list = [k for k, v in text_prompt_dict.items()]
    box_threshold_list = [v for k, v in text_prompt_dict.items()]
    key_text_list = [
        "dropped object",
        "arrow",
        "cone",
        "painting",
        # "road",
    ]

    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

    # load image
    if os.path.isdir(args.image_dir):
        image_list = sorted(os.listdir(args.image_dir))
        for i, image_fn in enumerate(tqdm(image_list)):
            if not (image_fn.endswith(".jpg") or image_fn.endswith(".png")):
                continue
            image_path = os.path.join(args.image_dir, image_fn)
            # print("Image pth: ", image_path)
            image_pil, image = load_image(image_path)            
            boxes_filt_list, pred_phrases_list = get_multiple_grounding_output(
                model, image, text_prompt_list, box_threshold_list, text_threshold, 
                cpu_only=args.cpu_only, 
            )

            key_cls = []
            key_phrases = []
            key_boxes = []

            if len(boxes_filt_list) <= 0:
                continue
            else:
                # box_shape = boxes_filt_list[0].size(1)
                key_boxes = torch.empty((0, 4), device = boxes_filt_list[0].device)
                for prompt_boxes, prompt_phrases in zip(boxes_filt_list, pred_phrases_list):
                    for i, pred_phrase in enumerate(prompt_phrases):
                        for key_text in key_text_list:
                            if key_text in pred_phrase:
                                key_boxes = torch.cat((key_boxes, prompt_boxes[i, :].view(-1, 4)), dim=0)
                                key_phrases.append(pred_phrase)
                                key_cls.append(key_text)
                                # continue
            # print("Key cls: ", key_cls)
            # print("Key boxes: ", key_boxes)
            # print("Key phrases: ", key_phrases)

            size = image_pil.size
            pred_all_dict = {
                "boxes": key_boxes,
                "size": [size[1], size[0]],  # H,W
                "labels": key_phrases,
            }
            image_with_all_box = plot_boxes_to_image(image_pil, pred_all_dict)[0]
            image_with_all_box.save(os.path.join(output_plot_all_dir, image_fn))

            # Filter the false drop objects
            reserve_ids = filter_drop_objects(key_boxes, key_cls)
            reserve_phrases = []
            reserve_boxes = []
            reserve_cls = []
            # print("Reserve IDs: ", reserve_ids)
            if len(reserve_ids) > 0:
                reserve_boxes = torch.empty((0, 4), device = boxes_filt_list[0].device)
                # print("Reserve boxes: ", reserve_boxes)
                for obj_id in reserve_ids:
                    reserve_boxes = torch.cat((reserve_boxes, key_boxes[obj_id, :].view(-1, 4)), dim=0)
                    reserve_phrases.append(key_phrases[obj_id])
                    reserve_cls.append(key_cls[obj_id])
                # print("Reserve boxes: ", reserve_boxes)
                # visualize pred

                pred_drop_dict = {
                    "boxes": reserve_boxes,
                    "size": [size[1], size[0]],  # H,W
                    "labels": reserve_phrases,
                }
                # save preds
                save_pth = os.path.join(output_result_dir, image_fn+".txt")
                save_results(pred_drop_dict, save_pth)

                # import ipdb; ipdb.set_trace()

                image_with_drop_box = plot_boxes_to_image(image_pil, pred_drop_dict)[0]
                image_with_drop_box.save(os.path.join(output_plot_drop_dir, image_fn))
