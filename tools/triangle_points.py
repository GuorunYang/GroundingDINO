import cv2
import os
import csv
import numpy as np

def load_gnss_csv(csv_pth):
    with open(csv_pth, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            print(row)


def load_camera_param(param_pth):
    # 读取标定参数
    with open('calibration_folder/calibration.txt', 'r') as f:
        lines = f.readlines()
        H = np.array([float(item) for item in lines[0].split()]).reshape(3, 3)
        K = np.array([float(item) for item in lines[1].split()]).reshape(3, 3)
        R_T = np.array([float(item) for item in lines[2].split()]).reshape(3, 4)
        dist = np.array([float(item) for item in lines[3].split()])
        


if __name__ == "__main__":

    image_folder = 'image_folder/'
    bbox_folder = 'bbox_folder/'

    images = sorted(os.listdir(image_folder))
    bboxes = sorted(os.listdir(bbox_folder))

    points_2d = []

    # 从检测框中读取2D点
    for bbox_file in bboxes:
        with open(os.path.join(bbox_folder, bbox_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                xmin, ymin, xmax, ymax = [int(x) for x in line.split()]
                # 使用框的中心点作为2D点
                points_2d.append([(xmin+xmax)/2, (ymin+ymax)/2])

    # 伪造3D重建，通常这需要双目摄像机或多视角
    # 这里我们假设深度为1，因为我们没有足够的信息
    points_3d = cv2.triangulatePoints(
        np.dot(K, R_T), 
        np.dot(K, R_T), 
        np.array(points_2d).T, 
        np.array(points_2d).T)
    points_3d /= points_3d[3]