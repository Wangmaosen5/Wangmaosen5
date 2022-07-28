import colorsys
import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image)
from utils.utils_bbox import DecodeBox

'''
训练自己的数据集必看注释！
'''


class YOLO(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        # "model_path"        : 'model_data/yolo_weights.pth',
        "model_path": 'G:/Wangmaosen/Code/YOLOv3_Sinatten_atten/YOLOv3_Sinatten_atten/logs/20220528_yolov3_voc_loss_pth/ep100-loss0.702-val_loss1.649.pth',
        "classes_path": 'model_data/mining_classes.txt',
        # ---------------------------------------------------------------------#
        #   anchors_path代表先验框对应的txt文件，一般不修改。
        #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
        # ---------------------------------------------------------------------#
        "anchors_path": 'model_data/yolo_anchors.txt',
        "anchors_mask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        # ---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        # ---------------------------------------------------------------------#
        "input_shape": [416, 416],
        # ---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # ---------------------------------------------------------------------#
        "confidence": 0.5,
        # ---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        # ---------------------------------------------------------------------#
        "nms_iou": 0.3,
        # ---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        # ---------------------------------------------------------------------#
        "letterbox_image": False,
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]),
                                   self.anchors_mask)

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    def generate(self):
        # ---------------------------------------------------#
        #   建立yolov3模型，载入yolov3模型的权重
        # ---------------------------------------------------#
        self.net = YoloBody(self.anchors_mask, self.num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            output_list = self.bbox_util.decode_box(outputs)

        return output_list


def show_CAM(image_path, feature_maps, class_id, all_ids=8, show_one_layer=True):
    """
    feature_maps: this is a list [tensor,tensor,tensor], tensor shape is [1, 3, N, N, all_ids]
    """
    SHOW_NAME = ["score", "class", "class_score"]
    img_ori = cv2.imread(image_path)
    layers0 = feature_maps[0].reshape([-1, all_ids])
    layers1 = feature_maps[1].reshape([-1, all_ids])
    layers2 = feature_maps[2].reshape([-1, all_ids])
    layers = torch.cat([layers0, layers1, layers2], 0)
    score_max_v = layers[:, 4].max()  # compute max of score from all anchor
    score_min_v = layers[:, 4].min()  # compute min of score from all anchor
    class_max_v = layers[:, 5 + class_id].max()  # compute max of class from all anchor
    class_min_v = layers[:, 5 + class_id].min()  # compute min of class from all anchor
    all_ret = [[], [], []]
    for j in range(3):  # layers
        layer_one = feature_maps[j]
        # compute max of score from three anchor of the layer
        anchors_score_max = layer_one[0, ..., 4].max(0)[0]
        # compute max of class from three anchor of the layer
        anchors_class_max = layer_one[0, ..., 5 + class_id].max(0)[0]

        scores = ((anchors_score_max - score_min_v) / (
                score_max_v - score_min_v))

        classes = ((anchors_class_max - class_min_v) / (
                class_max_v - class_min_v))

        layer_one_list = []
        layer_one_list.append(scores)
        layer_one_list.append(classes)
        layer_one_list.append(scores * classes)
        for idx, one in enumerate(layer_one_list):
            layer_one = one.cpu().numpy()
            ret = ((layer_one - layer_one.min()) / (layer_one.max() - layer_one.min())) * 255
            ret = ret.astype(np.uint8)
            gray = ret[:, :, None]
            ret = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            if not show_one_layer:
                all_ret[j].append(cv2.resize(ret, (img_ori.shape[1], img_ori.shape[0])).copy())
            else:
                ret = cv2.resize(ret, (img_ori.shape[1], img_ori.shape[0]))
                show = ret * 0.8 + img_ori * 0.2
                show = show.astype(np.uint8)
                cv2.imshow(f"one_{SHOW_NAME[idx]}", show)
                cv2.imwrite('./cam_results/head' + str(j) + 'layer' + str(idx) + SHOW_NAME[idx] + ".jpg", show)
                # cv2.imshow(f"map_{SHOW_NAME[idx]}", ret)
                # saveFile = "G:/Wangmaosen/Code/YOLOv3_Sinatten_atten/YOLOv3_Sinatten_atten/hot_map"  # 保存文件的路径
                # cv2.imwrite(saveFile, show)
        if show_one_layer:
            cv2.waitKey(0)
    if not show_one_layer:
        for idx, one_type in enumerate(all_ret):
            map_show = one_type[0] / 3 + one_type[1] / 3 + one_type[2] / 3
            show = map_show * 0.8 + img_ori * 0.2
            show = show.astype(np.uint8)
            map_show = map_show.astype(np.uint8)
            cv2.imshow(f"all_{SHOW_NAME[idx]}", show)
#            cv2.imwrite('./cam_results/head_cont' + str(idx) + SHOW_NAME[idx] + ".jpg", show)
            saveFile = "G:/Wangmaosen/Code/YOLOv3_Sinatten_atten/YOLOv3_Sinatten_atten/hot_map"  # 保存文件的路径
            cv2.imwrite(saveFile, show)
            # cvSaveImage(path, image)
            # cv2.imshow(f"map_{SHOW_NAME[idx]}", map_show)
        cv2.waitKey(0)


ret = []
stride = [13, 26, 52]
yolo = YOLO()
path = 'img/truck3.jpg'
image = Image.open(path)
output_list = yolo.detect_image(image)
# print(output_list)
for i, f in enumerate(output_list):
    ret.append(f.reshape(1, 3, stride[i], stride[i], 8))

# features1 = torch.randn(1,3,13,13,10)
# features2 = torch.randn(1,3,26,26,10)
# features3 = torch.randn(1,3,52,52,10)

show_CAM(path, ret, 2)