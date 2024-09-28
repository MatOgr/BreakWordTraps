import math

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def pth_processing(fp):
    class PreprocessInput(torch.nn.Module):
        def init(self):
            super(PreprocessInput, self).init()

        def forward(self, x):
            x = x.to(torch.float32)
            x = torch.flip(x, dims=(0,))
            x[0, :, :] -= 91.4953
            x[1, :, :] -= 103.8827
            x[2, :, :] -= 131.0912
            return x

    def get_img_torch(img):
        ttransform = transforms.Compose([transforms.PILToTensor(), PreprocessInput()])
        img = img.resize((224, 224), Image.Resampling.NEAREST)
        img = ttransform(img)
        img = torch.unsqueeze(img, 0)
        return img

    return get_img_torch(fp)


def norm_coordinates(normalized_x, normalized_y, image_width, image_height):
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)

    return x_px, y_px


def get_box(fl, w, h):
    idx_to_coors = {}
    for idx, landmark in enumerate(fl.landmark):
        landmark_px = norm_coordinates(landmark.x, landmark.y, w, h)

        if landmark_px:
            idx_to_coors[idx] = landmark_px

    x_min = np.min(np.asarray(list(idx_to_coors.values()))[:, 0])
    y_min = np.min(np.asarray(list(idx_to_coors.values()))[:, 1])
    endX = np.max(np.asarray(list(idx_to_coors.values()))[:, 0])
    endY = np.max(np.asarray(list(idx_to_coors.values()))[:, 1])

    (startX, startY) = (max(0, x_min), max(0, y_min))
    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

    return startX, startY, endX, endY


def display_emotion_prediction_box(
    img,
    box,
    label="",
    color=(128, 128, 128),
    txt_color=(255, 255, 255),
    line_width=2,
):
    """Display emotion details"""
    lw = line_width or max(round(sum(img.shape) / 2 * 0.003), 2)
    text2_color = (255, 0, 255)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, p1, p2, text2_color, thickness=lw, lineType=cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX

    tf = max(lw - 1, 1)
    text_fond = (0, 0, 0)
    text_width_2, text_height_2 = cv2.getTextSize(label, font, lw / 3, tf)
    text_width_2 = text_width_2[0] + round(((p2[0] - p1[0]) * 10) / 360)
    center_face = p1[0] + round((p2[0] - p1[0]) / 2)

    cv2.putText(
        img,
        label,
        (
            center_face - round(text_width_2 / 2),
            p1[1] - round(((p2[0] - p1[0]) * 20) / 360),
        ),
        font,
        lw / 3,
        text_fond,
        thickness=tf,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        label,
        (
            center_face - round(text_width_2 / 2),
            p1[1] - round(((p2[0] - p1[0]) * 20) / 360),
        ),
        font,
        lw / 3,
        text2_color,
        thickness=tf,
        lineType=cv2.LINE_AA,
    )
    return img


def display_FPS(img, text, margin=1.0, box_scale=1.0):
    img_h, img_w, _ = img.shape
    line_width = int(min(img_h, img_w) * 0.001)  # line width
    thickness = max(int(line_width / 3), 1)  # font thickness

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (0, 0, 0)
    font_scale = thickness / 1.5

    t_w, t_h = cv2.getTextSize(text, font_face, font_scale, None)[0]

    margin_n = int(t_h * margin)
    sub_img = img[
        0 + margin_n : 0 + margin_n + t_h + int(2 * t_h * box_scale),
        img_w - t_w - margin_n - int(2 * t_h * box_scale) : img_w - margin_n,
    ]

    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255

    img[
        0 + margin_n : 0 + margin_n + t_h + int(2 * t_h * box_scale),
        img_w - t_w - margin_n - int(2 * t_h * box_scale) : img_w - margin_n,
    ] = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

    cv2.putText(
        img=img,
        text=text,
        org=(
            img_w - t_w - margin_n - int(2 * t_h * box_scale) // 2,
            0 + margin_n + t_h + int(2 * t_h * box_scale) // 2,
        ),
        fontFace=font_face,
        fontScale=font_scale,
        color=font_color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
        bottomLeftOrigin=False,
    )

    return img
