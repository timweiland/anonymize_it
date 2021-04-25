#!/usr/bin/env python3
import cv2


def blacken_img_rect(img, left, top, end_x, end_y):
    start_point = (left, top)
    end_point = (end_x, end_y)
    img = cv2.rectangle(img, start_point, end_point, (0, 0, 0), -1)
    return img


def blacken_img(img, rects):
    for rect in rects:
        img = blacken_img_rect(img, rect[0], rect[1], rect[2], rect[3])
    return img
