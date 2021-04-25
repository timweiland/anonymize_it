#!/usr/bin/env python3
import cv2
import numpy as np
import layoutparser as lp

model = lp.Detectron2LayoutModel("lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config")


def detect_logos(img):
    layout = model.detect(img)
    figure_blocks = lp.Layout([b for b in layout if b.type == "ImageRegion"])
    rects = [
        (
            int(fig.block.x_1),
            int(fig.block.y_1),
            int(fig.block.x_2),
            int(fig.block.y_2),
        )
        for fig in figure_blocks
    ]
    return rects
