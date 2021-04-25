#!/usr/bin/env python3
import math
from io import StringIO
import numpy as np
import pandas as pd
import re
import cv2
import pytesseract


def preprocessing_pipeline(img):
    # Double image size
    image_resized = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Turn to grayscale
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    # Turn to black/white
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

    return thresh


def extract_pars(block_frame):
    pars = block_frame.par_num.unique()
    par_list = []
    for par in pars:
        par_list.append(block_frame[block_frame.par_num == par])
    return par_list


def extract_lines(par_frame):
    lines = par_frame.line_num.unique()
    line_list = []
    for line in lines:
        line_list.append(par_frame[par_frame.line_num == line])
    return line_list


def line_to_text(line_frame):
    return " ".join(line_frame.text)


def postprocess_text(text):
    processed = text.lower()
    processed = re.sub(r"[^A-Za-z0-9 ]+", "", processed)
    return processed


def get_block(dframe, num):
    return dframe[dframe.block_num == num]


def rows_to_rect(frame, img_width, img_height):
    if len(frame) <= 0:
        return (0, 0, 0, 0)
    min_left = np.inf
    min_top = np.inf
    max_left = 0
    max_top = 0
    for _, row in frame.iterrows():
        start_point = (row.left, row.top)
        end_point = (start_point[0] + row.width, start_point[1] + row.height)
        if row.width > 0.5 * img_width or row.height > 0.5 * img_height:
            continue
        min_left = min(min_left, start_point[0])
        max_left = max(max_left, end_point[0])
        min_top = min(min_top, start_point[1])
        max_top = max(max_top, end_point[1])
    if min_left == np.inf or min_top == np.inf or max_left == 0 or max_top == 0:
        return (0, 0, 0, 0)
    return (min_left, min_top, max_left, max_top)


def add_2_blocks_func(dframe, img_width, img_height, line):
    block_num = line.block_num.iloc[0]
    block_1 = get_block(dframe, block_num + 1)
    block_2 = get_block(dframe, block_num + 2)
    return [
        rows_to_rect(block_1, img_width, img_height),
        rows_to_rect(block_2, img_width, img_height),
    ]


def add_1_blocks_func(dframe, img_width, img_height, line):
    block_num = line.block_num.iloc[0]
    block_1 = get_block(dframe, block_num + 1)
    return [rows_to_rect(block_1, img_width, img_height)]


def add_cur_block_func(dframe, img_width, img_height, line):
    block_num = line.block_num.iloc[0]
    block = get_block(dframe, block_num)
    rect = rows_to_rect(block, img_width, img_height)
    return [rect]


def add_cur_par_func(dframe, img_width, img_height, line):
    block_num = line.block_num.iloc[0]
    par_num = line.par_num.iloc[0]
    block = get_block(dframe, block_num)
    par = block[block.par_num == par_num]
    return [rows_to_rect(par, img_width, img_height)]


def add_cur_line_func(dframe, img_width, img_height, line_frame):
    return [rows_to_rect(line_frame, img_width, img_height)]


def add_form_rect(dframe, img_width, img_height, line):
    h_line = line.h_line_idx.iloc[0]
    v_line = line.v_line_idx.iloc[0]
    if not h_line or not v_line:
        return []
    rect_rows = dframe[(dframe.h_line_idx == h_line) & (dframe.v_line_idx == v_line)]
    return [rows_to_rect(rect_rows, img_width, img_height)]


def keyword_substr(text, keyword):
    return keyword in text


def keyword_isword(text, keyword):
    if keyword in text.split():
        print("found keyword " + keyword)
    return keyword in text.split()


def get_blacken_rects(dframe, img_width, img_height):
    blacken_rects = []
    par_rects = []
    keyword_funcs = [
        ("kunde", keyword_substr, add_2_blocks_func),
        ("telefon", keyword_isword, add_cur_line_func),
        ("telefax", keyword_isword, add_cur_line_func),
        ("fax", keyword_isword, add_cur_line_func),
        ("tel", keyword_isword, add_cur_line_func),
        ("fahrzeug", keyword_isword, add_1_blocks_func),
        ("gmbh", keyword_isword, add_cur_block_func),
    ]
    block_nums = dframe.block_num.unique()
    for block_num in block_nums:
        cur_block = get_block(dframe, block_num)
        pars = extract_pars(cur_block)
        for par in pars:
            par_rects.append(rows_to_rect(par))
            lines = extract_lines(par)
            for line in lines:
                line = line[line.text.notna()]
                text = line_to_text(line)
                text = postprocess_text(text)
                for keyword, keyword_check, func in keyword_funcs:
                    if keyword_check(text, keyword):
                        blacken_rects += func(dframe, img_width, img_height, line)
    return blacken_rects, par_rects


def extractLines(img):
    # Preprocessing: Grayscale + Otsu
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Detect long horizontal lines
    horizontal_lines = []
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )
    cnts = cv2.findContours(
        detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x_min = np.inf
        x_max = 0
        y_avg = 0
        for point in c:
            x = point[0][0]
            y = point[0][1]
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_avg += y
        y_avg /= len(c)
        y_avg = int(y_avg)
        start = (x_min, y_avg)
        end = (x_max, y_avg)
        horizontal_lines.append((start, end))

    # Detect long vertical lines
    vertical_lines = []
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_vertical = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2
    )
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        y_min = np.inf
        y_max = 0
        x_avg = 0
        for point in c:
            y = point[0][1]
            x = point[0][0]
            y_min = min(y_min, y)
            y_max = max(y_max, y)
            x_avg += x
        x_avg /= len(c)
        x_avg = int(x_avg)
        start = (x_avg, y_min)
        end = (x_avg, y_max)
        vertical_lines.append((start, end))

    return horizontal_lines, vertical_lines


def contained_horizontal(line, rect):
    return rect[0][0] < line[1][0] and rect[1][0] > line[0][0]


def contained_vertical(line, rect):
    return rect[0][1] < line[1][1] and rect[1][1] > line[0][1]


def scale_rect(rect):
    return (int(rect[0] / 2), int(rect[1] / 2), int(rect[2] / 2), int(rect[3] / 2))


def classify_rects(df, horizontal_lines, vertical_lines):
    h_line_idxs = []
    v_line_idxs = []
    for _, row in df.iterrows():
        rect_start = (int(row.left / 2), int(row.top) / 2)
        rect_end = (int((row.left + row.width) / 2), int((row.top + row.height) / 2))
        rect = (rect_start, rect_end)
        h_y_max = 0
        h_line_max_idx = None
        for i, line in enumerate(horizontal_lines):
            y = line[0][1]
            if contained_horizontal(line, rect) and y < rect_end[1] and y > h_y_max:
                h_y_max = y
                h_line_max_idx = i
        h_line_idxs.append(h_line_max_idx)

        v_x_max = 0
        v_line_max_idx = None
        for i, line in enumerate(vertical_lines):
            x = line[0][0]
            if contained_vertical(line, rect) and x < rect_end[0] and x > v_x_max:
                v_x_max = x
                v_line_max_idx = i
        v_line_idxs.append(v_line_max_idx)
    df["h_line_idx"] = h_line_idxs
    df["v_line_idx"] = v_line_idxs
    return df


def get_blacken_rects_new(dframe, img_width, img_height):
    blacken_rects = []
    keyword_funcs = [
        ("tel", keyword_isword, add_form_rect),
        ("kunde", keyword_isword, add_form_rect),
        ("gmbh", keyword_isword, add_cur_block_func),
        ("tel", keyword_isword, add_cur_line_func),
        ("fax", keyword_isword, add_cur_line_func),
        ("email", keyword_isword, add_cur_line_func),
        ("emall", keyword_isword, add_cur_line_func),
        ("www", keyword_substr, add_cur_line_func),
        ("werksbeauftragter", keyword_isword, add_form_rect),
        ("abholer", keyword_isword, add_form_rect),
        ("baustellennummer", keyword_substr, add_form_rect),
        ("webseite", keyword_isword, add_cur_line_func),
        ("website", keyword_isword, add_cur_line_func),
    ]
    block_nums = dframe.block_num.unique()
    for block_num in block_nums:
        cur_block = get_block(dframe, block_num)
        pars = extract_pars(cur_block)
        for par in pars:
            lines = extract_lines(par)
            for line in lines:
                line = line[line.text.notna()]
                text = line_to_text(line)
                text = postprocess_text(text)
                for keyword, keyword_check, func in keyword_funcs:
                    if keyword_check(text, keyword):
                        blacken_rects += func(dframe, img_width, img_height, line)
    return blacken_rects


def detect_text(img):
    horizontal_lines, vertical_lines = extractLines(img)

    pipelined_img = preprocessing_pipeline(img)

    img_rgb = cv2.cvtColor(pipelined_img, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = img_rgb.shape

    # Run Tesseract OCR over the document and read result into pandas frame
    tess_data = StringIO(pytesseract.image_to_data(img_rgb))
    df = pd.read_csv(tess_data, sep="\t")

    # Cleanup
    df = df.replace(r"^\s*$", np.nan, regex=True)
    block_counts = df.groupby("block_num")["text"].count()
    non_empty_blocks = block_counts[block_counts > 0].index
    df_filled_blocks = df[df.block_num.isin(non_empty_blocks)]

    df_postprocessed = classify_rects(
        df_filled_blocks, horizontal_lines, vertical_lines
    )

    rects = get_blacken_rects_new(df_postprocessed, img_width, img_height)
    rects = list(map(scale_rect, rects))

    return rects
