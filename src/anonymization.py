#!/usr/bin/env python3
from text_detection import detect_text
from logo_detection import detect_logos


def anonymize(img):
    """
    Return a list of rectangles that, when blackened in the image, will anonymize private data.
    """
    rects_logos = detect_logos(img)
    rects_text = detect_text(img)

    total_rects_set = set(rects_logos)
    total_rects_set.update(rects_text)
    return list(total_rects_set)
