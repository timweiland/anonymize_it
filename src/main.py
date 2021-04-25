#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
from anonymization import anonymize
from util import blacken_img


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Anonymizing delivery notes made easy!"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)

    return parser.parse_args()


def main():
    args = parse_arguments()
    input_path = Path(args.input)
    img = cv2.imread(str(input_path))
    rects = anonymize(img)
    blackened_img = blacken_img(img, rects)
    output_path = args.output
    cv2.imwrite(output_path, blackened_img)


if __name__ == "__main__":
    main()
