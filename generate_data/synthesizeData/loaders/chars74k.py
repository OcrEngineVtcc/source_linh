import glob
import os

import cv2
import numpy as np
from skimage.draw import line_aa

from dataloader.utils.preprocessing import skeletonize_image

def parse_string_to_list(str_representation):
    """Read the string representation of row and col

    # Arguments
        str_representation [str]: the string representation

    # Returns
        [list of list of floats]: the representation
    """
    str_representation = str_representation.replace(';\n', ',')
    str_representation = str_representation.replace('\n', ',')
    str_representation = str_representation.replace('{', '[')
    str_representation = str_representation.replace('}', ']')
    str_representation = str_representation[7:-1]

    list_representation = eval(str_representation)

    return list_representation


def read_m_file(filename):
    """Read the trajectory m file

    # Arguments
        filename [str]: the path to m file

    # Returns:
        [list of list of floats]: each float contains the y point
        [list of list of floats]: each float contains the x point
    """
    with open(filename, 'r') as f:
        data = f.read()

    rows, cols, _ = data.split('\n\n')
    rows = parse_string_to_list(rows)
    cols = parse_string_to_list(cols)

    return rows, cols


def draw_strokes_on_canvas(rows, cols):
    """Draw the stroke on fixed-size canvas

    # Arguments
        rows [list of floats]: list of y coordinates
        cols [list of floats]: list of x coordinates

    # Returns
        [np array]: the numpy image
    """
    canvas = np.zeros((900, 1200), dtype=np.uint8)

    for each_stroke_idx in range(len(rows)):
        row = np.array(rows[each_stroke_idx], dtype=np.int32)
        col = np.array(cols[each_stroke_idx], dtype=np.int32)
        canvas[row, col] = 255

        for _idx in range(len(row)):
            if _idx == len(row) - 1:
                continue
            rr, cc, val = line_aa(
                row[_idx], col[_idx], row[_idx+1], col[_idx+1])
            canvas[rr, cc] = val * 255

    kernel =  np.ones((2, 2), dtype=np.uint8)
    canvas = cv2.dilate(canvas, kernel, iterations=5)

    return (255 - canvas).astype(np.uint8)


def draw_image_from_file(filename, out_folder=None):
    """Draw the image from .m file

    # Arguments
        filename [str]: the path to .m file
        out_folder [str]: the path to out folder. If is None, don't save

    # Returns
        [np array]: the numpy image (grayscale image)
    """
    rows, cols = read_m_file(filename)
    image = draw_strokes_on_canvas(rows, cols)

    if out_folder is not None:
        os.makedirs(out_folder, exist_ok=True)
        filename = os.path.basename(filename)
        filename, extension = os.path.splitext(filename)
        cv2.imwrite(
            os.path.join(out_folder, '{}.png'.format(filename)),
            image)

    return image

