# Load the TUAT dataset
# =============================================================================
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import xml.etree.ElementTree as ET

import cv2
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-folder', help='Input folder')
parser.add_argument('-o', '--output-folder', help='Png destination folder')
parser.add_argument('-f', '--filename', help='Work with a single file')


INKML_NAMESPACE = '{http://www.w3.org/2003/InkML}'
XML_NAMESPACE = '{http://www.w3.org/XML/1998/namespace}'

def show_image(np_img, bgr=False):
    """Show the image using PIL's Image.show()

    # Arguments
        np_img [np array]: the image in numpy array format
        bgr [bool]: applied only when np_img.shape == 3, denoting supplied
            color channels as BGR
    """
    if len(np_img.shape) == 3 and bgr:
        np_img = np.flip(np_img, 2)

    Image.fromarray(np_img).show()

def inkml_name(name):
    """Return the proper InkML name

    # Arguments
        name [str]: the identifier

    # Returns
        [str]: proper inkml name
    """
    return INKML_NAMESPACE + name

def xml_name(name):
    """Return the proper XML name

    # Arguments
        name [str]: the identifier

    # Returns
        [str]: proper xml name
    """
    return XML_NAMESPACE + name

def normalize_row(text):
    """Normalize single row (currently remove '\n' character)

    # Arguments
        text [str]: a row of text to normalize

    # Returns
        [str]: the text string without \n character
    """
    return text.replace('\n', '')

def get_transcription(annotations):
    """Get the character

    The char should reside in <annotation> that has type == 'transcription'.

    # Arguments
        annotations [list of xml elements]: list of annotation elements

    # Returns
        [str]: the character

    # Raises
        ValueError: when there isn't any character
    """
    for each_annotation in annotations:
        if each_annotation.get('type') == 'transcription':
            return normalize_row(each_annotation.text)

    raise ValueError('the list of annotations does not have transcription')

def get_strokes_id(strokes):
    """Extract the id from the list of strokes

    # Arguments
        strokes [list of xml elements]: list of <traceView />

    # Returns
        [list of str]: list of id
    """
    result = []
    for each_stroke in strokes:
        result.append(each_stroke.get('traceDataRef').split('#')[1])

    return result


def get_coordinate(coord_string):
    """Get the X, Y cordinate from coordinate string

    It should get from `a x, b y, c z` -> `[a, b, c], [x, y, z]`.

    # Arguments
        coord_string [str]: the coordinate string

    # Returns
        [list of numbers]: list of y coordinates
        [list of numbers]: list of x coordinates
    """
    x, y = [], []
    coords = coord_string.split(',')

    for each_coord in coords:
        temp_x, temp_y = each_coord.split(' ')
        x.append(int(temp_x))
        y.append(int(temp_y))

    return x, y


def generate_image_from_strokes(strokes):
    """Generate numpy image from the strokes data

    # Arguments
        strokes [list of strs]: list of strokes coordinates

    # Returns
        [np array]: the numpy image generated from the strokes
    """
    x_coords = []
    y_coords = []

    for each_stroke in strokes:
        temp_x, temp_y = get_coordinate(each_stroke)
        x_coords += temp_x
        y_coords += temp_y

    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)

    x_coords -= np.amin(x_coords)
    y_coords -= np.amin(y_coords)

    result = np.zeros((np.max(y_coords)+1, np.max(x_coords)+1), dtype=np.uint8)
    result[y_coords, x_coords] = 1

    return result


def load_each_inkml_file(filename):
    """Load each inkml file into numpy images.

    # Arguments
        filename [str]: a path to inkml file

    # Returns
        [tuple of np array + label]: the full image of the text
        [list of tuples of [np array + label]]: list of images, each image 
            is a character
    """
    tree = ET.parse(filename)

    # get all the strokes
    temp_strokes = tree.findall(inkml_name('trace'))
    strokes = {}
    for each_stroke in temp_strokes:
        _id = each_stroke.get(xml_name('id'))
        strokes[_id] = normalize_row(each_stroke.text)

    # which set of strokes belongs to which characters
    temp_chars = tree.findall(inkml_name('traceView'))
    chars = []
    for each_char in temp_chars:
        char = get_transcription(each_char.findall(inkml_name('annotation')))
        chars.append(
            (char, get_strokes_id(each_char.findall(inkml_name('traceView'))))
        )

    return chars, strokes


def dilate_image(image):
    kernel = np.ones((10,10), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)



from qutils import show_image
if __name__ == '__main__':
    import glob
    files = glob.glob('/Users/ducprogram/cinnamon/data_gen/HANDS-kondate-14-09-01/Kondate-jp.InkML.utf-8/*.iml')
    # filename = ''
    FOLDER = '/Users/ducprogram/Desktop/tuat-samples'
    FOLDER = '/Users/ducprogram/Desktop/tuat-samples-converted'
    for _idx, filename in enumerate(files):
        if _idx <= 66:
            continue
        try:
            print('Generating {}'.format(filename))
            fn = os.path.splitext(os.path.basename(filename))[0]
            chars, strokes = load_each_inkml_file(filename)
            # print('There are {} chars'.format(len(chars)))
            # for _idx in range(len(chars)):
            #     char0, stroke_list = chars[_idx]
            #     result = generate_image_from_strokes([strokes[_] for _ in stroke_list])
            #     result = (result * 255).astype(np.uint8)
            #     # show_image(result * 255)
            #     cv2.imwrite(os.path.join(FOLDER, '{}_{}.png'.format(str(_idx), char0)), result)
            all_chars = ''
            all_strokes = []
            for each_char, each_strokes in chars:
                all_chars += each_char
                all_strokes += [strokes[_] for _ in each_strokes]

            result = generate_image_from_strokes(all_strokes)
            result = (result * 255).astype(np.uint8)
            result = dilate_image(result)
            # result = 255 - result
            # show_image(result)
            cv2.imwrite(os.path.join(FOLDER, '{}.png'.format(fn)), result)
        except ValueError:
            print('Skip generating {}'.format(filename))

