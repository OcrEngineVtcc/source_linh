# Quick utility functions to help with development
# =============================================================================
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import glob
import json
import os
import math
import uuid

import cv2
import matplotlib

if os.name == 'posix' and 'DISPLAY' not in os.environ:
    # 'agg' backend for headless server (not connected to any display)
    matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import truncnorm


def check_allowed_char_version(filepath):
    """Check the allowed char version

    # Arguments
        filepath [str]: the filepath

    # Returns
        [bool]: True if new version, False otherwise
    """
    with open(filepath, 'r') as f_in:
        chars = [each_line.strip() for each_line in f_in.readlines()]

    if len(chars) < 1:
        raise ValueError('`filepath` is an empty file')

    for each_char in chars:
        if each_char == '':
            continue

        try:
            int(each_char)
        except ValueError:
            return True

    return False


def dump_allowed_char_file(filepath, char_list):
    """Dump the list of characters into allowed_char text file

    # Arguments
        filepath [str]: the path to allowed_char
        char_list [list of str]: the list of characters
    """
    char_list.sort()
    with open(filepath, 'w') as f_out:
        f_out.write('\n'.join(char_list))


def stack_images_vertically(*list_of_images):
    """Stack images vertically

    # Arguments
        list_of_images [list of np array]: the images to stack

    # Returns
        [np array]: 1 image that contains all images in the input
    """
    if (len(list_of_images) == 1 and type(list_of_images[0]) == list):
        list_of_images = list_of_images[0]
    elif len(list_of_images) == 0:
        raise AttributeError(
            "stack_images_vertically should contain at least 1 argument")


    width = max([each_image.shape[1] for each_image in list_of_images])
    height = (sum([each_image.shape[0] for each_image in list_of_images])
        + 2 * (len(list_of_images) - 1))
    result = np.zeros((height, width), dtype=np.uint8)

    current_position_y = 0
    for each_image in list_of_images:
        image_height, image_width = each_image.shape
        result[current_position_y:current_position_y + image_height,
               0:image_width] = each_image
        current_position_y = current_position_y + 2 + image_height

    return result


def get_truncated_normal(mean=0, std=1, low=-5, high=5, n_elements=1):
    """Get elements within a Gaussian mean

    # Arguments
        mean [float]: the mean of the distribution
        std [float]: the standard deviation of the distribution
        low [float]: the minimum value
        high [float]: the maximum value
        n_elements [int]: the number of elements to return

    # Returns
        [np array]: array of elements, each sampled from Guassian distribution
        or [int]: if n_elements = 1
    """
    if n_elements == 1:
        return truncnorm(
            (low - mean) / std, (high - mean) / std, loc=mean, scale=std).rvs()

    return truncnorm(
        (low - mean) / std, (high - mean) / std,
        loc=mean, scale=std).rvs(n_elements)


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


def show_images(image_list, label_list=None, max_columns=10, notebook=False):
    """Show list of images

    # Arguments
        image_list [list of np array]: list of images
        label_list [list of strings]: list of labels
        max_columns [int]: the maximum number of images to view side-by-side
        notebook [bool]: whether this function is called inside a notebook
    """
    if label_list is not None:
        if not isinstance(label_list, list):
            raise ValueError('`label_list` should be list')
        if len(image_list) != len(label_list):
            raise ValueError(
                '`image_list` should have the same length with `label_list`')

    columns = min(max_columns, len(image_list))
    rows = math.ceil(len(image_list) / columns)

    plt.figure(figsize=(20,10))
    for _idx, each_img in enumerate(image_list):
        plt.subplot(rows, columns, _idx+1)
        if label_list is not None:
            plt.title(label_list[_idx])
        plt.imshow(each_img, cmap='gray')

    if not notebook:
        plt.show()


def show_image_ascii(image, bgr=False):
    """Show the image in ascii

    # Arguments
        image [str or np array]: if type string, then it should be a path
            to the image
        bgr [bool]: whether the image has color channels BGR
    """
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    else:
        if len(image.shape) == 3:
            mode = cv2.COLOR_RGB2GRAY
            if bgr:
                mode = cv2.COLOR_BGR2GRAY
            image = cv2.cvtColor(image, mode)

    # Calculate image width and height
    height, width = image.shape
    columns, _ = os.get_terminal_size()
    rows = int(columns * height / width)

    # Transform and binarize image
    image = cv2.resize(image, (columns, rows), interpolation=cv2.INTER_CUBIC)
    image = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Draw image into string
    new_image = ['_' * (columns)]
    for each_row in image:
        col = []
        for each_col in each_row:
            if each_col == 0:
                col.append('#')
            else:
                col.append(' ')
        new_image.append(''.join(col))
    new_image.append('_' * (columns))

    # Print out
    for each_row in new_image:
        print(each_row)


def enhance_frequency_charsets(in_charset_path, out_charset_path=None):
    """Add latins and alphabets characters to the frequency charset

    # Arguments
        in_charset_path [str]: the path to charset text file
        out_charset_path [str]: the path to save the output charset text file.
            If None, it would replace the file in `in_charset_path`

    # Returns
        [list of ints]: the charsets
    """
    if out_charset_path is None:
        out_charset_path = in_charset_path

    alpha_nums = set(
        [char_code for char_code in range(ord('a'), ord('z')+1)]
        + [char_code for char_code in range(ord('A'), ord('Z')+1)]
        + [char_code for char_code in range(ord('0'), ord('9')+1)]
    )

    in_chars = set(np.loadtxt(in_charset_path, dtype=int))
    out_chars = list(in_chars.union(alpha_nums))
    out_chars.sort()

    np.savetxt(out_charset_path, out_chars, fmt='%d')

    return out_chars


def crop_polygon_from_image(image, xs, ys, cval=None):
    """Crop a polygon from an image

    # Arguments
        image [np array]: the image
        xs [list of ints]: list of x coordinates
        ys [list of ints]: list of y coordinates
        cval [int]: the fill value for non content. If None, it will be
            determined from the background

    # Returns
        [np array]: the cropped image
    """
    max_xs, min_xs = np.max(xs), np.min(xs)
    max_ys, min_ys = np.max(ys), np.min(ys)

    if cval is None:
        blur_image = cv2.medianBlur(image[min_ys:max_ys+1, min_xs:max_xs+1], 5)
        pixels, counts = np.unique(blur_image, return_counts=True)
        cval = pixels[np.argmax(counts)]

    polygons = list(zip(xs, ys))
    mask = Image.new('RGB', (image.shape[1], image.shape[0]), 0)
    ImageDraw.Draw(mask).polygon(polygons, outline=(1,1,1), fill=(1,1,1))
    mask = np.array(mask)

    result = (mask * image)[min_ys:max_ys+1, min_xs:max_xs+1].astype(np.uint8)
    background = (1 - mask[min_ys:max_ys+1, min_xs:max_xs+1]) * cval

    result = cv2.add(result, background.astype(np.uint8))

    return result


def split_image_by_points(image, points):
    """Split the image into smaller images

    # Argument
        image [2D-3D np array]: the image
        points [2D np array]: an array of dividing points with the following
            structure nparray([[x0,y0], [x1,y1],...])

    # Returns
        [list of np arrays]: the list of cutted images by the
    """
    points = points[np.argsort(points[:,0])]
    images = []
    for _idx, each_point in enumerate(points):
        if _idx == 0:
            last_point = 0

        images.append(image[:, last_point:each_point[0]])
        last_point = each_point[0]

    images.append(image[:, last_point:])
    return images


def extract_via_json_bounding_boxes_single_image(image_path, label_path, 
    out_path=None, name_key='placeholder'):
    """Extract via json bounding box

    @NOTE: This method does not handle splitting image by points.

    # Arguments
        image_path [str]: path to the image
        label_path [str/dict]: path to the label or a dictionary
        out_path [str]: path to folder to save extracted images
        name_key [str]: the identifier to filename. If the `name_key` is one
            of the labels, then the value of that label will be used; otherwise
            `name_key` itself will be used as filename identifier

    # Returns
        [list of objects]: list of extracted images with labels
    """
    if not os.path.isfile(image_path):
        raise ValueError('invalid image {}'.format(image_path))

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if isinstance(label_path, str):
        if not os.path.isfile(label_path):
            raise ValueError('invalid label {}'.format(label_path))
        with open(label_path, 'r') as in_file:
            labels = json.load(in_file)
    elif isinstance(label_path, dict):
        labels = label_path
    else:
        raise ValueError('unrecognizable label type {}'
            .format(type(label_path)))

    filename = os.path.basename(image_path)
    filesize = os.path.getsize(image_path)

    regions = labels['{}{}'.format(filename, filesize)]['regions']
    results = []
    for each_region in regions:
        shape_object = each_region['shape_attributes']
        if (shape_object['name'] == 'polyline' 
            or shape_object['name'] == 'polygon'): 
            cropped_image = crop_polygon_from_image(image,
                shape_object['all_points_x'], shape_object['all_points_y'])
        elif shape_object['name'] == 'rect':
            x, y = shape_object['x'], shape_object['y']
            width, height = shape_object['width'], shape_object['height']
            cropped_image = image[y:y+height+1, x:x+width+1]
        elif shape_object['name'] == 'point':
            # can use cv2.pointPolygonTest(contour, point, False)
            # contour is an array of (x,y) coordinates
            continue
        else:
            raise ValueError('unrecognizable region type {}'
                .format(shape_object['name']))

        results.append({
            'region_attributes': each_region['region_attributes'],
            'shape_attributes': shape_object,
            'image': cropped_image
        })

    if out_path is not None:
        labels = {}
        for each_result in results:
            filename = '{}_{}.png'.format(
                each_result['region_attributes'].get(name_key, name_key),
                uuid.uuid4().hex)
            cv2.imwrite(os.path.join(out_path, filename), each_result['image'])
            labels[filename] = each_result['region_attributes']

        with open(os.path.join(out_path, 'labels.json'), 'w') as out_file:
            json.dump(labels, out_file)

    return results


def extract_via_json_bounding_boxes_folder(image_folder, label_folder,
    out_path=None, name_key='placeholder'):
    """Extract via json bounding box

    # Arguments
        folder_path [str]: path to the image
        label_path [str/dict]: path to the label or a dictionary
        out_path [str]: path to folder to save extracted images
        name_key [str]: the identifier to filename. If the `name_key` is one
            of the labels, then the value of that label will be used; otherwise
            `name_key` itself will be used as filename identifier

    # Returns
        [list of objects]: list of extracted images with labels
    """
    # get the images
    EXTENSIONS = [
        'png', 'jpg', 'jpeg',
        'PNG', 'JPG', 'JPEG']
    image_files = []
    for ext in EXTENSIONS:
        image_files += glob.glob(os.path.join(image_folder, '*.{}'.format(ext)))

    # returns the result
    labels = {}
    results = []
    for each_image in image_files:
        # get the json path (json name is supposed to be the same as image name)
        json_filename = os.path.splitext(os.path.basename(each_image))[0]
        json_path = os.path.join(label_folder, '{}.json'.format(json_filename))

        # get the individual regions
        result = extract_via_json_bounding_boxes_single_image(
            each_image, json_path)
        results += result

        if out_path is not None:
            for idx, each_result in enumerate(result):
                filename = '{}_{}_{}.png'.format(
                    json_filename,
                    idx,
                    each_result['region_attributes'].get(name_key, name_key))
                cv2.imwrite(os.path.join(out_path, filename), each_result['image'])
                labels[filename] = each_result['region_attributes']

    if out_path is not None:
        with open(os.path.join(out_path, 'labels.json'), 'w') as out_file:
            json.dump(labels, out_file)

    return results


def dump_json(obj, filepath, sort_keys=False):
    """Dump the object into json

    # Arguments
        obj [dict]: the dictionary to be dumped into json
        filepath [str]: the filepath
        sort_keys [bool]: whether to sort the keys
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4, separators=(',', ': '),
                  sort_keys=sort_keys)


ENDING_CHARS = set('.,!]})?%、。:）」')
ALPHANUMS =set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
def split_by_length(text_line, length):
    """Split a text line by length

    The splitted text should satisfy the following requirements:
        - Should not have ending characters begin a new lines: .,!]})?%
        - Should not cut through an alphabet word or number or combination of
        alphabet and number (e.g. hello -> hel lo; H20 -> H, 20)
        - Should prefer the nearest space ' ' to cut if possible
        - Should remove trailing space after the cutting

    This function will balance the cutting requirements above with the set
    `length`.

    # Arguments
        text_line [str]: a string of text line
        length [int]: the desired average length
    """
    cut_indices = []
    stop_index = length
    half_span = int(length / 4)

    while stop_index < len(text_line):
        span_begin_index = stop_index - half_span
        span_end_index = stop_index + half_span
        search_span = text_line[span_begin_index:span_end_index]

        # looking for nearest space ' ' to the `stop_index`. If such case is
        # found, then we don't need to care for other requirements.
        spaces = []
        for _idx, each_char in enumerate(search_span):
            if each_char == ' ':
                spaces.append(_idx)
        if len(spaces) > 0:
            if len(spaces) == 1:
                stop_index = spaces[0] + span_begin_index
            elif len(spaces) > 1:
                min_difference = float('inf')
                desired_idx = None
                for _idx in spaces:
                    if abs(_idx - stop_index) < min_difference:
                        min_difference = abs(_idx - stop_index) < min_difference
                        desired_idx = _idx
                stop_index = desired_idx + span_begin_index

            cut_indices.append(stop_index)
            stop_index += length
            continue

        # if no space is found, then make sure that ending characters not on
        # new line
        if text_line[stop_index] in ENDING_CHARS:
            stop_index += 1
            if stop_index >= len(text_line):
                break
            else:
                cut_indices.append(stop_index)
                stop_index += length
                continue

        # should not cut through an alpha-numerical character
        if text_line[stop_index] in ALPHANUMS:
            for _idx in range(half_span):
                if (text_line[stop_index - _idx] not in ALPHANUMS and
                    text_line[stop_index - _idx] not in ENDING_CHARS):
                    stop_index = stop_index - _idx
                    break
                if (stop_index + _idx < len(text_line) and
                    text_line[stop_index + _idx] not in ALPHANUMS and
                    text_line[stop_index + _idx] not in ENDING_CHARS):
                    stop_index = stop_index + _idx
                    break

        cut_indices.append(stop_index)
        stop_index += length

    segments = []
    start_idx = 0
    for _idx, each_index in enumerate(cut_indices):
        text = text_line[start_idx:each_index].strip()
        segments.append(text)
        start_idx = each_index
    text = text_line[start_idx:].strip()
    segments.append(text)

    return segments


def break_down_long_text(in_path, out_folder, mean_char=15):
    """Break text file with long lines into smaller lines

    # Arguments
        in_path [str]: the path to text or folder containing text
        out_folder [str]: the output folder
        mean_char [int]: the average amount of chars (fuzzy)

    # Returns
        [list of str]: list of broken down strings
    """
    if os.path.isfile(in_path):
        in_path = [in_path]
    else:
        in_path = glob.glob(os.path.join(in_path, '*.txt'))

    for each_text in in_path:
        out_lines = []
        filename = os.path.basename(each_text)
        filename, _ = os.path.splitext(filename)
        out_path = os.path.join(out_folder, 
                                '{}_{}.txt'.format(filename, mean_char))

        with open(each_text, 'r') as in_file:
            in_lines = in_file.readlines()

        for _idx, each_line in enumerate(in_lines):

            if _idx % 1000 == 0:
                print('Working on line number {}'.format(_idx + 1))
            line = each_line.strip()

            if len(line) == 0:
                continue

            if len(line) < mean_char * 1.5:
                out_lines.append(line)
                continue

            splitted_lines = split_by_length(
                line,
                int(get_truncated_normal(mean=mean_char, std=3,
                                         low=1, high=mean_char*2))
            )
            out_lines += splitted_lines

        print('Processed into {} lines from {}'.format(
            len(out_lines), each_text))

        with open(out_path, 'w') as out_file:
            out_file.write('\n'.join(out_lines))

    return out_lines





