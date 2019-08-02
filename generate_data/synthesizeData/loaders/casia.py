# Load CASIA offline character and paragraph databases
# @NOTE: the CASIA datasets are distributed in .alz fileformat, use unalz to
# unpack it.
# @author: _john
# =============================================================================
import os
import pdb
import struct
from codecs import decode

import numpy as np
from PIL import Image


def decode_gb(raw_label, standard):
    """Decode gb code into unicode

    # Arguments
        raw_label [tuple of 2 byte strs]: the component byte strings

    # Returns
        [str]: the Unicode string
    """
    return decode(raw_label[0] + raw_label[1], encoding=standard)


def load_character_gnt(file_path):
    """
    Load characters and images from a given GNT file.

    # Arguments
        file_path [str]: the file path to load.

    # Returns
        [list of np array]: list of character images
        [list of str]: list of labels
        [list of tuples of nd array, str]: images that charset cannot be decoded
    """

    images = []
    labels = []
    bad_instances = []

    with open(file_path, 'rb') as f:
        while True:
            packed_length = f.read(4)
            if packed_length == b'':
                break

            _ = struct.unpack('<I', packed_length)[0]
            raw_label = struct.unpack('>cc', f.read(2))

            width = struct.unpack('<H', f.read(2))[0]
            height = struct.unpack('<H', f.read(2))[0]
            photo_bytes = struct.unpack('{}B'.format(height * width), f.read(height * width))
            image = np.array(photo_bytes).reshape(height, width).astype(np.uint8)
            try:
                label = decode_gb(raw_label, 'gb2312')
            except:
                bad_instances.append((image, raw_label))
                continue

            images.append(image)
            labels.append(label)

    return images, labels, bad_instances


def load_paragraph_dgr(file_path):
    """Load the paragraph files

    According to the documentation, 0xff is gabbage.

    # Arguments
        file_path [str]: the file path to load

    # Returns
        [list of dict]: list of separated characters on the document
        [dict]: other document detail
    """

    file_size = os.stat(file_path).st_size
    lines = []

    with open(file_path, 'rb') as f_in:

        # get header information
        header = struct.unpack('<i', f_in.read(4))[0]
        illustration_size = header - 36
        format_code = struct.unpack('<8c', f_in.read(8))
        illustration = struct.unpack('<{}c'.format(illustration_size), f_in.read(illustration_size))
        code_type = struct.unpack('<20c', f_in.read(20))
        code_length = struct.unpack('<h', f_in.read(2))[0]
        bit_per_pixel = struct.unpack('<h', f_in.read(2))[0]

        # get document information
        image_height = struct.unpack('<i', f_in.read(4))[0]
        image_width = struct.unpack('<i', f_in.read(4))[0]
        line_number = struct.unpack('<i', f_in.read(4))[0]

        # get line information
        for _line_idx in range(line_number):
            each_line = []
            char_number = struct.unpack('<i', f_in.read(4))[0]

            # get word information
            for _char_idx in range(char_number):
                char_label = struct.unpack('<{}c'.format(code_length),
                                           f_in.read(code_length))
                char_top = struct.unpack('<h', f_in.read(2))[0]
                char_left = struct.unpack('<h', f_in.read(2))[0]
                char_height = struct.unpack('<h', f_in.read(2))[0]
                char_width = struct.unpack('<h', f_in.read(2))[0]
                char_image = struct.unpack(
                    '<{}B'.format(char_height * char_width),
                    f_in.read(char_height * char_width))
                char_image = np.array(char_image).reshape(
                    char_height, char_width).astype(np.uint8)

                try:
                    char_label = decode_gb(char_label, 'gb18030')
                    char_good = True
                except:
                    if char_label != (b'\xff', b'\xff'):
                        pdb.set_trace()
                    char_good = False

                each_line.append({
                    'label': char_label, 'top': char_top, 'left': char_left,
                    'height': char_height, 'width': char_width,
                    'good': char_good, 'image': char_image
                })

            lines.append(each_line)

        if f_in.tell() != file_size:
            print(':WARNING: {} size mismatch after extracting: {} - {} '
                  '(current stream vs file size)'
                  .format(file_path, f_in.tell(), file_size))

    return lines, {
        'format_code': format_code,
        'code_type': code_type,
        'illustration': illustration,
        'bit_per_pixel': bit_per_pixel,
        'image_height': image_height,
        'image_width': image_width
    }


def construct_line(characters, draw_bad=False):
    """Construct the line

    # Arguments
        characters [list of objects]: list of characters and information
        draw_bad [bool]: whether to draw bad characters

    # Returns
        [np array]: the line image
        [str]: the corresponding label
    """
    characters = sorted(characters, key=lambda obj: obj['left'])

    min_top, max_bottom = float('inf'), float('-inf')
    for each_character in characters:
        if draw_bad is False and each_character['good'] is False:
            continue

        if each_character['top'] < min_top:
            min_top = each_character['top']

        if each_character['top'] + each_character['height'] > max_bottom:
            max_bottom = each_character['top'] + each_character['height']

    if min_top == float('inf') or max_bottom == float('-inf'):
        print(':WARNING: no valid character')
        return [], ""

    result_images = []
    result_labels = []
    last_right = characters[0]['left']
    for each_character in characters:
        if draw_bad is False and each_character['good'] is False:
            continue

        pad_top = each_character['top'] - min_top
        pad_bottom = max_bottom - (
            each_character['top'] + each_character['height'])
        pad_left = each_character['left'] - last_right
        if pad_left < 0:
            pad_left = 0

        image = np.pad(
            each_character['image'], ((pad_top, pad_bottom), (pad_left, 0)),
            mode='constant', constant_values=255)

        result_images.append(image)
        result_labels.append(each_character['label'])
        last_right = each_character['left'] + each_character['width']

    return np.concatenate(result_images, axis=1), ''.join(result_labels)


def construct_document(lines, draw_bad=False):
    """Draw the document

    # Arguments
        lines [list of list of characters]: list of line information
        draw_bad [bool]: whether to draw bad image

    # Returns
        [np array]: the document image
        [str]: the document label
    """
    raise NotImplementedError('for update')
