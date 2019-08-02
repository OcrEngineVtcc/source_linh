import json
import os
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FIELD_INDEX_REGEX = re.compile('\d+-?\d*?$')



def get_horizontal(obj):
    min_x = int(obj.find('bndbox').find('xmin').text)
    max_x = int(obj.find('bndbox').find('xmax').text)
    return min_x, max_x

def get_vertical(obj): 
    min_y = int(obj.find('bndbox').find('ymin').text)
    max_y = int(obj.find('bndbox').find('ymax').text)
    return min_y, max_y

def get_name(obj):
    return obj.find('name').text

def get_difficulty(obj):
    return int(obj.find('difficult').text)

def read_voc_file(filepath):
    """Read a PASCAL Voc file

    # Arguments
        filepath [str]: the path to pascal voc file

    # Returns
        [list of objs]: the list of bounding box objects
    """
    result = []

    root = ET.parse(filepath).getroot()
    objs = root.findall('object')

    for each_obj in objs:
        name = get_name(each_obj)
        min_x, max_x = get_horizontal(each_obj)
        min_y, max_y = get_vertical(each_obj)
        multi_line = get_difficulty(each_obj) == 1

        result.append({
            'name': name,
            'multi_line': multi_line,
            'position': {
                'top': min_y,
                'bottom': max_y,
                'left': min_x,
                'right': max_x
            }
        })

    return result

DRAW_COLOR = {
    'red': (200, 0, 0),
    'green': (0, 200, 0),
    'blue': (0, 0, 200)
}
NAME_FIELD = 0
ADDRESS_FIELD = 1
EMAIL_FIELD = 2
FUCK_UP = ["0548_004_box: address", 
    "0549_017_box: address", "0550_036_box: address"]

def filter_boxes(each_box):
    """Filter whether the box is a name/address/email
        {'name': name,
        'position': {
            'top': min_y,
            'bottom': max_y,
            'left': min_x,
            'right': max_x
        }}
    """
    position = each_box['position']
    if position['top'] < 850:
        return NAME_FIELD

    if '@' in each_box['name']:
        return EMAIL_FIELD

    return ADDRESS_FIELD


def draw_image(image_path, voc_path, save_path=''):
    """Draw the bounding box from `voc_path` on top of `image_path`

    # Arguments
        image_path [str]: the path to image file
        voc_path [str]: the path to voc file
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    bboxes = read_voc_file(voc_path)
    image = np.asarray(Image.open(image_path))
    image.setflags(write=1)

    from qutils import show_image
    # image = (255 - image*255).astype(np.uint8)
    image = (image*255).astype(np.uint8)
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)

    for each_box in bboxes:
        field_type = filter_boxes(each_box)
        if field_type == NAME_FIELD:
            color = 'red'
        elif field_type == ADDRESS_FIELD:
            color = 'green'
        elif field_type == EMAIL_FIELD:
            color = 'blue'
        else:
            raise ValueError()

        position = each_box['position']
        top, bottom = position['top'], position['bottom']
        left, right = position['left'], position['right']

        image[top:bottom+1, left:left+2] = DRAW_COLOR[color]     # left rect
        image[top:bottom+1, right:right+2] = DRAW_COLOR[color]   # right rect
        image[top:top+2, left:right+1] = DRAW_COLOR[color]       # top rect
        image[bottom:bottom+2, left:right+1] = DRAW_COLOR[color] # bottom rect

    image = Image.fromarray(image, mode='RGB')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/Library/Fonts/华文仿宋.ttf', 15)
    for each_box in bboxes:
        position = each_box['position']
        top, bottom = position['top'], position['bottom']
        left, right = position['left'], position['right']

        draw.text((left+5, top+5), each_box['name'], fill=(200,0,0), font=font)

    if save_path:
        filename = os.path.splitext(os.path.basename(image_path))[0]
        image.save(os.path.join(save_path, '{}_box.png'.format(filename)))
    else:
        show_image(np.asarray(image).astype(np.uint8))



# def extract_bounding_images(image_path, voc_path, save_path):
#     """Extract bounding boxes and labels

#     # Arguments:
#         image_path [str]: path to image file
#         voc_path [str]: path to voc file
#         save_path [str]: path to save the image
#     """
#     if save_path:
#         os.makedirs(save_path, exist_ok=True)

#     def get_min_top(top1, bottom1, top2, bottom2):
#         if top1 <= top2:
#             return top1, bottom1
#         else:
#             return top2, bottom2

#     def get_max_bottom(top1, bottom1, top2, bottom2):
#         if bottom1 >= bottom2:
#             return top1, bottom1
#         else:
#             return top2, bottom2

#     def check_overlap(top1, bottom1, top2, bottom2):
#         above_top, above_bottom = get_min_top(top1, bottom1, top2, bottom2)
#         below_top, below_bottom = get_max_bottom(top1, bottom1, top2, bottom2)

#         if above_bottom < below_top:
#             return False
        
#         # if (above_top == below_top) and (above_bottom == below_bottom):
#         #     return True

#         if (above_bottom - below_top) / (below_bottom - above_top) >= 0.5:
#             return True

#         return False

#     def find_related_boxes(box, other_boxes):
#         result = []
#         position = box['position']
#         top, bottom = position['top'], position['bottom']
#         for each_box in other_boxes:
#             each_position = each_box['position']
#             each_top, each_bottom = each_position['top'], each_position['bottom']
#             if check_overlap(top, bottom, each_top, each_bottom):
#                 result.append(each_box)
#         result.append(box)

#         return result

#     def construct_combine_image(list_of_objects):
#         list_of_objects.sort(key=lambda obj: obj['position']['left'])
#         min_top = float('inf')
#         max_bottom = 0
#         for obj in list_of_objects:
#             if obj['position']['top'] < min_top:
#                 min_top = obj['position']['top']
#             if obj['position']['bottom'] > max_bottom:
#                 max_bottom = obj['position']['bottom']

#         label = ' '.join([obj['name'] for obj in list_of_objects])

#         return (
#             min_top, max_bottom, list_of_objects[0]['position']['left'],
#             list_of_objects[-1]['position']['right'], label)


#     bboxes = read_voc_file(voc_path)
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)

#     extracted_regions = []
#     for each_box in bboxes:
#         position = each_box['position']
#         top, bottom = position['top'], position['bottom']
#         left, right = position['left'], position['right']
#         extracted_regions.append({
#             'name': each_box['name'],
#             'image': image[top:bottom+1, left:right+1, :],
#             'position': position
#         })

#     skip = 0
#     extra_regions = []
#     for _idx, each_region in enumerate(extracted_regions):
#         if skip > 0:
#             skip -= 1
#             continue
#         related_boxes = find_related_boxes(
#             each_region,
#             extracted_regions[_idx+1:])

#         if len(related_boxes) > 1:
#             skip = len(related_boxes) - 2
#             top, bottom, left, right, name = construct_combine_image(related_boxes)
#             extra_regions.append({
#                 'name': name,
#                 'image': image[top:bottom+1, left:right+1, :]
#             })

#     for each in extra_regions:
#         extracted_regions.append(each)

#     filename = os.path.splitext(os.path.basename(image_path))[0]
#     for each_image in extracted_regions:
#         image = Image.fromarray(each_image['image'])
#         image.save(os.path.join(
#             save_path,
#             '{}★{}.png'.format(filename, each_image['name'])))

def extract_bounding_images_with_types_test(image_path, voc_path, save_path):
    """Extract bounding boxes and labels

    # Arguments:
        image_path [str]: path to image file
        voc_path [str]: path to voc file
        save_path [str]: path to save the image
    """
    cells_path = os.path.join(save_path, 'cellcuts')
    addresses_path = os.path.join(save_path, 'addresscuts')
    emails_path = os.path.join(save_path, 'emailcuts')
    names_path = os.path.join(save_path, 'namecuts')

    os.makedirs(cells_path, exist_ok=True)
    os.makedirs(addresses_path, exist_ok=True)
    os.makedirs(emails_path, exist_ok=True)
    os.makedirs(names_path, exist_ok=True)


    def get_min_top(top1, bottom1, top2, bottom2):
        if top1 <= top2:
            return top1, bottom1
        else:
            return top2, bottom2

    def get_max_bottom(top1, bottom1, top2, bottom2):
        if bottom1 >= bottom2:
            return top1, bottom1
        else:
            return top2, bottom2

    def check_overlap(top1, bottom1, top2, bottom2):
        above_top, above_bottom = get_min_top(top1, bottom1, top2, bottom2)
        below_top, below_bottom = get_max_bottom(top1, bottom1, top2, bottom2)

        if above_bottom < below_top:
            return False
        
        # if (above_top == below_top) and (above_bottom == below_bottom):
        #     return True

        if (above_bottom - below_top) / (below_bottom - above_top) >= 0.5:
            return True

        return False

    def find_related_boxes(box, other_boxes):
        result = []
        position = box['position']
        top, bottom = position['top'], position['bottom']
        for each_box in other_boxes:
            each_position = each_box['position']
            each_top, each_bottom = each_position['top'], each_position['bottom']
            if check_overlap(top, bottom, each_top, each_bottom):
                result.append(each_box)
        result.append(box)

        return result

    def construct_combine_image(list_of_objects):
        list_of_objects.sort(key=lambda obj: obj['position']['left'])
        min_top = float('inf')
        max_bottom = 0
        for obj in list_of_objects:
            if obj['position']['top'] < min_top:
                min_top = obj['position']['top']
            if obj['position']['bottom'] > max_bottom:
                max_bottom = obj['position']['bottom']

        label = ' '.join([obj['name'] for obj in list_of_objects])

        return (
            min_top, max_bottom, list_of_objects[0]['position']['left'],
            list_of_objects[-1]['position']['right'], label)


    bboxes = read_voc_file(voc_path)
    image = np.asarray(Image.open(image_path))
    image.setflags(write=1)

    image = (image*255).astype(np.uint8)
    # if len(image.shape) == 2:
    #     image = np.stack([image, image, image], axis=-1)

    extracted_regions = []
    for each_box in bboxes:
        position = each_box['position']
        top, bottom = position['top'], position['bottom']
        left, right = position['left'], position['right']
        extracted_regions.append({
            'name': each_box['name'],
            'image': image[top:bottom+1, left:right+1],
            'position': position,
            'type': filter_boxes(each_box)
        })

    skip = 0
    extra_regions = []
    for _idx, each_region in enumerate(extracted_regions):
        if skip > 0:
            skip -= 1
            continue
        related_boxes = find_related_boxes(
            each_region,
            extracted_regions[_idx+1:])

        if len(related_boxes) > 1:
            skip = len(related_boxes) - 2
            top, bottom, left, right, name = construct_combine_image(related_boxes)
            extra_regions.append({
                'name': name,
                'image': image[top:bottom+1, left:right+1],
                'position': {
                    'top': top, 'bottom': bottom, 'left': left, 'right': right
                },
                'type': filter_boxes({
                    'name': name,
                    'position': {
                        'top': top, 'bottom': bottom,
                        'left': left, 'right': right
                    }
                })
            })

    for each in extra_regions:
        extracted_regions.append(each)


    filename = os.path.splitext(os.path.basename(image_path))[0]
    for each_image in extracted_regions:
        image = Image.fromarray(each_image['image'])
        image.save(os.path.join(
            cells_path,
            '{}<<SEP>>{}.png'.format(filename, each_image['name'])))
        data_type = each_image['type']
        if data_type == NAME_FIELD:
            image.save(os.path.join(
                names_path,
                '{}<<SEP>>{}.png'.format(filename, each_image['name'])))
        elif data_type == ADDRESS_FIELD:
            image.save(os.path.join(
                addresses_path,
                '{}<<SEP>>{}.png'.format(filename, each_image['name'])))
        elif data_type == EMAIL_FIELD:
            image.save(os.path.join(
                emails_path,
                '{}<<SEP>>{}.png'.format(filename, each_image['name'])))
        else:
            print('Holy SHIT')

def extract_bounding_images_with_types(image_path, voc_path, save_path):
    """Extract bounding boxes and labels

    # Arguments:
        image_path [str]: path to image file
        voc_path [str]: path to voc file
        save_path [str]: path to save the image
    """
    cells_path = os.path.join(save_path, 'cellcuts')
    addresses_path = os.path.join(save_path, 'addresscuts')
    emails_path = os.path.join(save_path, 'emailcuts')
    names_path = os.path.join(save_path, 'namecuts')

    os.makedirs(cells_path, exist_ok=True)
    os.makedirs(addresses_path, exist_ok=True)
    os.makedirs(emails_path, exist_ok=True)
    os.makedirs(names_path, exist_ok=True)


    def get_min_top(top1, bottom1, top2, bottom2):
        if top1 <= top2:
            return top1, bottom1
        else:
            return top2, bottom2

    def get_max_bottom(top1, bottom1, top2, bottom2):
        if bottom1 >= bottom2:
            return top1, bottom1
        else:
            return top2, bottom2

    def check_overlap(top1, bottom1, top2, bottom2):
        above_top, above_bottom = get_min_top(top1, bottom1, top2, bottom2)
        below_top, below_bottom = get_max_bottom(top1, bottom1, top2, bottom2)

        if above_bottom < below_top:
            return False
        
        # if (above_top == below_top) and (above_bottom == below_bottom):
        #     return True

        if (above_bottom - below_top) / (below_bottom - above_top) >= 0.5:
            return True

        return False

    def find_related_boxes(box, other_boxes):
        result = []
        position = box['position']
        top, bottom = position['top'], position['bottom']
        for each_box in other_boxes:
            each_position = each_box['position']
            each_top, each_bottom = each_position['top'], each_position['bottom']
            if check_overlap(top, bottom, each_top, each_bottom):
                result.append(each_box)
        result.append(box)

        return result

    def construct_combine_image(list_of_objects):
        list_of_objects.sort(key=lambda obj: obj['position']['left'])
        min_top = float('inf')
        max_bottom = 0
        for obj in list_of_objects:
            if obj['position']['top'] < min_top:
                min_top = obj['position']['top']
            if obj['position']['bottom'] > max_bottom:
                max_bottom = obj['position']['bottom']

        label = ' '.join([obj['name'] for obj in list_of_objects])

        return (
            min_top, max_bottom, list_of_objects[0]['position']['left'],
            list_of_objects[-1]['position']['right'], label)


    bboxes = read_voc_file(voc_path)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image.dtype == bool:
        image = image.astype(np.uint8)

    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)

    extracted_regions = []
    for each_box in bboxes:
        if each_box['multi_line']:
            print('Multi-line:', each_box['name'])
            continue

        position = each_box['position']
        top, bottom = position['top'], position['bottom']
        left, right = position['left'], position['right']
        extracted_regions.append({
            'name': each_box['name'],
            'image': image[top:bottom+1, left:right+1],
            'position': position,
            'type': filter_boxes(each_box)
        })

    skip = 0
    extra_regions = []
    for _idx, each_region in enumerate(extracted_regions):
        if skip > 0:
            skip -= 1
            continue
        related_boxes = find_related_boxes(
            each_region,
            extracted_regions[_idx+1:])

        if len(related_boxes) > 1:
            skip = len(related_boxes) - 2
            top, bottom, left, right, name = construct_combine_image(related_boxes)
            extra_regions.append({
                'name': name,
                'image': image[top:bottom+1, left:right+1],
                'position': {
                    'top': top, 'bottom': bottom, 'left': left, 'right': right
                },
                'type': filter_boxes({
                    'name': name,
                    'position': {
                        'top': top, 'bottom': bottom,
                        'left': left, 'right': right
                    }
                })
            })

    for each in extra_regions:
        extracted_regions.append(each)


    filename = os.path.splitext(os.path.basename(image_path))[0]
    for each_image in extracted_regions:
        # print(type(each_image['image']))
        # print(each_image['image'].shape)
        # print(np.unique(each_image['image']))
        # print(each_image['image'].dtype == bool)
        image = Image.fromarray(each_image['image'])
        image.save(os.path.join(
            cells_path,
            '{}★{}.png'.format(filename, each_image['name'])))
        data_type = each_image['type']
        if data_type == NAME_FIELD:
            image.save(os.path.join(
                names_path,
                '{}★{}.png'.format(filename, each_image['name'])))
        elif data_type == ADDRESS_FIELD:
            image.save(os.path.join(
                addresses_path,
                '{}★{}.png'.format(filename, each_image['name'])))
        elif data_type == EMAIL_FIELD:
            image.save(os.path.join(
                emails_path,
                '{}★{}.png'.format(filename, each_image['name'])))
        else:
            print('Holy SHIT')


def get_image_filetype(image_path):
    """Get image filetype

    # Arguments
        image_path [str]: path to image

    # Returns
        [str]: the filetype of image (e.g jpg, tif...)
    """
    _, ext = os.path.splitext(image_path)
    return ext[1:].lower()


def analyze_name(name, sep='★'):
    return name.split(sep)


def field_type_from_box_type(box_type):
    """Get the field type

    # Example:
        - address1-2 => address
        - address1 => address
    """
    start = FIELD_INDEX_REGEX.search(box_type).start()
    return box_type[:start]


def in_box(big_box, small_box):
    """Check if big_box in small_box"""
    sb_vertical_center = int((small_box['top'] + small_box['bottom']) / 2)
    sb_horizontal_center = int((small_box['left'] + small_box['right']) / 2)

    # check inside vertically
    inside_vertical = False
    if (sb_vertical_center > big_box['top'] and 
        sb_vertical_center < big_box['bottom']):
        inside_vertical = True

    # check inside horizontally
    inside_horizontal = False
    if (sb_horizontal_center > big_box['left'] and
        sb_horizontal_center < big_box['right']):
        inside_horizontal = True

    return inside_vertical and inside_horizontal


def collect_cells(big_box, cells):
    """Collect cells from a given paragraph

    # Arguments
        big_box [str]: the name of paragraph
        cells [list of objs]: the list of cells

    # Returns
        [obj]: a single object containing all cells
    """
    box_type = big_box['box_type']

    matched_cells = []
    for each_cell in cells:
        if not each_cell['box_type'].startswith('{}-'.format(box_type)):
            continue

        matched_cells.append(each_cell)

    if len(matched_cells) > 1:
        # then box_type is something like `address1`
        top = min(matched_cells, key=lambda obj: obj['top'])['top']
        bottom = max(matched_cells, key=lambda obj: obj['bottom'])['bottom']
        left = min(matched_cells, key=lambda obj: obj['left'])['left']
        right = max(matched_cells, key=lambda obj: obj['right'])['right']

        # the label is constructed from top to bottom
        matched_cells.sort(key=lambda obj: obj['top'])

    elif len(matched_cells) == 1:
        return None

    else:
        # then box_type is something like `paragraph1`
        matched_cells = []        
        for small_box in cells:
            if in_box(big_box, small_box):
                matched_cells.append(small_box)

        # the label is constructed from top to bottom ## TODO: error prone
        matched_cells.sort(key=lambda obj: obj['top'])

    box_label = []
    for each_cell in matched_cells:
        box_label.append(each_cell['box_label'])

    box_label = ' '.join(box_label)
    folder = field_type_from_box_type(box_type)
    obj = {
        'folder': folder,
        'top': big_box['top'], 'bottom': big_box['bottom'], 
        'left': big_box['left'], 'right': big_box['right'],
        'box_label': box_label
    }

    return obj


def extract_bounding_images_with_stars(image_path, voc_path, save_path):
    """Extract the bounding images"""

    os.makedirs(save_path, exist_ok=True)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    bboxes = read_voc_file(voc_path)

    cell_boxes = []
    _temp_paragraph_boxes = []
    for each_box in bboxes:
        if each_box['name'].startswith('paragragh'):
            box_type = each_box['name']
            box_label = ''
        else:
            box_type, box_label = analyze_name(each_box['name'])

        position = each_box['position']
        top, bottom = position['top'], position['bottom']
        left, right = position['left'], position['right']

        obj = {
            'box_type': box_type,
            'box_label': box_label,
            'top': top, 'bottom': bottom, 'left': left, 'right': right
        }
        if box_label == '':
            _temp_paragraph_boxes.append(obj)
        else:
            if not each_box['multi_line']:
                obj['folder'] = field_type_from_box_type(box_type)
                cell_boxes.append(obj)

    paragraph_boxes = []
    for each_box in _temp_paragraph_boxes:
        collected_cell = collect_cells(each_box, cell_boxes)
        if collected_cell is None:
            continue
        else:
            paragraph_boxes.append(collected_cell)


    filename, _ = os.path.splitext(os.path.basename(image_path))
    for each_cell in cell_boxes:
        folder_path = os.path.join(save_path, each_cell['folder'])
        os.makedirs(folder_path,exist_ok=True)
        cell_image = image[each_cell['top']:each_cell['bottom']+1,
                           each_cell['left']:each_cell['right']+1]
        cv2.imwrite(
            os.path.join(folder_path, '{}★{}.png'.format(filename, each_cell['box_label'])),
            cell_image)

    # for each_cell in paragraph_boxes:
    #     folder_path = os.path.join(save_path, each_cell['folder'])
    #     os.makedirs(folder_path,exist_ok=True)
    #     cell_image = image[each_cell['top']:each_cell['bottom']+1,
    #                        each_cell['left']:each_cell['right']+1]
    #     cv2.imwrite(
    #         os.path.join(folder_path, '{}.png'.format(each_cell['box_label'])),
    #         cell_image)


def extract_bounding_images(image_path, voc_path, save_path):
    """Extract bounding boxes and labels

    # Arguments:
        image_path [str]: path to image file
        voc_path [str]: path to voc file
        save_path [str]: path to save the image
    """
    bboxes = read_voc_file(voc_path)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    labels = {}

    extracted_regions = []
    for each_box in bboxes:
        position = each_box['position']
        top, bottom = position['top'], position['bottom']
        left, right = position['left'], position['right']

        extracted_regions.append({
            'name': each_box['name'],
            'image': image[top:bottom+1, left:right+1, :],
            'position': position
        })

    filename = os.path.splitext(os.path.basename(image_path))[0]
    for _idx, each_image in enumerate(extracted_regions):
        box_filename = '{}_{}.png'.format(filename, _idx)
        cv2.imwrite(os.path.join(save_path, box_filename), each_image['image'])
        labels[box_filename] = each_image['name'].split('★')[-1]

    return labels


if __name__ == '__main__':
    import glob
    folder_path = '/Users/ducprogram/Downloads/20180523/png'
    jpeg_files = (glob.glob(os.path.join(folder_path, '*.png'))
        + glob.glob(os.path.join(folder_path, '*.jpg')))

    labels = {}

    # extract_bounding_images_with_stars(
    #     '/Users/ducprogram/cinnamon/data_gen/cinnamon_datasets/JinData/2_image_all/1071S001439_A1.tif',
    #     '/Users/ducprogram/cinnamon/data_gen/cinnamon_datasets/JinData/2_image_all/1071S001439_A1.xml',
    #     '/Users/ducprogram/cinnamon/data_gen/cinnamon_datasets/JinData/results')

    for _idx, each_file in enumerate(jpeg_files):
        print(each_file)
        filename = os.path.splitext(os.path.basename(each_file))[0]

        try:
            # draw_image(
            #     each_file,
            #     os.path.join(folder_path, '{}.xml'.format(filename)),
            #     '/Users/ducprogram/Downloads/20180523/png/result/temp')

            # extract_bounding_images_with_stars(
            #     each_file,
            #     os.path.join(folder_path, '{}.xml'.format(filename)),
            #     '/Users/ducprogram/cinnamon/data_gen/cinnamon_datasets/JinData/results/SG_Foldings/2_image_all')
            label = extract_bounding_images(
                each_file,
                os.path.join(folder_path, '{}.xml'.format(filename)),
                '/Users/ducprogram/Downloads/20180523/png/result/')
            labels.update(label)
        except:
            print("Missing xml file for {}".format(filename))
            continue

    with open('/Users/ducprogram/Downloads/20180523/png/result/label.json',
        'w') as out_file:
        json.dump(labels, out_file)


    # draw_image(
    #     '/Users/ducprogram/Downloads/107BS000001_A1.tif',
    #     '/Users/ducprogram/Downloads/107BS000001_A1.xml',
    #     '/Users/ducprogram/Desktop')

    # extract_bounding_images_with_types(
    #     '/Users/ducprogram/cinnamon/data_gen/cinnamon_datasets/JinData/train_1098_2singlelineadd/0550_035.jpg',
    #     '/Users/ducprogram/cinnamon/data_gen/cinnamon_datasets/JinData/train_1098_2singlelineadd/0550_035.xml',
    #     '/Users/ducprogram/Desktop/test')