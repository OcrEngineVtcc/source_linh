import cv2
import glob
import os
import json
import math
import multiprocessing
import pdb
import pickle
import re
import time
import unicodedata
import numpy as np
import numpy.random as random

from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import rotate

from fontTools.ttLib import TTFont
from fontTools.ttLib.sfnt import readTTCHeader

import sys
print(sys.path)
sys.path.append('../')

from utils.synthesizeData.generator.base_generator import BaseOCRGenerator
from utils.synthesizeData.helper.helper_controller import FFG_Helper

import utils.synthesizeData.utils.constants as constants
from utils.synthesizeData.utils.misc import (get_truncated_normal,
                                             check_allowed_char_version, dump_json)
import utils.synthesizeData.configRandomParameter.rdcf_ffg_field8 as configAugment

# from utils.synthesizeData.utils.normalize import load_conversion_table, normalize_text

from utils.synthesizeData.utils.preprocessing import (
    crop_image, adjust_stroke_width,
    normalize_grayscale_color, unsharp_masking, skeletonize_image,
    trim_image_horizontally)
from utils.synthesizeData.utils.augment import (HandwritingMildAugment, PrintMildAugment)


IS_DEBUG_LOG = False


class FFGLineGenerator(BaseOCRGenerator):
    """
    This generator assumes there are:
        (1) a collection of text files, from which the content of images will
            be generated;
        (2) a collection of pickle files, each of which contains
            [list of np array images] and [list of labels]. Each image should
            have white background, black foreground, and not having outer
            background

    Usage example:
    ```
    import os
    from dataloader.generate.image import HandwrittenLineGenerator

    lineOCR = HandwrittenLineGenerator()
    lineOCR.load_character_database('images.pkl')
    lineOCR.load_text_database('text.txt')

    # to get random characters
    X, widths, y = lineOCR.get_batch(4, 1800)

    # or to generate 100 images
    lineOCR.generate_images(start=0, end=100, save_dir='/tmp/Samples')
    ```
    """


    def __init__(self, height=70, helper=None, limit_per_char=3000, verbose=2,
                 allowed_chars=None, is_binary=False, augmentor=None,
                 deterministic=True):
        """Initialize the generator.

        # Arguments
            limit_per_char [int]: the maximum number of images that each
                character will contain (to save on RAM)
            augmentor [Augment object]: the augmentator to use
        """


        super(FFGLineGenerator, self).__init__(
            height=height, helper=helper, allowed_chars=allowed_chars,
            is_binary=is_binary, verbose=verbose)

        self.fonts = []

        # image configuration
        self.limit_per_char = limit_per_char
        self.char_2_imgs = {}
        self.char_choosen = {}  # array of index was choosen with this char

        self.background_value = 1 if is_binary else 255

        self.interpolation = (
            cv2.INTER_NEAREST if self.is_binary else
            cv2.INTER_LINEAR)

        # utility
        self.augment = (
            augmentor(is_binary=is_binary) if augmentor is not None
            else HandwritingMildAugment(is_binary=is_binary))
        self.helper = FFG_Helper() if helper is None else helper
        self.deterministic = deterministic

    def _remove_unknown_characters(self, text):
        """Check for characters in the text that are not in database.

        # Arguments
            text [str]: the string to check

        # Returns
            [str]: the text that have missing characters removed
            [set of str]: missing characters
        """
        exist = []
        missing_chars = set([])
        for each_char in text:
            if each_char not in self.char_2_imgs:
                missing_chars.add(each_char)
            else:
                exist.append(each_char)

        return ''.join(exist), missing_chars

    def _get_config(self, text, default_config={}):
        """Returns a configuration object for the text

        The idea is that each line of text will have a specific configuration,
        which then will be used during image generation. The configuration file
        has the following format: {
            config_key: config_value
            'text': [list of {}s with length == len(text), with each {} is a
                     config for that specific word]
        }

        The returning object contains these following keys:
            'text': a list of each character configuration

        Each character configuration object contains these following keys:
            'skewness': the skew angle of character
            'character_normalization_mode': an integer specifcy how to
                normalize resized characters
            'space': the space distance to the next character
            'bottom': the bottom padding
            'height': height of character (in pixel)
            'width_noise': the amount to multiply with character width

        # Arguments
            text [str]: the text to generate string
            default_config [dict]: the default config value

        # Returns
            [obj]: the configuration object
        """
        # config = {'text': []}
        # text_config = {'height': 0, 'width': 0, 'skewness': 0, 'space': 0}

        # Configuration for each char
        random.seed()
        is_space_close = False  # random.random() > 0.8
        is_skewness = random.random() > 0.95
        is_curve_line = ((len(text) > 30) and (random.random() > 0.95)
                         and default_config.get('is_curve_line', False))
        is_last_numbers_up = (len(text) > 10 and self._is_number(text[-1]) and
                              self._is_number(text[-2]) and self._is_number(text[-3]) and
                              random.random() > 0.9)
        if is_skewness:
            skew_value = random.randint(-5, 5)

        if is_curve_line:
            curve_start = random.randint(5, len(text) - 10)
            curve_middle = random.randint(curve_start + 3, len(text) - 1)
            curve_end = random.randint(curve_middle, len(text))

            curve_max_angle = random.randint(10, 45)
            curve_first_half = set(range(curve_start, curve_middle))
            curve_second_half = set(range(curve_middle, curve_end + 1))
            curve_delta_first_half = curve_max_angle / len(curve_first_half)
            curve_delta_second_half = curve_max_angle / len(curve_second_half)

            # convex vs concave curve
            curve_type = -1 if random.random() <= 0.5 else 1

        # how to normalize character width
        is_character_normalize = 1.0 if self.is_binary else random.random()
        if is_character_normalize <= 0.7:
            character_normalization_mode = 3
        elif is_character_normalize <= 0.8:
            character_normalization_mode = 2
        elif is_character_normalize <= 0.9:
            character_normalization_mode = 1
        else:
            character_normalization_mode = 0
        # End char configuration

        # each text configuration
        text_config = []
        last_bottom = 0
        for _idx, each_char in enumerate(text):

            # space
            space = configAugment.getRandomSpaceCharHW()

            # skew value
            if not self._is_number(each_char) and is_skewness:
                skew = 2 * random.random() - 1 + skew_value
            else:
                skew = 0

            # character height and width
            if each_char in constants.SMALL_CHARS:
                height_ratio = random.uniform(low=0.15, high=0.3)
            elif each_char in constants.MEDIUM_CHARS:
                height_ratio = random.uniform(low=0.3, high=0.5)
            elif each_char in constants.SMALL_KATA_HIRA:
                height_ratio = random.uniform(low=0.5, high=0.7)
            elif each_char in constants.SMALL_LATIN:
                height_ratio = random.uniform(low=0.6, high=0.75)
            elif each_char in constants.NORMAL_FORCE_SMALLER:
                height_ratio = random.uniform(low=0.7, high=0.75)
            elif each_char in constants.KATA_HIRA:
                height_ratio = random.uniform(low=0.75, high=0.95)
            elif each_char in constants.NUMBERS:
                height_ratio = random.uniform(low=0.9, high=1.0)
            else:
                height_ratio = get_truncated_normal(1.00, 0.02, 0.97, 1.1)

            height = int(self.character_height * height_ratio)
            width_noise = get_truncated_normal(1, 0.02, 1.00, 1.05)

            # character bottom padding value
            base_bottom = last_bottom
            if random.random() > 0.9:
                base_bottom = last_bottom + random.randint(-3, 3)
                last_bottom = base_bottom

            if each_char in constants.MIDDLE_CHARS:
                bottom = base_bottom + random.randint(10, 25)
            elif each_char in constants.TOP_CHARS:
                bottom = base_bottom + random.randint(30, 45)
            else:
                bottom = base_bottom

            text_config.append({'skewness': skew, 'space': space,
                                'character_normalization_mode': character_normalization_mode,
                                'bottom': bottom, 'height': height,
                                'width_noise': width_noise})

        # perform curve line configuration
        if is_curve_line:
            total_delta_bottom = 0
            total_delta_angle = 0
            for _idx, each_config in enumerate(text_config):
                if _idx in curve_first_half:
                    total_delta_bottom += curve_type * random.randint(2, 5)
                    each_config['bottom'] += total_delta_bottom
                    total_delta_angle += curve_type * curve_delta_first_half
                    each_config['skewness'] += total_delta_angle
                elif _idx in curve_second_half:
                    total_delta_bottom += curve_type * random.randint(2, 5)
                    each_config['bottom'] += total_delta_bottom
                    total_delta_angle -= curve_type * curve_delta_second_half
                    each_config['skewness'] += total_delta_angle
                elif _idx == curve_middle:
                    total_delta_bottom += curve_type * random.randint(2, 5)
                    each_config['bottom'] += total_delta_bottom
                    each_config['skewness'] += curve_type * curve_max_angle
                elif _idx >= curve_end:
                    each_config['bottom'] += total_delta_bottom

        # normalize the bottom value (such that the lowest value should be 3)
        if len(text_config) > 0:
            min_bottom = min(text_config, key=lambda obj: obj['bottom'])['bottom']
        else:
            min_bottom = 2

        for each_config in text_config:
            each_config['bottom'] = each_config['bottom'] - min_bottom + 3

        return {
            'text': text_config,
        }

    def _generate_single_image(self, char, config, char_idx=None):
        """Generate a character image

        # Arguments
            char [str]: a character to generate image
            config [obj]: a configuration object for this image
            char_idx [int]: to determine the character beforehand

        # Returns
            [np array]: the generated image for that specific string
        """

        if char not in self.char_2_imgs.keys():
            return None

        choice = (random.choice(len(self.char_2_imgs[char]))
                  if char_idx is None
                  else char_idx)

        image = self.char_2_imgs[char][choice]

        # rotate image
        image = rotate(image, config['skewness'], order=1,
                       cval=self.background_value)

        # resize image
        if char not in constants.NOT_RESIZE:
            height, width = image.shape
            desired_width = int(
                width * (config['height'] / height) * config['width_noise'])
            image = self._resize_character(
                image, config['height'],
                desired_width, config['character_normalization_mode'])

        # add horizontal space and bottom space
        image = np.pad(image, ((0, config['bottom']), (0, config['space'])),
                       'constant', constant_values=self.background_value)

        return image

    def _generate_sequence_image(self, text, debug=False, font=None):
        """Generate string image of a given text

        # Arguments
            text [str]: the text that will be used to generate image

        # Returns
            [np array]: the image generated
            [str]: the text label
            [list of str]: list of missing characters
        """

        # gen handwriten
        char_images = []
        default_config = {}

        # remove characters in text that not exist in etl character images
        text, missChar = self._remove_unknown_characters(text)
        if len(text)== 0:
            text = str(random.randint(0,100))

        self.character_height = configAugment.getConfigHeightHW()  # to help resize to self.height
        self.height = self.character_height
        config = self._get_config(text, default_config)
        

        # Calculate the average height of a character
        if self.deterministic:
            indices = {each_char: self._get_random_choice_index(each_char)
                           for each_char in list(set(text))}
        else:
            indices = {}

        for _idx, each_char in enumerate(text):
            char_images.append(self._generate_single_image(
                    each_char, config['text'][_idx], indices.get(each_char, None))
                )

            # Normalize character image height to have the same height by padding
            # the top into desired_height
        if len(char_images)>0:
            max_height = max(
                char_images, key=lambda obj: obj.shape[0]).shape[0]
        else:
            max_height = 64

        desired_height = max_height + 4
        norm_img_seq = []
        for each_img in char_images:
            top_pad = desired_height - each_img.shape[0] - 3
            norm_img_seq.append(np.pad(each_img, ((top_pad, 3), (0, 0)),
                                           mode='constant', constant_values=self.background_value))

        image = np.concatenate(norm_img_seq, axis=1)

        # add padding space behind the final characters
        image = np.pad(image, ((0, 0), (0, 10)),
                           mode='constant', constant_values=self.background_value)

        image = self.augment.augment_line(image)

        return image, text

    def _get_random_choice_index(self, each_char):
        while True:
            index = random.choice(len(self.char_2_imgs[each_char]))
            if self.char_choosen.get(each_char) is None:
                self.char_choosen[each_char] = []
                self.char_choosen[each_char].append(index)
                break
            else:
                if index in self.char_choosen[each_char]:
                    if len(self.char_choosen[each_char]) < len(self.char_2_imgs[each_char]):
                        continue
                    else:
                        # no more option to choose -> choose this current value of index
                        # reset current dictionary
                        self.char_choosen = {}
                        break
                else:
                    self.char_choosen[each_char].append(index)
                    break
        return index

    def _resize_character(self, image, desired_height, desired_width,
                          character_normalization_mode=4):
        """Resize and normalize the character

        This method optionally normalizes the characters, so that the affect
        of resizing characters do not have a bias affects on the model.
        Sometimes we can skip normalization to provide more noise effects.

        # Arguments
            image [np array]: the character image to resize
            desired_height [int]: the desired height to resize the image into
            desired_width [int]: the desired width to resize the image into
            character_normalization_mode [int]: to have value of 0-3, variate
                the normalization scheme

        # Returns
            [np array]: the resized character image
        """
        original_height, original_width = image.shape
        ratio = desired_height / original_height

        # Resize the character
        image = cv2.resize(image, (desired_width, desired_height),
                           interpolation=self.interpolation)

        # Adjust the stroke width, color, and deblur the result with some
        # randomness to enhance model robustness
        if character_normalization_mode > 0:
            image = adjust_stroke_width(image, ratio, is_binary=self.is_binary)

        if character_normalization_mode > 1:
            image = normalize_grayscale_color(image)

        if character_normalization_mode > 2:
            image = unsharp_masking(image)

        return image

    def load_character_database(self, file_path, shuffle=True):
        """Load image database into the dataset

        # Arguments
            file_path [str]: the path to pickle file to load X, y
            shuffle [bool]: whether to shuffle the wholething
        """
        with open(file_path, 'rb') as f:
            X, y = pickle.load(f)

            if shuffle:
                idx_permutation = random.permutation(len(y))
                X = [X[each_idx] for each_idx in idx_permutation]
                y = [y[each_idx] for each_idx in idx_permutation]

            # Sanity check a random image
            unique_pixels = len(np.unique(X[random.choice(idx_permutation)]))
            if self.is_binary:
                if unique_pixels != 2:
                    print(':WARNING: binary image should have 2 pixel values '
                          'but have {} values'.format(unique_pixels))
            else:
                if unique_pixels == 2:
                    print(':WARNING: the loaded dataset might be binary data')

            # listKeyUnique = []
            for _idx, each_X in enumerate(X):
                key = y[_idx]

                if key in self.char_2_imgs:
                    if len(self.char_2_imgs[key]) > self.limit_per_char:
                        continue
                # else:
                    # listKeyUnique.append(key)

                if (self.allowed_chars is not None
                        and key not in self.allowed_chars):
                    continue

                # debug
                if IS_DEBUG_LOG:
                    filename = 'C:\\Users\\ABC\\Desktop\\deletetemp\\GenDataHWBB\\synthesizedKana\\test1.png'
                    cv2.imwrite(filename, each_X)

                if key in self.char_2_imgs:
                    self.char_2_imgs[key].append(each_X)
                else:
                    self.char_2_imgs[key] = [each_X]

        if self.verbose >= 2:
            print('{} loaded'.format(file_path))
            # print(len(listKeyUnique))
            # print(listKeyUnique)
            a=1

    def load_background_image_files(self, folder_path):
        """Load background image files

        # Arguments
            folder_path [str]: the path to folder that contains background
        """
        if self.is_binary:
            print(':WARNING: background image files are not loaded for binary '
                  'generation mode.')
        else:
            self.augment.add_background_image_noises(folder_path)

    def initialize(self):
        """Initialize the generator
        """
        super(FFGLineGenerator, self).initialize()

    def generate_images(self, start=0, end=None, save_dir=None,
                        label_json=False, mode=None, debug=True,
                        text_lines=None, **kwargs):
        """Generate the images

        Subclass from super, in order to determine batch mode. Other
        arguments can be seen in the subclass

        # Arguments
            mode [int]: the mode of this generator
        """

        return super(FFGLineGenerator, self).generate_images(
            start, end, save_dir, label_json, debug=debug, text_lines=text_lines
        )


class PrintedLineGenerator(BaseOCRGenerator):
    """
    This generator assumes there are:
        (1) a collection of text files, from which the content of images will
            be generated;
        (2) a collection of font files

    Usage example:
    ```
    import os
    from dataloader.generate.image import PrintedLineGenerator

    lineOCR = PrintedLineGenerator()
    lineOCR.load_fonts('./fonts/')
    lineOCR.load_text_database('text.txt')

    # to get random characters
    X, widths, y = lineOCR.get_batch(4, 1800)

    # or to generate 100 images
    lineOCR.generate_images(start=0, end=100, save_dir='/tmp/Samples')
    ```
    """
    FONT_EXTENSIONS = ['otf', 'ttf', 'OTF', 'TTF']

    def __init__(self, height=74, num_workers=8, allowed_chars=None,
                 helper=None, augmentor=None, is_binary=False, verbose=2):
        """Initialize the generator"""
        super(PrintedLineGenerator, self).__init__(height=height,
                                                   helper=helper, allowed_chars=allowed_chars, verbose=verbose)

        # image configuration
        self.fonts = []

        # utility
        self.augment = PrintMildAugment(is_binary=is_binary, cval=255) 

    def _remove_unknown_characters(self, text, font_check):
        """Check for characters in the text that are not in database.

        # Arguments
            text [str]: the string to check

        # Returns
            [str]: the text that have missing characters removed
            [set of str]: missing characters
        """
        exist = []
        for each_char in text:
            if self._is_char_supported_by_font(each_char, font_check):
                exist.append(each_char)
            # else:
                # print(each_char, '  ---- char is not support by font')
        return ''.join(exist)

    def _is_char_supported_by_font(self, char, font_check):
        """Check whether the current font supported to draw `char`

        # Arguments
            char [str]: the character to check
            font_check [TTFont]: the font check object

        # Returns
            [bool]: True if the font support `char`, False otherwise
        """
        for cmap in font_check['cmap'].tables:
            if cmap.isUnicode():
                if ord(char) in cmap.cmap:
                    return True

        return False

    def _get_config(self, text, font,  font_draw, font_check,  default_config={}):
        """Returns a configuration object for the text

        The idea is that each line of text will have a specific configuration,
        which then will be used during image generation. The configuration file
        has the following format: {
            config_key: config_value
            'text': [list of {}s with length == len(text), with each {} is a
                     config for that specific word]
        }

        The returning object contains these following keys:
            'text': a list of each character configuration

        Each character configuration object contains these following keys:
            'skewness': the skew angle of character
            'character_normalization_mode': an integer specifcy how to
                normalize resized characters
            'space': the space distance to the next character
            'bottom': the bottom padding
            'height': height of character (in pixel)
            'width_noise': the amount to multiply with character width

        # Arguments
            text [str]: the text to generate string
            default_config [dict]: the default config value

        # Returns
            [obj]: the configuration object
        """
        # config = {'angle': 0, 'text': []}
        # text_config = {'space': 0}

        # General rules for each character
        is_space_close = False  # random.random() > 0.8
        random.seed()
        

        piecewise_augment = 1 if random.random() < 0.3 else 0
        if piecewise_augment == 1:
            piecewise_augment = 2 if random.random() < 0.5 else 1
        # End general rules

        # each text configuration
        text_config = []
        last_bottom = 0
        space = configAugment.getRandomSpaceCharPT()
        for _idx, each_char in enumerate(text):
            text_config.append({
                'piecewise_augment': piecewise_augment,
                'space': space,
                'font_draw': font_draw,
                'font_check': font_check
            })

        return {
            'font': font,
            'text': text_config
        }

    def _generate_single_image(self, char, config):
        """Generate a character image

        # Arguments
            char [str]: a character to generate image
            config [obj]: a configuration object for this image

        # Returns
            [np array]: the generated image for that specific string
        """
        if not self._is_char_supported_by_font(char, config['font_check']):
            return None

        image = Image.new('L', (150, 100), 255)
        draw = ImageDraw.Draw(image)
        draw.text((15, 15), char, font=config['font_draw'], fill=0)

        image = np.array(image)

        if len(np.unique(image)) < 2:
            return None

        if config['piecewise_augment'] == 0:
            image = image
        elif config['piecewise_augment'] == 1:
            image = self.augment.char_piecewise_affine_1.augment_image(image)
            image = cv2.threshold(image, 0, 255,
                                  cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        elif config['piecewise_augment'] == 2:
            image = self.augment.char_piecewise_affine_2.augment_image(image)
            image = cv2.threshold(image, 0, 255,
                                  cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        image = trim_image_horizontally(image)
        return image

    def _generate_sequence_image(self, text, debug=True):
        """Generate string image of a given text

        # Arguments
            text [str]: the text that will be used to generate image

        # Returns
            [np array]: the image generated
            [str]: the text label
            [list of str]: list of missing characters
        """
    
        #remove not support characters by font
        random.seed()
        font = random.choice(self.fonts)
        isOk = False
        while not isOk:
            try:
                font_draw = ImageFont.truetype(font, 60)
                font_check = TTFont(font)
                isOk = True
            except:
                random.seed()
                font = random.choice(self.fonts)
                None

        text = self._remove_unknown_characters(text, font_check)
        if len(text)== 0:
            text = str(random.randint(0,100))
            
        config = self._get_config(text, font, font_draw, font_check, {})

        label = ''
        missing_chars = []
        base_image = np.zeros((100, len(text) * 100), np.uint8)
        current_horizontal = 0
        for _idx, each_char in enumerate(text):
            if each_char == ' ':
                current_horizontal += config['text'][_idx]['space']
                current_horizontal = max(0, current_horizontal)
                current_horizontal += random.randint(12, 25)
                label += each_char
                continue

            char_image = self._generate_single_image(each_char,
                                                     config['text'][_idx])
            if char_image is not None:
                char_image = 255 - char_image

                current_horizontal += config['text'][_idx]['space']
                current_horizontal = max(0, current_horizontal)
                next_horizontal = current_horizontal + char_image.shape[1]
                base_image[:, current_horizontal:next_horizontal] = (
                    np.bitwise_or(
                        char_image,
                        base_image[:, current_horizontal:next_horizontal]))
                current_horizontal = next_horizontal
                label += each_char
            else:
                missing_chars.append(each_char)

        image = 255 - base_image
        if len(np.unique(image)) < 2:
            if debug:
                return None, label, {
                    'missing_chars': missing_chars
                }
            else:
                return None, label

        image = crop_image(image)
        image = self.augment.augment_line(image)
        if self.is_binary:
            image = cv2.threshold(
                image, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        return image, label, font

    def load_fonts(self, font_folder):
        """Load all the font files recursively in the font folder

        # Arguments
            font_folder [str]: the folder containing all the fonts
        """
        self.fonts = []
        for each_ext in self.FONT_EXTENSIONS:
            self.fonts += glob.glob(
                os.path.join(font_folder, '**', '*.{}'.format(each_ext)),
                recursive=True)

        if len(self.fonts) == 0:
            print(':WARNING: no font loaded from {}'.format(font_folder))
        else:
            if self.verbose > 2:
                print(':INFO: {} fonts loaded'.format(len(self.fonts)))

    def load_background_image_files(self, folder_path):
        """Load background image files

        # Arguments
            folder_path [str]: the path to folder that contains background
        """
        if self.is_binary:
            print(':WARNING: background image files are not loaded for binary '
                  'generation mode.')
        else:
            self.augment.add_background_image_noises(folder_path)

    def initialize(self):
        """Initialize the generator

        This method will be called when all text and image files are loaded. It
        will:
            1. check for missing characters
            2. construct label_2_char, char_2_label
            3. randomize corpus_lines
        """

        super(PrintedLineGenerator, self).initialize()

