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

from main_source.utils.synthesizeData.generator.base_generator import BaseOCRGenerator
from main_source.utils.synthesizeData.helper.helper_controller import FFG_Helper

import main_source.utils.synthesizeData.utils.constants as constants
from main_source.utils.synthesizeData.utils.misc import (get_truncated_normal,
                                             check_allowed_char_version, dump_json)
import main_source.utils.synthesizeData.configRandomParameter.rdcf_ffg_field8 as configAugment

# from utils.synthesizeData.utils.normalize import load_conversion_table, normalize_text

from main_source.utils.synthesizeData.utils.preprocessing import (
    crop_image, adjust_stroke_width,
    normalize_grayscale_color, unsharp_masking, skeletonize_image,
    trim_image_horizontally)
from main_source.utils.synthesizeData.utils.augment import (HandwritingMildAugment, PrintMildAugment)


IS_DEBUG_LOG = False


class FFGKatakanaGenerator(BaseOCRGenerator):
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


        super(FFGKatakanaGenerator, self).__init__(
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
            cv2.INTER_CUBIC)
        self.configKata = configAugment.getKanakataConfig()

        # utility
        # self.augment = (
        #     augmentor(is_binary=is_binary) if augmentor is not None
        #     else HandwritingMildAugment(is_binary=is_binary))
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
                if each_char == '・':
                    exist.append(each_char)
                else:
                    missing_chars.add(each_char)
            else:
                exist.append(each_char)

        return ''.join(exist), missing_chars

    def _get_config(self, text, default_config={}):
        height_line = configAugment.getRandomHeightKata()
        fix_all_line_height = 80
        fix_width_each_box = 58
        deltaCharHeight = 5
        margin_bottom_top = 5

        random.seed()
        bottom_base_line = random.randint(margin_bottom_top, fix_all_line_height - height_line - margin_bottom_top)

        # each text configuration
        text_config = []
        for _idx, each_char in enumerate(text):
            random.seed()
            text_config.append({'delta_bottom': random.randint(-2, 5),
                                'bottm_baseline': bottom_base_line,
                                })

        return {
            'text': text_config,
        }

    def _get_config_kata(self, text, default_config={}):
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

        '''
                #there are 13 box each line
                #width: 760  #height:80
                #2 lines: 1500 #height: 80
                #==>> calculate width each box: 58
                #each line will be choose a:
                  - height
                  - padding bottom
                  - padding top
                #each character will be choose:
                  - height => calculate width
                  - padding left right base on width of box and width of character after resizing
        '''
        height_line = configAugment.getRandomHeightKata()

        random.seed()
        bottom_base_line = random.randint(self.configKata['MARGIN_BOTTOM_TOP'], min(self.configKata['FIX_ALL_LINE_HEIGHT']-height_line-self.configKata['MARGIN_BOTTOM_TOP'], 25))



        # each text configuration
        text_config = []
        for _idx, each_char in enumerate(text):
            random.seed()
            bottom = bottom_base_line + random.randint(-2,5)
            random.seed()
            desired_height = height_line+random.randint(-self.configKata['DELTA_CHAR_HEIGHT'],self.configKata['DELTA_CHAR_HEIGHT'])
            if each_char == '゙' or each_char == '゚':
                random.seed()
                bottom = bottom_base_line + random.randint(5,10)
                desired_height = height_line + random.randint(0,
                                                              self.configKata['DELTA_CHAR_HEIGHT'])
            if each_char == '.':
                random.seed()
                bottom = bottom_base_line + random.randint(-5,0)

            top = max(0, self.configKata['FIX_ALL_LINE_HEIGHT'] - bottom - desired_height)
            text_config.append({'bottom': bottom,
                                'height': desired_height,
                                'top': top,
                                'height_line': height_line,
                                'bottom_base': bottom_base_line
                                })

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
        '''
        #there are 13 box each line
        #width: 760  #height:80
        #2 lines: 1500 #height: 80
        #==>> calculate width each box: 58
        #each line will be choose a:
          - height
          - padding bottom
          - padding top
        #each character will be choose:
          - height => calculate width
          - padding left right base on width of box and width of character after resizing
        '''

        if char == '・':
            charChoice = '.'
        else:
            charChoice = char

        if charChoice not in self.char_2_imgs.keys():
            return None

        isFinishChoose = False
        while not isFinishChoose:
            choice = self._get_random_choice_index(char)

            image = self.char_2_imgs[charChoice][choice]
            # numberRandom = random.randint(0, 10000)
            # cv2.imwrite('C:\\Users\\ABC\\Desktop\\deletetemp\\GenDataHWBB\\synthesizedKana\\test_bf'+str(numberRandom)+'_'+str(choice)+'.png', image)

            # # rotate image
            # image = rotate(image, config['skewness'], order=1,
            #                cval=self.background_value)

            # resize image
            if charChoice not in constants.NOT_RESIZE:
                image = self._resize_character_kata(image, config['height'])
            else:
                random.seed()
                height, _ = image.shape
                if char == '.':
                    config['bottom'] = config['bottom_base']
                    config['top'] = max(0,self.configKata['FIX_ALL_LINE_HEIGHT'] - config['bottom'] - height)
                else:

                    config['bottom'] = config['bottom_base'] + int(config['height_line']/2)
                    config['top'] = max(0,self.configKata['FIX_ALL_LINE_HEIGHT'] - config['bottom'] - height)



            # add horizontal space and bottom space
            _, width = image.shape
            if width < self.configKata['FIX_BOX_WIDTH']-8:
                remain = int((self.configKata['FIX_BOX_WIDTH']-width)/2) + 3
                random.seed()
                left = max(0, random.randint(5, remain))
                if char == '゙' or char == '゚':
                    random.seed()
                    left = random.randint(3, 7)
                elif char == ')':
                    left = 0
                right = max(0, self.configKata['FIX_BOX_WIDTH']-left-width)
                if right == 0:
                    left = max(0, self.configKata['FIX_BOX_WIDTH']-right-width)

                image = np.pad(image, ((config['top'], config['bottom']), (left, right)),'constant', constant_values=self.background_value)

                # cv2.imwrite('C:\\Users\\ABC\\Desktop\\deletetemp\\GenDataHWBB\\synthesizedKana\\test_af'+str(numberRandom)+'.png', image)
                isFinishChoose = True
            else:
                continue
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

        ### remember add space to fix 13 character each line or 26 char for multiple lines

        # remove characters in text that not exist in etl character images
        text, missChar = self._remove_unknown_characters(text)
        # print('text after remove: ', text, missChar)
        #normalize text


        if len(text) == 0:
            text = str(random.randint(0, 100))

        # self.character_height = configAugment.getConfigHeightHW()  # to help resize to self.height
        # self.height = self.character_height
        config = self._get_config_kata(text, default_config)
        

        # Calculate the average height of a character
        if self.deterministic:
            indices = {each_char: self._get_random_choice_index(each_char)
                           for each_char in list(set(text))}
        else:
            indices = {}


        for _idx, each_char in enumerate(text):
            if (each_char == '・'):
                each_charChoice = '.'
            else:
                each_charChoice = each_char

            char_images.append(self._generate_single_image(
                    each_char, config['text'][_idx], indices.get(each_charChoice, None))
                )

        # desired_height = max_height + 4
        # norm_img_seq = []
        # for each_img in char_images:
        #     top_pad = desired_height - each_img.shape[0] - 3
        #     norm_img_seq.append(np.pad(each_img, ((top_pad, 3), (0, 0)),
        #                                    mode='constant', constant_values=self.background_value))



        image = np.concatenate(char_images, axis=1)
        _, width = image.shape
        paddingValue = 0
        if len(text) <= 13:
            paddingValue = max(0, self.configKata['ONE_LINE_WIDTH']- width)
        else:
            paddingValue = max(0, self.configKata['TWO_LINE_WIDTH'] - width)

        image = np.pad(image, ((0, 0), (0, paddingValue)),
                           'constant', constant_values=self.background_value)
        # cv2.imwrite('C:\\Users\\ABC\\Desktop\\deletetemp\\GenDataHWBB\\synthesizedKana\\test_final.png', image)


        # add padding space behind the final characters
        # image = np.pad(image, ((0, 0), (0, 10)),
        #                    mode='constant', constant_values=self.background_value)

        # image = self.augment.augment_line(image)

        return image, text

    def _get_random_choice_index(self, each_char):
        if (each_char == '・'):
            each_char = '.'
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

    def _resize_character_kata(self, image, desired_height):
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
        desired_width = int(original_width * desired_height / original_height)
        # Resize the character
        image = cv2.resize(image, (desired_width, desired_height),
                           interpolation=self.interpolation)

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
        super(FFGKatakanaGenerator, self).initialize()

    def generate_images(self, start=0, end=None, save_dir=None,
                        label_json=False, mode=None, debug=True,
                        text_lines=None, **kwargs):
        """Generate the images

        Subclass from super, in order to determine batch mode. Other
        arguments can be seen in the subclass

        # Arguments
            mode [int]: the mode of this generator
        """

        return super(FFGKatakanaGenerator, self).generate_images(
            start, end, save_dir, label_json, debug=debug, text_lines=text_lines
        )


# class PrintedLineGenerator(BaseOCRGenerator):
#     """
#     This generator assumes there are:
#         (1) a collection of text files, from which the content of images will
#             be generated;
#         (2) a collection of font files
#
#     Usage example:
#     ```
#     import os
#     from dataloader.generate.image import PrintedLineGenerator
#
#     lineOCR = PrintedLineGenerator()
#     lineOCR.load_fonts('./fonts/')
#     lineOCR.load_text_database('text.txt')
#
#     # to get random characters
#     X, widths, y = lineOCR.get_batch(4, 1800)
#
#     # or to generate 100 images
#     lineOCR.generate_images(start=0, end=100, save_dir='/tmp/Samples')
#     ```
#     """
#     FONT_EXTENSIONS = ['otf', 'ttf', 'OTF', 'TTF']
#
#     def __init__(self, height=74, num_workers=8, allowed_chars=None,
#                  helper=None, augmentor=None, is_binary=False, verbose=2):
#         """Initialize the generator"""
#         super(PrintedLineGenerator, self).__init__(height=height,
#                                                    helper=helper, allowed_chars=allowed_chars, verbose=verbose)
#
#         # image configuration
#         self.fonts = []
#
#         # utility
#         self.augment = PrintMildAugment(is_binary=is_binary, cval=255)
#
#     def _remove_unknown_characters(self, text, font_check):
#         """Check for characters in the text that are not in database.
#
#         # Arguments
#             text [str]: the string to check
#
#         # Returns
#             [str]: the text that have missing characters removed
#             [set of str]: missing characters
#         """
#         exist = []
#         for each_char in text:
#             if self._is_char_supported_by_font(each_char, font_check):
#                 exist.append(each_char)
#             # else:
#                 # print(each_char, '  ---- char is not support by font')
#         return ''.join(exist)
#
#     def _is_char_supported_by_font(self, char, font_check):
#         """Check whether the current font supported to draw `char`
#
#         # Arguments
#             char [str]: the character to check
#             font_check [TTFont]: the font check object
#
#         # Returns
#             [bool]: True if the font support `char`, False otherwise
#         """
#         for cmap in font_check['cmap'].tables:
#             if cmap.isUnicode():
#                 if ord(char) in cmap.cmap:
#                     return True
#
#         return False
#
#     def _get_config(self, text, font,  font_draw, font_check,  default_config={}):
#         """Returns a configuration object for the text
#
#         The idea is that each line of text will have a specific configuration,
#         which then will be used during image generation. The configuration file
#         has the following format: {
#             config_key: config_value
#             'text': [list of {}s with length == len(text), with each {} is a
#                      config for that specific word]
#         }
#
#         The returning object contains these following keys:
#             'text': a list of each character configuration
#
#         Each character configuration object contains these following keys:
#             'skewness': the skew angle of character
#             'character_normalization_mode': an integer specifcy how to
#                 normalize resized characters
#             'space': the space distance to the next character
#             'bottom': the bottom padding
#             'height': height of character (in pixel)
#             'width_noise': the amount to multiply with character width
#
#         # Arguments
#             text [str]: the text to generate string
#             default_config [dict]: the default config value
#
#         # Returns
#             [obj]: the configuration object
#         """
#         # config = {'angle': 0, 'text': []}
#         # text_config = {'space': 0}
#
#         # General rules for each character
#         is_space_close = False  # random.random() > 0.8
#         random.seed()
#
#
#         piecewise_augment = 1 if random.random() < 0.3 else 0
#         if piecewise_augment == 1:
#             piecewise_augment = 2 if random.random() < 0.5 else 1
#         # End general rules
#
#         # each text configuration
#         text_config = []
#         last_bottom = 0
#         space = configAugment.getRandomSpaceCharPT()
#         for _idx, each_char in enumerate(text):
#             text_config.append({
#                 'piecewise_augment': piecewise_augment,
#                 'space': space,
#                 'font_draw': font_draw,
#                 'font_check': font_check
#             })
#
#         return {
#             'font': font,
#             'text': text_config
#         }
#
#     def _generate_single_image(self, char, config):
#         """Generate a character image
#
#         # Arguments
#             char [str]: a character to generate image
#             config [obj]: a configuration object for this image
#
#         # Returns
#             [np array]: the generated image for that specific string
#         """
#         if not self._is_char_supported_by_font(char, config['font_check']):
#             return None
#
#         image = Image.new('L', (150, 100), 255)
#         draw = ImageDraw.Draw(image)
#         draw.text((15, 15), char, font=config['font_draw'], fill=0)
#
#         image = np.array(image)
#
#         if len(np.unique(image)) < 2:
#             return None
#
#         if config['piecewise_augment'] == 0:
#             image = image
#         elif config['piecewise_augment'] == 1:
#             image = self.augment.char_piecewise_affine_1.augment_image(image)
#             image = cv2.threshold(image, 0, 255,
#                                   cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#         elif config['piecewise_augment'] == 2:
#             image = self.augment.char_piecewise_affine_2.augment_image(image)
#             image = cv2.threshold(image, 0, 255,
#                                   cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#
#         image = trim_image_horizontally(image)
#         return image
#
#     def _generate_sequence_image(self, text, debug=True):
#         """Generate string image of a given text
#
#         # Arguments
#             text [str]: the text that will be used to generate image
#
#         # Returns
#             [np array]: the image generated
#             [str]: the text label
#             [list of str]: list of missing characters
#         """
#
#         #remove not support characters by font
#         random.seed()
#         font = random.choice(self.fonts)
#         isOk = False
#         while not isOk:
#             try:
#                 font_draw = ImageFont.truetype(font, 60)
#                 font_check = TTFont(font)
#                 isOk = True
#             except:
#                 random.seed()
#                 font = random.choice(self.fonts)
#                 None
#
#         text = self._remove_unknown_characters(text, font_check)
#         if len(text)== 0:
#             text = str(random.randint(0,100))
#
#         config = self._get_config(text, font, font_draw, font_check, {})
#
#         label = ''
#         missing_chars = []
#         base_image = np.zeros((100, len(text) * 100), np.uint8)
#         current_horizontal = 0
#         for _idx, each_char in enumerate(text):
#             if each_char == ' ':
#                 current_horizontal += config['text'][_idx]['space']
#                 current_horizontal = max(0, current_horizontal)
#                 current_horizontal += random.randint(12, 25)
#                 label += each_char
#                 continue
#
#             char_image = self._generate_single_image(each_char,
#                                                      config['text'][_idx])
#             if char_image is not None:
#                 char_image = 255 - char_image
#
#                 current_horizontal += config['text'][_idx]['space']
#                 current_horizontal = max(0, current_horizontal)
#                 next_horizontal = current_horizontal + char_image.shape[1]
#                 base_image[:, current_horizontal:next_horizontal] = (
#                     np.bitwise_or(
#                         char_image,
#                         base_image[:, current_horizontal:next_horizontal]))
#                 current_horizontal = next_horizontal
#                 label += each_char
#             else:
#                 missing_chars.append(each_char)
#
#         image = 255 - base_image
#         if len(np.unique(image)) < 2:
#             if debug:
#                 return None, label, {
#                     'missing_chars': missing_chars
#                 }
#             else:
#                 return None, label
#
#         image = crop_image(image)
#         image = self.augment.augment_line(image)
#         if self.is_binary:
#             image = cv2.threshold(
#                 image, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#
#         return image, label, font
#
#     def load_fonts(self, font_folder):
#         """Load all the font files recursively in the font folder
#
#         # Arguments
#             font_folder [str]: the folder containing all the fonts
#         """
#         self.fonts = []
#         for each_ext in self.FONT_EXTENSIONS:
#             self.fonts += glob.glob(
#                 os.path.join(font_folder, '**', '*.{}'.format(each_ext)),
#                 recursive=True)
#
#         if len(self.fonts) == 0:
#             print(':WARNING: no font loaded from {}'.format(font_folder))
#         else:
#             if self.verbose > 2:
#                 print(':INFO: {} fonts loaded'.format(len(self.fonts)))
#
#     def load_background_image_files(self, folder_path):
#         """Load background image files
#
#         # Arguments
#             folder_path [str]: the path to folder that contains background
#         """
#         if self.is_binary:
#             print(':WARNING: background image files are not loaded for binary '
#                   'generation mode.')
#         else:
#             self.augment.add_background_image_noises(folder_path)
#
#     def initialize(self):
#         """Initialize the generator
#
#         This method will be called when all text and image files are loaded. It
#         will:
#             1. check for missing characters
#             2. construct label_2_char, char_2_label
#             3. randomize corpus_lines
#         """
#
#         super(PrintedLineGenerator, self).initialize()

