import abc
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

import cv2
import numpy as np
import numpy.random as random

from utils.synthesizeData.helper.helper_controller import TrainingHelper

class BaseOCRGenerator(metaclass=abc.ABCMeta):

    def __init__(self, height=64, allowed_chars=None, helper=None,
        is_binary=False, verbose=2):
        """Initialize the object

        # Arguments
            height [int]: the height of text line
            helper [TrainingHelper object]: use as hook when postprocess result
            allowed_chars [str]: the path to a text file of allowed character.
                Each character a line, and each character should be in
                ordinal form
            is_binary [bool]: whether this is a binary or grayscale image
                generator
            verbose [int]: the verbosity level
        """
        if not isinstance(height, int):
            raise ValueError('`height` should be an integer')

        if not isinstance(verbose, int):
            raise ValueError('`verbose` should be an integer')

        self.height = height        # the height of generated image
        self.verbose = verbose
        self.is_binary = is_binary
        self.folder_path = None
        self.folder_list_files = None

        # text configuration
        # self.conversion_table = load_conversion_table()
        self.corpus_lines = []      # list of text strings
        self.corpus_size = 0

        # utility
        self.helper = TrainingHelper() if helper is None else helper
        self.iterations = 0
        self.allowed_chars = None
        if allowed_chars is not None:
            if check_allowed_char_version(allowed_chars):
                with open(allowed_chars, 'r') as f_in:
                    self.allowed_chars = [
                        each_line for each_line in f_in.read().splitlines()]
                    self.allowed_chars = set(self.allowed_chars)
            else:
                print(':WARNING: number allowed_char text file is deprecated')
                self.allowed_chars = np.loadtxt(allowed_chars, dtype=int)
                self.allowed_chars = set(
                    [chr(each_ord) for each_ord in self.allowed_chars])


    ###################
    # Utility methods #
    ###################
    def _is_number(self, char):
        """Whether the character is a number

        # Arguments
            char [str]: the character to check

        # Returns
            [bool]: True if the character is a number, False otherwise
        """
        if 48 <= ord(char) <= 57:
            return True

        return False


    ######################
    # Generation methods #
    ######################
    @abc.abstractmethod
    def _remove_unknown_characters(self, text):
        pass

    @abc.abstractmethod
    def _get_config(self):
        pass

    @abc.abstractmethod
    def _generate_single_image(self):
        pass

    @abc.abstractmethod
    def _generate_sequence_image(self, text, debug=True):
        pass


    #####################
    # Interface methods #
    #####################
    def get_image(self, image_path, label_file=None, is_binarized=False):
        """Get the image and label from image_path

        The input image should have dark text on white background. The filename
        for for the image should also contains label, and have the form of
        <idx_number>_<label>.[png/jpg]

        # Arguments
            image_path [str]: the path to stored image
            label_file [str or dictionary]: the file containing label. If it is
                a function, then it will take `image_path` and returns the
                label, if it is a dictionary, then the key should be
                image_path's filename, if it is None, then label is the
                filename
            is_binarized [bool]: whether the loaded image is already in
                binary form

        # Returns
            [np array]: the image in binary form (0 - 1) or grayscale form
                (0 - 255)
            [str]: the corresponding label
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.is_binary:
            if not is_binarized:
                image = cv2.GaussianBlur(image, (3, 3), 0)
                image = cv2.threshold(image, 0, 255,
                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            image = (image / 255).astype(np.uint8)

        if label_file is None:
            filename = os.path.basename(image_path)
            filename, _ = os.path.splitext(filename)
            label = '_'.join(filename.split('_')[1:])
        elif isinstance(label_file, dict):
            filename = os.path.basename(image_path)
            # filename, _ = os.path.splitext(filename)
            label = label_file.get(filename, None)
        elif callable(label_file):
            label = label_file(image_path)
        if label:
            # return image, normalize_text(label, self.conversion_table)
            return image, label
        else:
            return image, None


    def generate_images(self, start=0, end=None, save_dir=None,
        label_json=False, debug=True, text_lines=None):
        """Generate images into png files

        # Arguments
            start [int]: the first text used to generate image
            end [int]: the last text used to generate image. If None, then it
                will generate to the last text corpus_lines
            save_dir [str]: path to the folder
            label_json [bool]: whether to generate a json label file
            debug [bool]: whether to retrieve debug information
            text_lines [str or list of str]: the text line to specifically
                generate. If None, then generate from self.corpus_lines
        """
        max_width = 0
        missing_chars = set([])
        if end is None:
            end = len(self.corpus_lines)

        labels = {}
        now = int(time.time())
        if save_dir == None:
            save_dir = str(now)
        os.makedirs(save_dir, exist_ok=True)

        if isinstance(text_lines, str):
            text_lines = [text_lines]

        debug_info = {}
        corpus = (
            self.corpus_lines[start:end]
            if text_lines is None
            else text_lines)
        for _idx, each_text in enumerate(corpus):
            image, chars, each_debug_info = self._generate_sequence_image(
                text=each_text, debug=True)

            if image is None:
                print(':WARNING: image is None for text {}'.format(each_text))
                continue

            if len(np.unique(image)) < 2:
                print(':WARNING: image has just a single color for text {}'
                    .format(each_text))
                continue

            if chars == '':
                print(':WARNING: there isn\'t any character for text {}'
                    .format(each_text))
                continue

            if max_width < image.shape[1]:
                max_width = image.shape[1]

            if len(chars) < len(each_text):
                print(' Label: {} --> Move to: {} --> Debug: {}'
                    .format(each_text, chars, each_debug_info))

            if self.is_binary:
                image = (image * 255).astype(np.uint8)

            if label_json:
                filename = '{:0>6}_{}.png'.format(_idx, now)
            else:
                filename = '{:0>6}_{}.png'.format(_idx, chars)
            
            cv2.imwrite(os.path.join(save_dir, filename), image)
            labels[filename] = chars
            debug_info[filename] = each_debug_info
            missing_chars = missing_chars.union(
                set(each_debug_info['missing_chars']))

        if label_json:
            dump_json(
                labels,
                os.path.join(save_dir, 'labels.json'),
                sort_keys=True)
        
        if debug:
            dump_json(debug_info, os.path.join(save_dir, 'debug.json'))

        if self.verbose >= 2:
            print('Max-Width', max_width)
            print('Missing:', missing_chars)

    def get_helper(self):
        """Return the helper"""
        return self.helper

    def initialize(self):
        """Intialize the generator, augmentator, helper"""
        # self.augment.build_augmentators()
        a=1

    def initialize_folder(self, folder_path):
        """Initialize folder to generate images from folders

        # Arguments
            folder_path [str]: the path to folder containing images
        """
        self.folder_path = folder_path
        self.folder_list_files = glob.glob(os.path.join(folder_path, '*.png'))
        self.folder_list_files += glob.glob(os.path.join(folder_path, '*.jpg'))

