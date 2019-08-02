# Load ETL dataset
# =============================================================================
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import argparse
import os
import struct

import bitstring
import cv2
import numpy as np
from PIL import Image
import random


from main_source.utils.synthesizeData.utils.normalize import normalize_char
from main_source.utils.synthesizeData.utils.char_code import (jis0201_to_unicode, jis0208_to_unicode,
    kata_to_hira)

IS_DEBUG = True

def etl1_loader(file_path, dictChar, verbose=0):
    """Load an ETL1 file, and return numpy image file and labels

    ## NOTE: the image has limited pixel ranges, to view an image, multiply it
    with 28.

    # Arguments
        file_path [str]: the path to ETL1 file

    # Returns
        [numpy array]: image X, in numpy array form
        [list of str]: the list of corresponding labels
    """
    X, y = [], []
    file_size = os.stat(file_path).st_size
    skip = 0
    uniqueLabel = {}

    with open(file_path, 'rb') as f:
        while skip * 2052 < file_size:
            f.seek(skip*2052)
            s = f.read(2052)
            r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)

            try:

                iF = np.asarray(
                    Image.frombytes('F', (64, 63), r[18], 'bit', 4))
                label = normalize_char(jis0201_to_unicode(r[3]))

                if IS_DEBUG:
                    iFNew = iF *28

                    # filename = 'C:\\Users\\ABC\\Desktop\\deletetemp\\GenDataHWBB\\synthesizedKana\\'+'_'+label+'_'+str(skip)+'_01_test_ori'+'.png'
                    # cv2.imwrite(filename, iF)

                    filename = 'C:\\Users\\ABC\\Desktop\\deletetemp\\GenDataHWBB\\synthesizedKana\\'+'_'+label+'_'+str(skip)+'_02_test_grey'+'.png'
                    cv2.imwrite(filename, iFNew)

                    binary = iFNew < 60
                    binaryInt = binary.astype(int)*255

                    filename = 'C:\\Users\\ABC\\Desktop\\deletetemp\\GenDataHWBB\\synthesizedKana\\'+'_'+label+'_'+str(skip)+'_03_test_binary_60'+'.png'
                    cv2.imwrite(filename, binaryInt)

                    binary2 = iFNew < 70
                    binaryInt2 = binary.astype(int) * 255

                    filename = 'C:\\Users\\ABC\\Desktop\\deletetemp\\GenDataHWBB\\synthesizedKana\\' + '_' + label + '_' + str(
                        skip) + '_03_test_binary_70' + '.png'
                    cv2.imwrite(filename, binaryInt2)



                X.append(iF)
                y.append(label)

                if uniqueLabel.get(label) is None:
                    uniqueLabel[label] = 1
                else:
                    uniqueLabel[label] = uniqueLabel[label] + 1

                if dictChar.get(label) is None:
                    dictChar[label] = 1
                else:
                    dictChar[label] = dictChar[label] + 1

            except KeyError as error:
                if verbose > 0:
                    print(':WARNING: {}'.format(error))
                skip += 1
                continue

            skip += 1

        rs = np.asarray(X, dtype=np.uint8)

        print(file_path)
        print(uniqueLabel)

    return rs, y, dictChar


def etl3_loader(file_path):
    """Load an ETL1 file, and return numpy image file and labels

    @NOTE: the image has limited pixel ranges, to view an image, multiply it
    with 30.

    # Arguments
        file_path [str]: the path to ETL3 file

    # Returns
        [numpy array]: image X, in numpy array form
        [list of str]: the list of coressponding labels
    """
    file_size = os.stat(file_path).st_size
    skip = 0
    uniqueLabel = {}

    X, y = [], []
    each_file = bitstring.ConstBitStream(filename=file_path)
    while skip * 6 * 3936 < file_size:
        each_file.pos = skip * 6 * 3936
        r = each_file.readlist(
            '2*uint:36,uint:8,pad:28,uint:8,pad:28,4*uint:6,pad:12,'
            '15*uint:36,pad:1008,bytes:21888')
        iF = Image.frombytes('F', (r[18], r[19]), r[-1], 'bit', 4)
        X.append(np.asarray(iF))

        label = normalize_char(str(jis0201_to_unicode(r[2])))
        y.append(normalize_char(str(jis0201_to_unicode(r[2]))))

        if uniqueLabel.get(label) is None:
            uniqueLabel[label] = 1
        else:
            uniqueLabel[label] = uniqueLabel[label] + 1

        skip += 1

    print(uniqueLabel)

    return np.asarray(X, dtype=np.uint8), y


def etl4_loader(file_path):
    """Load an ETL4 file, and return numpy image file and labels

    @NOTE: the image has limited pixel ranges, to view an image, multiply it
    with 28.

    # Arguments
        file_path [str]: the path to ETL3 file

    # Returns
        [numpy array]: image X, in numpy array form
        [list of str]: the list of coressponding labels
    """
    X, y  = [], []
    file_size = os.stat(file_path).st_size

    uniqueLabel = {}

    skip = 0
    each_file = bitstring.ConstBitStream(filename=file_path)
    while skip * 6 * 3936 < file_size:
        each_file.bytepos = skip * 6 * 3936
        r = each_file.readlist(
            '2*uint:36,uint:8,pad:28,uint:8,pad:28,4*uint:6,pad:12,'
            '15*uint:36,pad:1008,bytes:21888')

        iF = Image.frombytes('F', (r[18], r[19]), r[-1], 'bit', 4)
        X.append(np.array(iF))
        label = normalize_char(jis0201_to_unicode(r[2]))
        y.append(kata_to_hira(normalize_char(jis0201_to_unicode(r[2]))))
        skip += 1

        if uniqueLabel.get(label) is None:
            uniqueLabel[label] = 1
        else:
            uniqueLabel[label] = uniqueLabel[label] + 1

    print(uniqueLabel)
    return np.asarray(X, dtype=np.uint8), y


def etl5_loader(file_path):
    """Load an ETL5 file, and return numpy image file and labels

    @NOTE: the image has limited pixel ranges, to view an image, multiply it
    with 28.

    # Arguments
        file_path [str]: the path to ETL3 file

    # Returns
        [numpy array]: image X, in numpy array form
        [list of str]: the list of coressponding labels
    """
    X, y = [], []
    file_size = os.stat(file_path).st_size

    uniqueLabel = {}

    skip = 0
    each_file = bitstring.ConstBitStream(filename=file_path)

    while skip * 6 * 3936 < file_size:
        each_file.bytepos = skip * 6 * 3936
        r = each_file.readlist(
            '2*uint:36,uint:8,pad:28,uint:8,pad:28,4*uint:6,pad:12,'
            '15*uint:36,pad:1008,bytes:21888')

        iF = Image.frombytes('F', (r[18], r[19]), r[-1], 'bit', 4)
        X.append(np.array(iF))
        y.append(normalize_char(jis0201_to_unicode(r[2])))
        skip += 1
        label = normalize_char(jis0201_to_unicode(r[2]))
        if uniqueLabel.get(label) is None:
            uniqueLabel[label] = 1
        else:
            uniqueLabel[label] = uniqueLabel[label] + 1

    print(uniqueLabel)
    return np.asarray(X, dtype=np.uint8), y


def etl6_loader(file_path, verbose=0):
    """Load an ETL6 file, and return numpy image file and labels

    ## NOTE: the image has limited pixel ranges, to view an image, multiply it
    with 28.

    # Arguments
        file_path [str]: the path to ETL1 file

    # Returns
        [numpy array]: image X, in numpy array form
        [list of str]: the list of corresponding labels
    """
    X, y = [], []
    file_size = os.stat(file_path).st_size
    skip = 0
    uniqueLabel = {}

    with open(file_path, 'rb') as f:
        while skip * 2052 < file_size:
            f.seek(skip * 2052)
            s = f.read(2052)
            r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)

            try:
                iF = np.asarray(
                    Image.frombytes('F', (64, 63), r[18], 'bit', 4))
                label = normalize_char(jis0201_to_unicode(r[3]))
                X.append(iF)
                y.append(label)

                # label = normalize_char(jis0201_to_unicode(r[2]))
                if uniqueLabel.get(label) is None:
                    uniqueLabel[label] = 1
                else:
                    uniqueLabel[label] = uniqueLabel[label] + 1

                if IS_DEBUG and (label == '゚' or label == '゙'):
                    iFNew = iF *28

                    # filename = 'C:\\Users\\ABC\\Desktop\\deletetemp\\GenDataHWBB\\synthesizedKana\\'+'_'+label+'_'+str(skip)+'_01_test_ori'+'.png'
                    # cv2.imwrite(filename, iF)

                    filename = 'C:\\Users\\ABC\\Desktop\\deletetemp\\GenDataHWBB\\synthesizedKana\\'+'6_'+label+'_'+str(skip)+'_02_test_grey'+'.png'
                    cv2.imwrite(filename, iFNew)

                    binary = iFNew < 60
                    binaryInt = binary.astype(int)*255

                    filename = 'C:\\Users\\ABC\\Desktop\\deletetemp\\GenDataHWBB\\synthesizedKana\\'+'6_'+label+'_'+str(skip)+'_03_test_binary_60'+'.png'
                    cv2.imwrite(filename, binaryInt)

                    binary2 = iFNew < 70
                    binaryInt2 = binary.astype(int) * 255

                    filename = 'C:\\Users\\ABC\\Desktop\\deletetemp\\GenDataHWBB\\synthesizedKana\\' + '6_' + label + '_' + str(
                        skip) + '_03_test_binary_70' + '.png'
                    cv2.imwrite(filename, binaryInt2)

            except KeyError as error:
                if verbose > 0:
                    print(':WARNING: {}'.format(error))
                skip += 1
                continue

            skip += 1
    print(uniqueLabel)
    return np.asarray(X, dtype=np.uint8), y


def etl7_loader(file_path, verbose=0):
    """Load an ETL1 file, and return numpy image file and labels

    @NOTE: the image has limited pixel ranges, to view an image, multiply it
    with 16.

    # Arguments
        file_path [str]: the path to ETL1 file

    # Returns
        [numpy array]: image X, in numpy array form
        [list of str]: the list of corresponding labels
    """
    X, y = [], []
    file_size = os.stat(file_path).st_size
    skip = 0
    uniqueLabel = {}

    with open(file_path, 'rb') as f:
        while skip * 2052 < file_size:
            f.seek(skip * 2052)
            s = f.read(2052)
            r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)

            try:
                iF = np.asarray(
                    Image.frombytes('F', (64, 63), r[18], 'bit', 4))
                label = normalize_char(jis0201_to_unicode(r[3]))
                X.append(iF)
                y.append(label)

                if uniqueLabel.get(label) is None:
                    uniqueLabel[label] = 1
                else:
                    uniqueLabel[label] = uniqueLabel[label] + 1

            except KeyError as error:
                if verbose > 0:
                    print(':WARNING: {}'.format(error))
                skip += 1
                continue

            skip += 1
    print(uniqueLabel)
    return np.asarray(X, dtype=np.uint8), y


def etl8b_loader(file_path):
    """Load an ETL8 file, and return numpy image file and labels

    ## NOTE: the image has limited pixel ranges, to view an image, multiply it
    with 255.

    # Arguments
        file_path [str]: the path to ETL8B file

    # Returns
        [numpy array]: image X, in numpy array form
        [list of str]: the list of corresponding labels
    """
    X, y = [], []
    file_size = os.stat(file_path).st_size
    skip = 0
    uniqueLabel = {}

    with open(file_path, 'rb') as f:
        while (skip + 1) * 512 < file_size:
            f.seek((skip+1)*512)
            s = f.read(512)
            r = struct.unpack('>2H4s504s', s)

            try:
                img = np.asarray(Image.frombytes('1', (64,63), r[3], 'raw'))
                label = jis0208_to_unicode(hex(r[1]).split('x')[1].upper())
                X.append(img)
                y.append(label)

                if uniqueLabel.get(label) is None:
                    uniqueLabel[label] = 1
                else:
                    uniqueLabel[label] = uniqueLabel[label] + 1

            except KeyError as error:
                print(':WARNING: {}'.format(error))
                skip += 1
                continue


            skip += 1

    print(uniqueLabel)
    return np.asarray(X, dtype=np.uint8), y


def etl8g_loader(file_path):
    """Load ETL-8G dataset

    ## NOTE: each record in X has pixel value in range [0, 9]. To view an
    image, multiply it with 28.

    # Arguments:
        file_path [str]: the path to ETL8G file

    # Returns
        [3D np array]: the X record of shape (num_data x height x width)
        [list of str]: the labels
    """
    X, y = [], []
    file_size = os.stat(file_path).st_size
    skip = 0
    uniqueLabel = {}
    with open(file_path, 'rb') as f:
        while (skip * 8199) < file_size:
            f.seek(skip * 8199)
            s = f.read(8199)
            r = struct.unpack('>2H8sI4B4H2B30x8128s11x', s)

            try:
                img = np.asarray(
                    Image.frombytes('F', (128,127), r[14], 'bit', 4))
                label = jis0208_to_unicode(hex(r[1]).split('x')[1].upper())
                X.append(img)
                y.append(label)
                if uniqueLabel.get(label) is None:
                    uniqueLabel[label] = 1
                else:
                    uniqueLabel[label] = uniqueLabel[label] + 1

            except KeyError as error:
                print(':WARNING: {}'.format(error))
                skip += 1
                continue

            skip += 1
    print(uniqueLabel)
    return np.asarray(X, dtype=np.uint8), y


def etl9b_loader(file_path):
    """Load the ETL9B dataset

    X is the numpy array of shape N x 63 x 64, with max pixel value is True
    and min pixel value is False.
    y is the list of JIS encoding. Refer here for JIS <-> Unicode conversion:
    ftp://ftp.unicode.org/Public/MAPPINGS/OBSOLETE/EASTASIA/JIS/JIS0208.TXT

    ## NOTE: to view image, multiply it with 255.

    # Arguments:
        file_path [str]: the path to ETL8G file

    # Returns
        [3D np array]: the X record of shape (num_data x height x width)
        [list of str]: the labels
    """
    X, y = [], []
    file_size = os.stat(file_path).st_size
    record_size = 576
    skip = 0

    with open(file_path, 'rb') as f:
        while (skip * record_size) < file_size:
            f.seek(skip * record_size)
            s = f.read(record_size)
            r = struct.unpack('>2H4s504s64x', s)

            try:
                img = np.asarray(Image.frombytes('1', (64,63), r[3], 'raw'))
                label = jis0208_to_unicode(hex(r[1]).split('x')[1].upper())
                X.append(img)
                y.append(label)
            except KeyError as error:
                print(':WARNING: {}'.format(error))
                skip += 1
                continue

            skip += 1

    return np.asarray(X, dtype=np.uint8), y


def etl9g_loader(file_path):
    """Load ETL9G dataset

    X is the numpy array of shape N x 127 x 128, with max pixel value is 11.0
    and min pixel value is 0.0
    y is the list of JIS encoding. Refer here for JIS <-> Unicode conversion:
    ftp://ftp.unicode.org/Public/MAPPINGS/OBSOLETE/EASTASIA/JIS/JIS0208.TXT

    ## NOTE: to view image, multiply it with 23

    # Returns
        [3D np array]: the X record of shape (num_data x height x width)
        [list of str]: the labels
    """
    X, y = [], []
    file_size = os.stat(file_path).st_size
    record_size = 8199
    skip = 0

    with open(file_path, 'rb') as f:
        while (skip * record_size) < file_size:
            f.seek(skip * record_size)
            s = f.read(record_size)
            r = struct.unpack('>2H8sI4B4H2B34x8128s7x', s)
            img = np.asarray(Image.frombytes('F', (128,127), r[14], 'bit', 4))
            label = hex(r[1]).split('x')[1]

            X.append(img)
            y.append(jis0208_to_unicode(label.upper()))

            skip += 1

    return np.asarray(X, dtype=np.uint8), y


def dump_samples(X, filename, n_row, n_col):
    """Dump samples into disk

    # Arguments
        X [np array]: image of [C x H x W]
        filename [str]: the sample
    """
    permutation = np.random.permutation(X.shape[0])
    X[permutation,:,:] = X

    rows = []
    for _idx in range(n_row):
        images = [X[idx] for idx in range(n_col*_idx, n_col*(_idx+1))]
        images = np.concatenate(images, axis=1)
        rows.append(images)

    image = np.concatenate(rows, axis=0)
    cv2.imwrite(filename, image)

    return image


def extract_maru_tenten_from_etl6_7(file_path,  multiply, dict):
    file_size = os.stat(file_path).st_size
    skip = 0

    with open(file_path, 'rb') as f:
        while skip * 2052 < file_size:
            f.seek(skip * 2052)
            s = f.read(2052)
            r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)

            try:
                iF = np.asarray(
                    Image.frombytes('F', (64, 63), r[18], 'bit', 4))
                label = normalize_char(jis0201_to_unicode(r[3]))

                if label == '゚' or label == '゙':
                    iFNew = iF * multiply
                    if dict.get(label) is None:
                        dict[label] = []
                    dict[label].append(iFNew)

            except KeyError as error:
                skip += 1
                continue

            skip += 1
    return dict