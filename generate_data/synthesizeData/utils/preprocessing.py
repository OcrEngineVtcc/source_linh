# Utility functions to preprocess images
# =============================================================================
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import math

import cv2
import numpy as np
from scipy.ndimage import filters, measurements
from skimage.morphology import skeletonize


def dim0(s):
    """Dimension of the slice list for dimension 0."""
    return s[0].stop - s[0].start


def dim1(s):
    """Dimension of the slice list for dimension 1."""
    return s[1].stop - s[1].start


def max_dim(b):
    if (dim0(b) > dim1(b)):
        return dim0(b)
    else:
        return dim1(b)


def min_dim(pix_index):
    """Return the size of a smaller dimension of an image

    # Arguments
        pix_index [np slice]: slices object

    # Returns
        [int]: the shape of the smaller dimension
    """
    if (dim0(pix_index) > dim1(pix_index)):
        return dim1(pix_index)
    else:
        return dim0(pix_index)


def aspect_normalized(a):
    asp = height(a) * 1.0 / width(a)
    if asp < 1:
        asp = 1.0 / asp
    return asp


def label(image, **kw):
    """Expand scipy.ndimage.measurements.label to work with more data types.

    The default function is inconsistent about the weight types it accepts on 
    different platforms

    # Arguments
        image [np array]: the image

    # Returns
        [np array]: the label of connected components
    """
    try:
        return measurements.label(image, **kw)
    except:
        pass

    types = ["int32", "uint32", "int64", "unit64", "int16", "uint16"]
    for t in types:
        try:
            return measurements.label(array(image, dtype=t), **kw)
        except:
            pass

    # let it raise the same exception as before
    return measurements.label(image, **kw)


def find_objects(image, **kw):
    """Expand scipy.ndimage.measurements.find_objects to work w/ more data.

    The default function is inconsistent about the weight types it accepts on
    different platforms.

    # Arguments
        image [np array]: the image

    # Returns
        [tuple of np arrays]: the label of connected components and their
            corresponding positions
    """
    try:
        return measurements.find_objects(image, **kw)
    except:
        pass

    types = ["int32", "uint32", "int64", "unit64", "int16", "uint16"]
    for t in types:
        try:
            return measurements.find_objects(array(image, dtype=t), **kw)
        except:
            pass

    # let it raise the same exception as before
    return measurements.find_objects(image, **kw)


def crop_image(image, invert=False, binary8bit=True):
    """Crop the image such that the image does not have any padding

    Note that the image should be in binary form. The default option is the
    image has black text (0 pixel value) on a white background (255 pixel
    value)

    # Arguments
        image [np array]: the binary image to cut
        invert [bool]: if true, the image has black background, else white
            background. Default to False
        binary8bit [bool]: if true, then the image has range of 0-255, else
            it has range of 0-1. Default to True

    # Returns
        [np array]: the cut image
    """
    top, left = 0, 0
    bottom, right = image.shape
    coeff = 255 if binary8bit else 1

    # if the image is blank
    if len(np.unique(image)) == 1:
        return image

    if invert:
        while top < bottom:
            if np.sum(image[top]) > 0:
                break
            top += 1

        while bottom > top:
            if np.sum(image[bottom - 1]) > 0:
                break
            bottom -= 1

        while left < right:
            if np.sum(image[:, left]) > 0:
                break
            left += 1

        while right > left:
            if np.sum(image[:, right - 1]) > 0:
                break
            right -= 1

        return image[top:bottom, left:right]

    height, width = image.shape
    while top < bottom:
        if np.sum(image[top]) < width * coeff:
            break
        top += 1

    while bottom > top:
        if np.sum(image[bottom - 1]) < width * coeff:
            break
        bottom -= 1

    while left < right:
        if np.sum(image[:, left]) < height * coeff:
            break
        left += 1

    while right > left:
        if np.sum(image[:, right - 1]) < height * coeff:
            break
        right -= 1

    return image[top:bottom, left:right]


def trim_image_horizontally(image, invert=False, binary8bit=True):
    """Crop the image such that the image does not have any horizontal padding

    Note that the image should be in binary form. The default option is the
    image has black text (0 pixel value) on a white background (255 pixel
    value)

    # Arguments
        image [np array]: the binary image to cut
        invert [bool]: if true, the image has black background, else white
            background. Default to False
        binary8bit [bool]: if true, then the image has range of 0-255, else
            it has range of 0-1. Default to True

    # Returns
        [np array]: the cut image
    """
    top, left = 0, 0
    bottom, right = image.shape
    coeff = 255 if binary8bit else 1

    # if the image is blank
    if len(np.unique(image)) == 1:
        return image

    if invert:
        while left < right:
            if np.sum(image[:, left]) > 0:
                break
            left += 1

        while right > left:
            if np.sum(image[:, right - 1]) > 0:
                break
            right -= 1

        return image[top:bottom, left:right]

    height, width = image.shape
    while left < right:
        if np.sum(image[:, left]) < height * coeff:
            break
        left += 1

    while right > left:
        if np.sum(image[:, right - 1]) < height * coeff:
            break
        right -= 1

    return image[top:bottom, left:right]


def smoothen_image(image, sampling_level=4, is_binary=False):
    """Normalize character width to a predefined level

    # Arguments
        image [np array]: a binary image (1: background, 0: foreground)
        sampling_level [int]: the higher the sampling level, the smoother the
            dilation can be, default to 4, maximum 5 (higher value will cost
            more memory)

    # Returns
        [np array]: an image with width normalized to `pixel` level
    """
    max_pixel = 1 if is_binary else 255
    sampling_level = int(min(sampling_level, 5))
    sampling_level = int(max(sampling_level, 1))

    image = max_pixel - image

    # smoothen the edges
    for _ in range(sampling_level):
        image = cv2.pyrUp(image)

    for _ in range(4):
        image = cv2.medianBlur(image, 3)

    for _ in range(sampling_level):
        image = cv2.pyrDown(image)

    # revert the image
    image = max_pixel - image

    return image.astype(np.uint8)


def adjust_stroke_width(image, resize_ratio, is_binary=False):
    """Adjust the stroke width

    This function is usually used to adjust stroke width whenever the
    character image is resized.

    # Arguments
        image [np array]: the character image that stroke-width needs to be
            adjusted (note that the image should have black stroke on white
            background)
        resize_ratio [float]: the amount of resize that was applied to the
            image
        is_binary [bool]: whether the character image is the binary image

    # Returns:
        [np array]: the image with stroke-width adjusted
    """
    max_pixel = 1 if is_binary else 255
    image = (max_pixel - image).astype(np.uint8)

    if resize_ratio > 1.3:
        resize_ratio = int(math.ceil(resize_ratio))
        kernel = np.ones((resize_ratio, resize_ratio), dtype=np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
    elif resize_ratio < 0:
        resize_ratio = int(match.ceil(1 / resize_ratio))
        if resize_ratio >= 2:
            kernel = np.ones((resize_ratio, resize_ratio), dtype=np.uint8)
            image = cv2.dilate(image, kernel, iterations=1)
    else:
        return (max_pixel - image).astype(np.uint8)

    # After dilation or erotion, there will likely be blur, clean the blur
    # and just get the original foreground
    bin_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY |
                              cv2.THRESH_OTSU)[1]
    bin_image = (bin_image / 255).astype(np.uint8)
    image = image * bin_image
    image = (max_pixel - image).astype(np.uint8)
    image = smoothen_image(image, is_binary)

    return image


def unsharp_masking(image):
    """Deblur an image using unsharp masking with fixed values

    # Arguments
        image [np array]: the orginal image

    # Returns
        [np array]: the deblur images
    """
    blur = cv2.GaussianBlur(image, (9, 9), 10.0)
    unsharp_image = cv2.addWeighted(image, 1.5, blur, -0.5, 0)
    return unsharp_image


def _construct_compare_3d_array(image, compare_value=None):
    """Construct the array to support normalizing color

    This method works for grayscale image. It is used in the method
    `normalize_grayscale_color`. The idea is that we want to normalize
    the pixels in input image to have values in `compare_value`. For fast
    processing, we want to vectorize the process, and this method creates
    a compare object to help with the vectorization process in
    `normalize_grayscale_color`.

    # Arguments
        image [2D np array]: the grayscale image that we wish to compare

    # Returns
        [3D np array]: the numpy array we wish to compare to with size of
            (the amount of compare values, input image height, input image 
            width)
    """
    if compare_value is None:
        compare_value = np.array(
            [3, 31, 59,  87, 115, 143, 171, 199, 227, 255],
            dtype=np.int32)

    height, width = image.shape
    compare = np.ones((height, width, len(compare_value)), dtype=np.int32)
    compare = compare_value * compare
    compare = compare.transpose(2, 0, 1)
    return compare


def normalize_grayscale_color(image, compare_value=None):
    """Normalize the color of image character

    This method moves each pixel in `image` to the closest value in
    `compare_value`.

    # Arguments
        image [2D np array]: the grayscale image
        compare_value [1D np array]: the list of values

    # Returns
        [2D np array]: image of the same original shape, but with pixel values
            replaced with values in `compare_value`
    """
    height, width = image.shape
    if compare_value is None:
        compare_value = np.array(
            [3, 31, 59,  87, 115, 143, 171, 199, 227, 255],
            dtype=np.int32)

    # Construct the comparision object to support vectorization process
    compare = _construct_compare_3d_array(image)

    # Calculate the absolute difference of the image with the compare object
    image = image.astype(np.int32)      # to expand integer range
    abs_difference = np.abs(compare - image)

    # Get the ids of closest `compare_value`
    min_position = np.argmin(abs_difference, axis=0)

    # Reconstruct the image using the closet `compare_value`
    result = np.zeros(abs_difference.shape, dtype=np.uint8)
    result[
        min_position.flatten(),
        np.concatenate([np.ones((width,)) * _idx for _idx in range(height)])
        .astype(np.int32),
        list(range(width)) * height] = 1
    result = result * compare.astype(np.uint8)
    result = np.sum(result, axis=0)

    return result.astype(np.uint8)


def skeletonize_image(image, pixel, is_binary):
    """Normalize character width to a predefined level

    # Arguments
        image [np array]: a binary image (1: background, 0: foreground)
        pixel [int]: a pre-defined character width
        is_binary [bool]: whether the input image is a binary image

    # Returns
        [np array]: an image with width normalized to `pixel` level
    """
    if not is_binary:
        image = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        image = (image / 255).astype(np.uint8)

    # skeletonize the image
    image = 1 - image
    image = skeletonize(image)
    image = image.astype(np.uint8)

    # dilate the image to have `pixel` width
    if pixel > 1:
        kernel = np.ones((pixel, pixel), dtype=np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
    image = 1 - image

    # convert image to 8-bit to smoothen and binarize if possible
    image = image * 255
    image = smoothen_image(image, is_binary=False)

    if is_binary:
        image = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        image = (image / 255).astype(np.uint8)

    return image.astype(np.uint8)


def convert_from_binary_to_grayscale(image):
    """Convert a binary image into grayscale

    @NOTE: There is simple way using Gaussian method to transform binary image
    to grayscale image in LineOCRGenerator._construct_pencil_stroke.

    # Arguments
        image [np array]: the 8bit binary image to convert to grayscale. The
            image should have dark stroke on white background

    # Returns
        [np array]: the grayscale image
    """
    if len(np.unique(image)) > 2:
        raise AttributeError(
            "input should be binary image, instead receive input of {} values"
            .format(len(np.unique(image))))

    if np.max(image) > 1:
        image = (image / 255).astype(np.uint8)
        print(":WARNING: input should be binary of 0 and 1, instead receive"
              " image of {} values".format(np.unique(image)))

    # Invert the image to have foreground 1 and background 0
    invert_bin = 1 - image

    # Create the random kernel
    low_value = random.uniform(low=0.4, high=0.8)
    pixel_range = random.uniform(0.19, 1 - low_value)
    high_value = low_value + pixel_range

    kernel_scale = random.choice(3) + 1
    small_shape = (int(image.shape[0] / kernel_scale),
                   int(image.shape[1] / kernel_scale))
    large_shape = (image.shape[1], image.shape[0])  # width x height
    kernel = cv2.resize(
        random.uniform(low=low_value, high=high_value, size=small_shape),
        large_shape,
        interpolation=cv2.INTER_NEAREST).astype(np.float32)

    # Apply the kernel to image
    image = invert_bin * np.random.uniform(
        low=low_value, high=low_value + pixel_range, size=image.shape)
    image = (image * 255).astype(np.uint8)
    image = 255 - image
    image = cv2.blur(image, (2, 2))
    if random.random() < 0.3:
        image = cv2.blur(image, (2, 2))

    return image.astype(np.uint8)


def histogram_matching(source, template_image=None, template_hist=None):
    """Perform histogram matching (histogram specification)

    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image. Modified from:
    https://stackoverflow.com/questions/32655686
        /histogram-matching-of-two-images-in-python-2-x

    # Arguments
        source [ndarray]: the source image
        template_image [ndarray]: the image to match histogram template
        template_hist [ndarray]: the histogram template

    # Returns
        [ndarray]: the result image
    """
    if template_image is None and template_hist is None:
        raise ValueError('either `template_image` or `template_hist` should '
                         'be supplied')

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_counts, s_pixels = np.histogram(source.ravel(), 256, [0,256])
    if template_hist is not None:
        if template_hist.size != 256:
            raise ValueError('`template_hist` should be a 1D array of length '
                             '256, got {} instead'.format(template_hist.size))
        t_counts = template_hist
    else:
        t_counts, _ = np.histogram(template_image.ravel(), 256, [0,256])

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, s_pixels[:-1])

    return interp_t_values[source]
