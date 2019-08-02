import numpy as np

class TrainingHelper(object):

    def initialize_helper(self, **kwargs):
        pass

    def postprocess_label(self, label, **kwargs):
        return label

    def postprocess_image(self, image, **kwargs):
        return image

    def postprocess_outputs(self, images, widths, labels, **kwargs):
        return images, widths, labels


class FFG_Helper(TrainingHelper):
    """
    Help with making out data clean for generation
    """
    def __init__(self):
        """Initialize the helper"""

        self.label_2_char = {}
        self.char_2_label = {}

    def initialize_helper(self, allowed_chars, **kwargs):
        """Update the label_2_char and char_2_label dictionaries"""

        letter_list = list(allowed_chars)
        letter_list.sort()
        for _idx, each_character in enumerate(letter_list):
            self.label_2_char[_idx+1] = each_character
            self.char_2_label[each_character] = _idx+1

        self.label_2_char[0] = '_pad_'
        self.char_2_label['_pad_'] = 0

    def is_char_exists(self, char):
        """Check if the character exists in database

        # Arugments
            char [str]: the character to check

        # Returns
            [bool]: True if the character exists, False otherwise
        """
        return char in self.char_2_label

    # def postprocess_label(self, label, **kwargs):
    #     """Perform post-processing on the label

    #     # Arguments
    #         text [str]: the text to post-process

    #     # Returns
    #         [str]: the post-processed text
    #     """
    #     if kwargs.get('label_converted_to_list', False):
    #         label_ = []
    #         for each_char in label:
    #             label_.append(self.char_2_label[each_char])
    #         label = label_

    #     return label

    def postprocess_image(self, image, **kwargs):
        """Perform post-processing on the image

        # Arguments
            image [np array]: the image to postprocess

        # Returns
            image [np array]: the result image
        """
        if kwargs.get('get_6_channels', False):
            channel_first = kwargs.get('channel_first', False)
            kwargs['append_channel'] = False
            image = self._get_6_features(1 - image, channel_first)

        if kwargs.get('append_channel', False):
            axis = 0 if kwargs.get('channel_first', False) else -1
            image = np.expand_dims(image, axis=axis)

        return image

    def postprocess_outputs(self, images, widths, labels, **kwargs):
        return np.asarray(images, dtype=np.uint8), widths, labels

    def get_number_of_classes(self):
        """Get the number of classes

        # Returns
            [int]: number of classes
        """
        return len(self.char_2_label)

    def _get_4_features(self, image, channel_first):
        """Create a 4-channel feature image

        The features include: binary image, canny edge image, gradient
        in y and gradient in x.

        # Arguments
            image [np array]: the image, should be binary
            channel_first [bool]: whether the image is CxHxW or HxWxC

        # Returns
            [np array]: the 4-channel image
        """
        edge = (cv2.Canny(image * 255, 50, 150) / 255).astype(np.uint8)
        dx = cv2.Scharr(image, ddepth=-1, dx=1, dy=0) / 16
        dy = cv2.Scharr(image, ddepth=-1, dx=0, dy=1) / 16

        axis = 0 if channel_first else -1
        image_4_features = np.stack([image, edge, dx, dy], axis=axis)

        return image_4_features

    def _get_6_features(self, image, channel_first):
        """Create a 6-channel feature image

        The features include: binary image, canny edge image, gradient
        in y, gradient in x, crop 4 and crop 8.

        # Arguments
            image [np array]: the image, should be binary
            channel_first [bool]: whether the image is CxHxW or HxWxC

        # Returns
            [np array]: the 6 channel image
        """
        height, width = image.shape

        image_4_features = self._get_4_features(image, channel_first)
        img1 = cv2.resize(image[4:-4,:], (width, height),
                          interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(image[8:-8,:], (width, height),
                          interpolation=cv2.INTER_LINEAR)

        axis = 0 if channel_first else -1
        img1 = np.expand_dims(img1, axis=-1)
        img2 = np.expand_dims(img2, axis=-1)

        return np.concatenate([image_4_features, img1, img2], axis=axis)
