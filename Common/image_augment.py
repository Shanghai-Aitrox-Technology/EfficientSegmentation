import random
import numpy as np
import scipy.ndimage as nd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.filters import gaussian_filter


class DataAugmentor(object):
    """
    Augment 3D image in npy format with z,y,x order.
    """

    def __init__(self):
        super(DataAugmentor, self).__init__()
        self.dims = 3
        self.axes_3d = [(0, 1), (0, 2), (1, 2)]
        self.labels = [1]

    def random_flip(self, npy_image, npy_label=None):
        """
        Random flip image with mask in each axis.
        """
        degree = self.dims - 1
        while degree >= 0:
            if np.random.choice([0, 1]):
                npy_image = np.flip(npy_image, axis=degree)
                if npy_label is not None:
                    npy_label = np.flip(npy_label, axis=degree)
            degree -= 1

        if npy_label is not None:
            return npy_image, npy_label

        return npy_image

    def random_rotate(self, npy_image, npy_label=None, min_angle=-5, max_angle=5, axes=None):
        """
        Random rotate image in one axes, which is selected randomly or passed in, as well as mask.
        """
        if np.random.choice([0, 1]):
            theta = np.random.uniform(min_angle, max_angle)
            if axes is None:
                axes_random_id = np.random.randint(low=0, high=len(self.axes_3d))
                axes = self.axes_3d[axes_random_id]

            npy_image = nd.rotate(npy_image, theta, axes=axes)
            if npy_label is not None:
                npy_label = nd.rotate(npy_label, theta, axes=axes)

        if npy_label is not None:
            return npy_image, npy_label

        return npy_image

    def random_swap(self, npy_image, npy_label=None, axis1=None, axis2=None):
        """
        Random swap image in a pair of axis, which is selected randomly or passed in, as well as mask.
        """
        if np.random.choice([0, 1]):
            if (axis1 is None) or (axis2 is None):
                axis1 = np.random.randint(0, self.dims)
                axis2 = np.random.randint(0, self.dims)
                while axis2 == axis1:
                    axis2 = np.random.randint(0, self.dims)

            npy_image = np.swapaxes(npy_image, axis1, axis2)
            if npy_label is not None:
                npy_label = np.swapaxes(npy_label, axis1, axis2)

        if npy_label is not None:
            return npy_image, npy_label

        return npy_image

    @staticmethod
    def random_zoom(npy_image, npy_label=None, min_percentage=0.8, max_percentage=1.2):
        zoom_factor = np.random.random() * (max_percentage - min_percentage) + min_percentage
        zoom_matrix = np.array([[zoom_factor, 0, 0, 0],
                                [0, zoom_factor, 0, 0],
                                [0, 0, zoom_factor, 0],
                                [0, 0, 0, 1]])

        if npy_label is not None:
            return nd.affine_transform(npy_image, zoom_matrix), nd.affine_transform(npy_label, zoom_matrix)

        return nd.affine_transform(npy_image, zoom_matrix)

    @staticmethod
    def random_shift(npy_image, npy_mask=None, max_percentage=0.2):
        [image_depth, image_height, image_width] = npy_image.shape

        distance_z = int(image_depth * max_percentage / 2)
        distance_y = int(image_height * max_percentage / 2)
        distance_x = int(image_width * max_percentage / 2)

        shift_z = np.random.randint(-distance_z, distance_z)
        shift_y = np.random.randint(-distance_y, distance_y)
        shift_x = np.random.randint(-distance_x, distance_x)

        offset_matrix = np.array([[1, 0, 0, shift_z],
                                  [0, 1, 0, shift_y],
                                  [0, 0, 1, shift_x],
                                  [0, 0, 0, 1]])

        if npy_mask is not None:
            return nd.affine_transform(npy_image, offset_matrix), nd.affine_transform(npy_mask, offset_matrix)

        return nd.affine_transform(npy_image, offset_matrix)

    @staticmethod
    def elastic_transform_3d(npy_image, npy_label=None, alpha=1, sigma=20, bg_val=0.0, method="linear"):
        """
        Elastic deformation of images as described in [Simard2003]_.
        [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
        Modified from:
        https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
        Modified to take 3 and 4 dimensional inputs
        Deforms both the image and corresponding label file
        image tri-linear interpolated
        Label volumes nearest neighbour interpolated
        """
        shape = npy_image.shape

        # Define coordinate system
        coords = np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2])

        # Initialize interpolators
        im_intrps = RegularGridInterpolator(coords, npy_image, method=method, bounds_error=False, fill_value=bg_val)

        # Get random elastic deformations
        dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                             mode="constant", cval=0.) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                             mode="constant", cval=0.) * alpha
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                             mode="constant", cval=0.) * alpha

        # Define sample points
        z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
        indices = np.reshape(z + dz, (-1, 1)), \
                  np.reshape(y + dy, (-1, 1)), \
                  np.reshape(x + dx, (-1, 1))

        # Interpolate 3D image
        npy_image = im_intrps(indices).reshape(shape)

        # Interpolate labels
        if npy_label is not None:
            lab_intrp = RegularGridInterpolator(coords, npy_label, method="nearest", bounds_error=False, fill_value=0)
            npy_label = lab_intrp(indices).reshape(shape).astype(npy_label.dtype)
            return npy_image, npy_label

        return npy_image

    @staticmethod
    def random_crop_to_labels(npy_image, npy_label):
        """
        Random center crop image according to mask.
        :return:
        """
        [image_depth, image_height, image_width] = npy_image.shape

        indices = np.where(npy_label > 0)
        [max_d, max_h, max_w] = np.max(np.array(indices), axis=1)
        [min_d, min_h, min_w] = np.min(np.array(indices), axis=1)

        z_min = int(min_d * np.random.random())
        y_min = int(min_h * np.random.random())
        x_min = int(min_w * np.random.random())

        z_max = int(image_depth - (image_depth - max_d) * np.random.random())
        y_max = int(image_height - (image_height - max_h) * np.random.random())
        x_max = int(image_width - (image_width - max_w) * np.random.random())

        z_min = int(np.max([0, z_min]))
        y_min = int(np.max([0, y_min]))
        x_min = int(np.max([0, x_min]))

        z_max = int(np.min([image_depth, z_max]))
        y_max = int(np.min([image_height, y_max]))
        x_max = int(np.min([image_width, x_max]))

        return npy_image[z_min: z_max, y_min: y_max, x_min: x_max], npy_label[z_min: z_max, y_min: y_max, x_min: x_max]

    @staticmethod
    def random_crop_to_extend_labels(npy_image, npy_label, max_extend=(20, 20, 20)):
        """
        Random center crop image according to mask.
        :return:
        """
        [image_depth, image_height, image_width] = npy_image.shape

        indices = np.where(npy_label > 0)
        [max_d, max_h, max_w] = np.max(np.array(indices), axis=1)
        [min_d, min_h, min_w] = np.min(np.array(indices), axis=1)

        extend_z_start = np.random.randint(-max_extend[0]//3, max_extend[0])
        extend_y_start = np.random.randint(-max_extend[1]//3, max_extend[1])
        extend_x_start = np.random.randint(-max_extend[2]//3, max_extend[2])
        extend_z_end = np.random.randint(-max_extend[0]//3, max_extend[0])
        extend_y_end = np.random.randint(-max_extend[1]//3, max_extend[1])
        extend_x_end = np.random.randint(-max_extend[2]//3, max_extend[2])

        z_min = int(min_d + extend_z_start)
        y_min = int(min_h + extend_y_start)
        x_min = int(min_w + extend_x_start)

        z_max = int(max_d + extend_z_end)
        y_max = int(max_h + extend_y_end)
        x_max = int(max_w + extend_x_end)

        z_min = int(np.max([0, z_min]))
        y_min = int(np.max([0, y_min]))
        x_min = int(np.max([0, x_min]))

        z_max = int(np.min([image_depth, z_max]))
        y_max = int(np.min([image_height, y_max]))
        x_max = int(np.min([image_width, x_max]))

        return npy_image[z_min: z_max, y_min: y_max, x_min: x_max], npy_label[z_min: z_max, y_min: y_max, x_min: x_max]

    @staticmethod
    def augment_brightness_additive(npy_image, npy_label, labels=None, additive_range=(-200, 200)):
        if labels is None:
            labels = [1]

        for i in labels:
            if np.random.choice([0, 1]):
                gray_value = np.random.randint(additive_range[0], additive_range[1], 1)
                npy_image[npy_label == i] += gray_value[0]

        return npy_image

    @staticmethod
    def augment_brightness_multiplicative(npy_image, npy_label, labels=None, multiplier_range=(0.75, 1.25)):
        if labels is None:
            labels = [1]

        for i in labels:
            if np.random.choice([0, 1]):
                multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
                npy_image[npy_label == i] *= multiplier
        return npy_image

    @staticmethod
    def augment_gaussian_noise(npy_image, noise_variance=(0, 0.1)):
        if noise_variance[0] == noise_variance[1]:
            variance = noise_variance[0]
        else:
            variance = random.uniform(noise_variance[0], noise_variance[1])
        npy_image += np.random.normal(0.0, variance, size=npy_image.shape)

        return npy_image
