# This file is part of the
#   ctunet Project (https://github.com/vfmatzkin/ctunet).
# Copyright (c) 2021, Franco Matzkin
# License: MIT
#   Full Text: https://github.com/vfmatzkin/ctunet/blob/main/LICENSE

""" Common functions used in the project."""

import configparser
import os
import random
import timeit

import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
from raster_geometry import cylinder, cube


def veri_folder(path=None):
    """Verifies if the pred_folder exists, if not, it will be created.

    :param path: pred_folder to be ckecked/created.
    :return: path of the pred_folder
    """
    if not path:
        return None
    else:
        os.makedirs(path, exist_ok=True)
        return path


class dice_loss(nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()

    def forward(self, output, masks):
        b_size = masks.size(0)
        # num_classes = masks.size(1)
        probs = output
        mask = masks

        num = (probs.view(b_size, -1) * mask.view(b_size, -1)).sum(1)
        den1 = (probs.view(b_size, -1) * probs.view(b_size, -1)).sum(1)
        den2 = (mask.view(b_size, -1) * mask.view(b_size, -1)).sum(1)

        eps = 0.0000001
        return 1 - 2 * torch.mean(((num + eps) / (den1 + den2 + eps)))


def dice_coeff(pred, target):
    b_size = target.size(0)
    eps = 0.0000001

    dice_arr = []
    for i in range(b_size):
        probs = pred[i][0]
        mask = target[i][0]

        num = (probs.flatten() * mask.flatten()).sum()
        den1 = (probs.flatten() * probs.flatten()).sum()
        den2 = (mask.flatten() * mask.flatten()).sum()

        dice = 2 * (num + eps) / (den1 + den2 + eps)
        dice_arr.append(dice.item())

    return np.mean(dice_arr)


def one_hot_encoding(pttensor):
    """
    Given a hard segmentation (PyTorch tensor), it returns the 1-hot encoding.
    """
    batch_size = pttensor.shape[0]
    hard_segm = pttensor.cpu().numpy()
    labels = np.unique(hard_segm)
    dims = hard_segm.shape

    one_class = True if len(labels) == 1 else False
    len_lab = 2 if one_class else len(labels)

    one_hot = np.ndarray(
        shape=(batch_size, len_lab, dims[-3], dims[-2], dims[-1]),
        dtype=np.float32,
    )

    # Transform the Hard Segmentation GT to one-hot encoding
    for j, label_value in enumerate(labels):
        one_hot[:, j, :, :, :] = np.array(hard_segm == label_value).astype(
            np.int16
        )

    if one_class:
        one_hot[:, 1, :, :, :] = 0 * one_hot[:, 0, :, :, :]

    encoded = torch.from_numpy(one_hot)
    return encoded


def hard_segm_from_tensor(prob_map, keep_dims=False):
    """ Get hard segmentation from tensor.
    Given probMap in the form of a Theano tensor: ( shape = (N, numClasses, H,
    W, D), float32) or ( shape = (numClasses, H, W, D), float32), where the
    values indicates the probability for every class, it returns a hard
    segmentation per image sample where the value corresponds to the
    segmentation label with the highest probability.

    :param prob_map: probability maps containing the per-class prediction/
    :param keep_dims: Preserve dimensions after applying argmax

    :return: if a 5D tensor is provided (N, numClasses, H, W, D), it returns a
    4D tensor with shape = (N, H, W, D).
    If a 4D tensor is provided, it returns a 3D tensor with shape = H, W, D
    """
    if len(list(prob_map.shape)) == 5:
        prob_map = (torch.argmax(prob_map, dim=1)).type(torch.float)
        prob_map = prob_map.unsqueeze(1) if keep_dims else prob_map
    else:
        prob_map = (torch.argmax(prob_map, dim=0)).type(torch.float)
        prob_map = prob_map.unsqueeze(0) if keep_dims else prob_map
    return prob_map


def salt_and_pepper(img, noise_probability=1, noise_density=0.2,
                    salt_ratio=0.1):
    batch_size = img.shape[0]
    output = np.copy(img).astype(np.uint8)
    noise_density = np.random.uniform(0, noise_density)
    for i in range(batch_size):
        r = random.uniform(0, 1)  # Random number
        if noise_probability >= r:  # Inside the probability
            black_dots = (
                    np.random.uniform(0, 1, output[i, :, :, :].shape)
                    > noise_density * (1 - salt_ratio)
            ).astype(np.uint8)
            white_dots = 1 - (
                    np.random.uniform(0, 1, output[i, :, :, :].shape)
                    > noise_density * salt_ratio
            ).astype(np.uint8)
            output[i, :, :, :] = np.logical_and(output[i, :, :, :], black_dots)
            output[i, :, :, :] = np.logical_or(output[i, :, :, :], white_dots)
    return output


def skull_random_hole(img, prob=1, return_extracted=False):
    """ Simulate craniectomies placing random binary shapes.

    Given a batch of 3D images (PyTorch tensors), crop a random cube or box
    placed in a random position of the image with the sizes given in d.

    :param img: Input image.
    :param prob: probability of adding the noise (by default flip a coin).
    :param return_extracted: Return extracted bone flap.
    """
    is_tensor = True if type(img) == torch.Tensor else False
    if is_tensor:  # Batch of PyTorch tensors
        batch_size = img.shape[0]
        output = np.copy(img).astype(np.uint8)
        if return_extracted:
            flap = np.copy(img).astype(np.uint8)
        for i in range(batch_size):
            np_img = output[i, :, :, :]
            if not return_extracted:
                output[i] = random_blank_patch(np_img, prob, return_extracted)
            else:
                output[i], flap[i] = random_blank_patch(np_img, prob,
                                                        return_extracted)
        output = output if not is_tensor else torch.tensor(output,
                                                           dtype=torch.int8)
        if not return_extracted:
            return output
        else:
            flap = flap if not is_tensor else torch.tensor(flap,
                                                           dtype=torch.int8)
            return output, flap
    else:  # Single image of Preprocessor class
        np_img = sitk.GetArrayFromImage(img)
        o_spacing = img.GetSpacing()
        o_direction = img.GetDirection()
        o_origin = img.GetOrigin()

        np_img = np_img.astype(np.uint8)
        c_img = random_blank_patch(np_img, prob)

        img_o = sitk.GetImageFromArray(c_img)
        img_o.SetSpacing(o_spacing)
        img_o.SetOrigin(o_origin)
        img_o.SetDirection(o_direction)
        return img_o


def shape_3d(center, size, image_size, shape="flap"):
    """ Generate a

    Return a 3D numpy sphere or cube given the center, size and image size. It creates a mask of the shape usingthe p-norm concept.

    :param center: array with the coordinates center of the shape.
    :param size: single number that represents the size of the shape in each dimension.
    :param image_size: size of the output array.
    :param shape: shape to return. Currently 'circle' and 'cube' are allowed.
    :return:
    """
    if type(image_size) == sitk.Image:
        image_size = sitk.GetArrayFromImage(image_size).shape

    if shape in ["circle", "sphere"]:
        ord = 2
    elif shape in ["square", "box", "cube"]:
        ord = np.inf
    elif shape in ["flap", "autoimplant"]:
        c_diam = (
                np.random.uniform(0.25, 1) * size / 4
        )
        center_relative = tuple(l / r for l, r in zip(center, image_size))
        z_edge_1 = (
            (center[0]) / image_size[0],
            (center[1] - size / 2) / image_size[1],
            (center[2] - size / 2) / image_size[2],
        )
        z_edge_2 = (
            (center[0]) / image_size[0],
            (center[1] - size / 2) / image_size[1],
            (center[2] + size / 2) / image_size[2],
        )

        cyl1 = cylinder(image_size, size, c_diam, 0, z_edge_1).astype(np.uint8)
        cyl2 = cylinder(image_size, size, c_diam, 0, z_edge_2).astype(np.uint8)
        cub1 = cube(image_size, size, center_relative).astype(np.uint8)

        mask = np.logical_or(cyl1, np.logical_or(cyl2, cub1)).astype(np.uint8)
        return 1 - mask
    else:
        print(
            "Shape {} is not supported. Setting shape as sphere".format(shape)
        )
        ord = 2
    distance = np.linalg.norm(
        np.subtract(np.indices(image_size).T, np.asarray(center)),
        axis=len(center),
        ord=ord,
    )
    shape_np = 1 - np.ones(image_size).T * (distance <= size)
    return shape_np.T


def get_img_center(img):
    np_img = sitk.GetArrayFromImage(img) if type(img) == sitk.Image else img
    return tuple([int(s / 2) for s in np_img.shape])


def random_blank_patch(image, prob=1, return_extracted=False, p_type="random"):
    r = random.uniform(0, 1)  # Random number
    if prob >= r:  # Inside the probability -> crop
        image_size = image.shape

        while True:
            # Define center of the mask
            center = np.array(
                [np.random.randint(0, dim) for dim in image.shape]
            )  # random point
            plane_cond = (
                    center[1] * (3 / 7 * image_size[0] / image_size[1]) +
                    center[0]
                    > 0.65 * image_size[0]
            )  # Plane
            if image[tuple(center)] and plane_cond:  # white pixel
                break

        # Define radius
        min_radius = (np.min(image_size) // 5) - 1
        max_radius = np.max([min_radius, np.max(image_size) // 3.5])
        size = np.random.randint(min_radius, max_radius)

        valid_shapes = ["sphere", "box", "flap"]
        p_type = (
            valid_shapes[np.random.randint(0, len(valid_shapes))]
            if p_type not in valid_shapes
            else p_type
        )
        if p_type == "sphere":
            shape_np = shape_3d(center, size, image_size, shape="sphere")
        elif p_type == "box":
            shape_np = shape_3d(center, size, image_size, shape="box")
        else:
            shape_np = shape_3d(center, size, image_size, shape="flap")

        # Mask the image
        masked_out = np.logical_and(image, shape_np).astype(
            np.uint8
        )  # Apply the mask

        if not return_extracted:
            return masked_out
        else:
            extracted = np.logical_and(image, 1 - shape_np).astype(np.uint8)
            return masked_out, extracted
    else:
        if not return_extracted:
            return image
        else:
            return image, np.zeros_like(image)


def unpad(x, pad_width):
    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return x[tuple(slices)]


def fixed_pad(v, final_img_size=None, mode="constant", constant_values=(0, 0),
              return_padding=False, ):
    if final_img_size is None:
        print("Desired image size not provided!")
        return None

    for i in range(0, len(final_img_size)):
        if v.shape[i] > final_img_size[i]:
            print("The input size is bigger than the output size!")
            print(v.shape, " vs ", final_img_size)
            return None

    padding = (
        (0, final_img_size[0] - v.shape[0]),
        (0, final_img_size[1] - v.shape[1]),
        (0, final_img_size[2] - v.shape[2]),
    )

    if not return_padding:
        return np.pad(v, padding, mode, constant_values=constant_values)
    else:
        return (
            np.pad(v, padding, mode, constant_values=constant_values),
            padding,
        )


def fixed_pad_sitk(sitk_img, pad):
    arr = sitk.GetArrayFromImage(sitk_img)
    outimg = sitk.GetImageFromArray(
        fixed_pad(arr, pad))
    outimg.SetSpacing(sitk_img.GetSpacing())
    outimg.SetOrigin(sitk_img.GetOrigin())
    outimg.SetDirection(sitk_img.GetDirection())
    return outimg


def random_flip(img, probability=0.5, axis=None):
    batch_size = img.shape[0]
    for i in range(batch_size):
        r = random.uniform(0, 1)  # Random number
        if probability >= r:  # Inside the probability
            if axis is None:
                ax = random.randint(1, 3)
            else:
                ax = axis
            if ax == 1:
                img[i, :, :, :] = torch.flip(img[i, :, :, :], dims=[0])
            if ax == 2:
                img[i, :, :, :] = torch.flip(img[i, :, :, :], dims=[1])
            if ax == 3:
                img[i, :, :, :] = torch.flip(img[i, :, :, :], dims=[2])
    return img


def np_to_sitk(np_img, origin, direction, spacing):
    sitk_out = sitk.GetImageFromArray(np_img)
    sitk_out.SetOrigin(origin)
    sitk_out.SetDirection(direction)
    sitk_out.SetSpacing(spacing)

    return sitk_out


def get_sitk_img(np_img, origin, direction, spacing):
    if len(np_img.shape) > 3:  # More than one image
        sitk_out_list = []
        for img in np_img:
            sitk_out_list.append(np_to_sitk(img, origin, direction, spacing))
        return sitk_out_list
    else:
        return np_to_sitk(np_img, origin, direction, spacing)


def set_cfg_params(cfg_file=None, default_dict=None):
    """
    From a .ini ConfigParser file, create a dictionary with its image in the
    corresponding image types. Currently int, float, bool and string (default)
    types are supported as prefixes in the .ini files.
    The first two chars of each variable name will identify the file type (i_,
    f_, b_ and s_ are supported).

    :param cfg_file: Path of the cfg file
    :param default_dict: Dictionary with the necessary and minimum parameters
    initialized with None values or similar
    (useful in the case the configuration file does not provide all the
    required params).
    :return: A dictionary with the parameters set in the .ini file (and the
    default ones if not changed and provided).
    """
    if cfg_file is None:
        return

    out_dict = (
        default_dict if default_dict is not None else dict()
    )  # Initialize the dictionary
    config = configparser.ConfigParser()
    config.read(cfg_file)

    for each_section in config.sections():
        for (key, value) in config.items(each_section):
            if key[:2] == "i_":  # int
                out_dict[key[2:]] = config[each_section].getint(key)
            elif key[:2] == "f_":  # float
                out_dict[key[2:]] = config[each_section].getfloat(key)
            elif key[:2] == "b_":  # bool
                out_dict[key[2:]] = config[each_section].getboolean(key)
            elif key[:2] == "s_":  # string
                out_dict[key[2:]] = value
            else:  # string by default
                out_dict[key] = value

    return out_dict


def print_params_dict(dic):
    """
    Given a dicitonary, print its values in a table-like format.

    :param dic: Dictionary to print
    """
    print("{:<20} {:<30}".format("Parameter", "Value"))
    for key in dic:
        v = dic[key]
        print("{:<15} {:<10}".format(key, str(v)))


def tic():
    """ Start measuring time.

    The returned value should be stored in a variable and passed to toc_eps
    on each epoch.

    :return: default timer.
    """
    return timeit.default_timer()


def toc_eps(ep_time, n_epoch, epochs, print_out=True):
    """
    Stop measuring time and estimate remaining time.

    Calculate training remaining time given a previous time, the current epoch
    and the total epochs.

    :param ep_time: Time per epoch (last)
    :param n_epoch: Number of epoch.
    :param epochs: Total of epochs.
    :param print_out: print the result
    """
    ep_time = timeit.default_timer() - ep_time
    time1 = int(ep_time * (epochs + 1 - n_epoch))
    time_h = time1 // 3600
    time_m = (time1 - time_h * 3600) // 60
    if print_out:
        print(
            "({}%) Remaining time (HH:MM): {}:{}\n".format(
                int(100 * n_epoch / float(epochs)), time_h, time_m
            )
        )
    return ep_time


def get_sitk_metadata(sitk_img):
    origin = sitk_img.GetOrigin()
    direction = sitk_img.GetDirection()
    spacing = sitk_img.GetSpacing()
    return origin, direction, spacing


def view(tensor):
    sitk.Show(sitk.GetImageFromArray(tensor.cpu().detach().numpy()))


def erode(sitk_img, times=1):
    """ Given a SimpleITK image, erode it according to the times parameter

    :param sitk_img: SimpleITK binary image.
    :param times: Number of erosions performed.
    :return:
    """
    for _ in range(times):
        sitk_img = sitk.ErodeObjectMorphology(sitk_img)
    return sitk_img


def dilate(sitk_img, times=1):
    """ Given a SimpleITK image, dilate it according to the times parameter

    :param sitk_img: SimpleITK binary image.
    :param times: Number of dilations performed.
    :return:
    """
    for _ in range(times):
        sitk_img = sitk.DilateObjectMorphology(sitk_img)
    return sitk_img


def erode_dilate(inp_img, p=1, min_it=0, max_it=1):
    """ Apply an image erosion/dilation with a probability of application p.

    :param inp_img: Input image. It could be a torch.Tensor, np.ndarray or
    sitk.Image.
    :param p: Probability for applying the transform.
    :param min_it: Minimum number of iterations.
    :param max_it: Minimum number of iterations.
    :return: Eroded or dilated image in the same image type.
    """
    if np.random.rand() > p:  # do nothing
        return inp_img

    is_tensor = is_array = False
    if type(inp_img) == torch.Tensor:
        is_tensor = True
        img = inp_img.numpy()
        img = sitk.GetImageFromArray(img)
    elif type(inp_img) == np.ndarray:
        is_array = True
        img = sitk.GetImageFromArray(inp_img)

    times = np.random.randint(min_it, max_it)
    out_img = np.random.choice([erode, dilate])(img, times)

    if is_array:
        return sitk.GetArrayFromImage(out_img)
    elif is_tensor:
        return torch.tensor(sitk.GetArrayFromImage(out_img))
    else:
        return out_img