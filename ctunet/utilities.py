# This file is part of the
#   ctunet Project (https://github.com/vfmatzkin/ctunet).
# Copyright (c) 2021, Franco Matzkin
# License: MIT
#   Full Text: https://github.com/vfmatzkin/ctunet/blob/main/LICENSE

""" Common functions used in the project."""

import configparser
import os
import timeit

import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
from ctunet.pytorch.transforms import fixed_pad
from raster_geometry import cylinder, cube
import monai.metrics as mon


def makedir(path=None):
    """Creates the folder in path if not exists.

    :param path: pred_folder to be created.
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
    dc = mon.compute_meandice(
        torch.movedim(torch.nn.functional.one_hot(torch.argmax(pred, 1)),
                      4, 1),
        target,
        include_background=False)
    return torch.mean(dc)


def hausdorff(result_b, reference_b):
    inf_alt = max(reference_b.shape)
    hd = mon.compute_hausdorff_distance(
        torch.movedim(
            torch.nn.functional.one_hot(torch.argmax(result_b, 1)),
            4, 1),
        reference_b)
    hdc = torch.nan_to_num(hd, nan=inf_alt, posinf=inf_alt, neginf=inf_alt)
    return torch.mean(hdc)


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


def fixed_pad_sitk(sitk_img, pad):
    arr = sitk.GetArrayFromImage(sitk_img)
    outimg = sitk.GetImageFromArray(
        fixed_pad(arr, pad))
    outimg.SetSpacing(sitk_img.GetSpacing())
    outimg.SetOrigin(sitk_img.GetOrigin())
    outimg.SetDirection(sitk_img.GetDirection())
    return outimg


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
    if not os.path.exists(cfg_file):
        raise FileNotFoundError("The provided cfg file does not exist "
                                f"({cfg_file}).")

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


