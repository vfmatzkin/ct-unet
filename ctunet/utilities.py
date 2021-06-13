import configparser
import csv
import os
import random
import string
import timeit
from datetime import datetime
from shutil import copyfile

import SimpleITK as sitk
import matplotlib.transforms

# plt.switch_backend('agg')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from medpy import metric
from raster_geometry import cylinder, cube
from scipy import stats as sps
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from torch.utils.data import DataLoader


# from dataset_classes import BrainSegmentationDataset


def save_image(inputPath, outputFolder, outImg):
    # Extract path and filename
    pathImg, filenameImg = os.path.split(inputPath)

    # Create out directory if not exists
    outputDir = os.path.join(pathImg, outputFolder)
    veri_folder(outputDir)

    savedImg = os.path.join(outputDir, filenameImg)
    # Write image
    sitk.WriteImage(outImg, savedImg)
    return savedImg


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


def addGaussianNoise(img):
    sigma = 0.1

    gauss = np.random.normal(scale=sigma, size=img.shape)
    img = img + gauss

    return img


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


def dice_coef_per_class_avg(output, masks):
    b_size = masks.size(0)
    num_classes = masks.size(1)
    probs = output
    mask = masks

    loss = []
    eps = 0.0000001
    num = probs.view(b_size, num_classes, -1) * mask.view(
        b_size, num_classes, -1
    )
    den1 = probs.view(b_size, num_classes, -1) * probs.view(
        b_size, num_classes, -1
    )
    den2 = mask.view(b_size, num_classes, -1) * mask.view(
        b_size, num_classes, -1
    )
    for c in range(num_classes):
        n = num[:, c].sum(1)
        d1 = den1[:, c].sum(1)
        d2 = den2[:, c].sum(1)
        loss.append(2 * torch.mean(((n + eps) / (d1 + d2 + eps))))
    return sum(loss) / len(loss)


class dice_loss_per_class_avg(nn.Module):
    def __init__(self):
        super(dice_loss_per_class_avg, self).__init__()

    def forward(self, output, masks):
        return 1 - dice_coef_per_class_avg(output, masks)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def dice_coef_avg(input, target):
    dc = dice_coef_per_class_avg(input, target)
    return sum(dc) / len(dc)


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


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        test: A tensor of shape [N, *]
        target: A tensor of shape same with test
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction="mean"):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert (
                predict.shape[0] == target.shape[0]
        ), "test & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = (
                torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1)
                + self.smooth
        )

        loss = num / den

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise Exception("Unexpected reduction {}".format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with test
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert (
                predict.shape == target.shape
        ), "test & target shape do not match"
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert (
                            self.weight.shape[0] == target.shape[1]
                    ), "Expect weight shape [{}], get[{}]".format(
                        target.shape[1], self.weight.shape[0]
                    )
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / target.shape[1]


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


def savePlots(train_loss, val_loss, name="", folder_path="sitk_img"):
    """
        Save in the current pred_folder (train_loss.png, test_loss.png) the train and test loss over time.
        Note that if you are running this code in a remote server (with no X server running), you sholud call:
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
    os.path.join(previous_folder, 'orig_dims')
        :param name: optional name.
        :param train_loss: Training loss array
        :param val_loss: Test loss array
    """

    trainName = name + "_train_loss.png"
    valName = name + "_val_loss.png"

    print("Saving plots... {}, {}\n".format(trainName, valName))

    # Check if output pred_folder exists
    veri_folder(folder_path)

    plt.figure()
    plt.plot(train_loss)
    plt.title("Train loss")
    plt.savefig(os.path.join(folder_path, trainName))
    plt.close()

    plt.figure()
    plt.plot(val_loss)
    plt.title("Validation loss")
    plt.savefig(os.path.join(folder_path, valName))
    plt.close()


def array_to_plot(
        losses, types, name="", plotTitle=None, x_label=None, y_label=None
):
    # name: nombre del model
    # types: que perdidas hay en losses (training, validation, etc)

    lossPlotName = name + "_{}.png".format(y_label.lower())

    print("Saving plot... {}\n".format(lossPlotName))

    # Check if output pred_folder exists
    veri_folder("img_path")

    bbox = matplotlib.transforms.Bbox([[-0.2, -0.36], [8, 5]])

    for i, arr in enumerate(losses):
        plt.plot(arr, label=types[i])

    if plotTitle is not None:
        plt.title(plotTitle)

    # Put titles to the axis
    if x_label is not None:
        plt.xlabel(x_label)

    if y_label is not None:
        plt.ylabel(y_label)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    plt.savefig(os.path.join("img_path", lossPlotName), dpi=200,
                bbox_inches=bbox)
    plt.close()


def randDeform(img, sigma=4, mask=None):
    """
        Randomly deforms a numpy 3D image and its mask (optional)

    :param img:
    :param sigma: scalar or sequence of scalars. Standard deviation for Gaussian kernel.
    The standard deviations of the Gaussian image_id are given for each axis as a sequence,
    or as a single number, in which case it is equal for all axes.

    :return:
    """
    shape = img.shape
    alpha = np.random.uniform(0, 1000)

    w = 2

    dx = (
            gaussian_filter(
                (np.random.rand(*shape)) * w - 1, sigma, mode="constant",
                cval=0
            )
            * alpha
    )
    dy = (
            gaussian_filter(
                (np.random.rand(*shape)) * w - 1, sigma, mode="constant",
                cval=0
            )
            * alpha
    )
    dz = (
            gaussian_filter(
                (np.random.rand(*shape)) * w - 1, sigma, mode="constant",
                cval=0
            )
            * alpha
    )

    x, y, z = np.mgrid[0: shape[0], 0: shape[1], 0: shape[2]]
    inds = (
        np.reshape(x + dx, (-1, 1)),
        np.reshape(y + dy, (-1, 1)),
        np.reshape(z + dz, (-1, 1)),
    )

    img = map_coordinates(img, inds, order=1, mode="constant").reshape(shape)
    if mask is not None:
        mask = map_coordinates(mask, inds, order=0, mode="constant").reshape(
            shape
        )
        return img, mask

    return img


def sitkRandDeform(img, mask=None, sigma=4):
    """
    Randomly deform a SimpleITK image and its mask

    :param img: SimpleITK image
    :param Mask: Mask (optional)
    :return: SimpleITK image consisting in the transformed image.
    """

    # Array sitk_img
    img_a = sitk.GetArrayFromImage(img)
    if mask is not None:
        mask_a = sitk.GetArrayFromImage(mask)

    # Deformed array
    if mask is not None:
        img_d, mask_d = randDeform(img_a, sigma, mask_a)
    else:
        img_d = randDeform(img_a, sigma)

    # Out image
    img_o = sitk.GetImageFromArray(img_d)
    img_o.SetSpacing(img.GetSpacing())
    img_o.SetOrigin(img.GetOrigin())
    img_o.SetDirection(img.GetDirection())

    if mask is not None:
        mask_o = sitk.GetImageFromArray(mask_d)
        mask_o.SetSpacing(mask.GetSpacing())
        mask_o.SetOrigin(mask.GetOrigin())
        mask_o.SetDirection(mask.GetDirection())
        return img_o, mask_o

    return img_o


def get_max_dims(img_folder, default_dims=None, ext=".nii.gz"):
    """Given a pred_folder with .nii.gz images, returns the biggest sizes in each dimension. Useful for adding padding in NN
        that require all the images to be the same size

    :param img_folder:
    :param default_dims:
    :return:
    """
    if default_dims:
        dims = default_dims
    else:
        dims = [0, 0, 0]

    for root, dirs, files in os.walk(img_folder):  # Files and subfold
        for i, name in enumerate(sorted(files, key=len)):
            if name.endswith(ext):
                filepath = os.path.join(root, name)  # Reconstruct file path.
                img = sitk.ReadImage(filepath)

                img_size = np.array(img.GetSize())
                dims[0] = max(dims[0], img_size[0])
                dims[1] = max(dims[1], img_size[1])
                dims[2] = max(dims[2], img_size[2])
    return dims


def create_csv(
        data_folder,
        csvname="UNetSP.csv",
        splits=None,
        image_identifier=None,
        mask_identifier=None,
        image_extension=".nii.gz",
        include_path=True,
):
    """Create csv file of images in data_folder. In that pred_folder,
    a "UNetSP.csv" will be created.

    :param data_folder: Input pred_folder
    """
    print("Generating CSV file. Saving as: ", csvname)

    splits = None if np.sum(splits) <= 0 or np.sum(splits) > 1 else splits

    filelist = [
        (
            (image_identifier if image_identifier else "image"),
            (mask_identifier if mask_identifier else "mask"),
        )
    ]

    for name in os.listdir(data_folder):
        f_ext = os.path.splitext(name)[1]
        if f_ext in image_extension:
            if mask_identifier and mask_identifier not in name:
                if image_identifier:
                    mask_name = name.replace(image_identifier, mask_identifier)
                    if include_path:
                        filepath = os.path.join(data_folder, name)
                        filepath_m = os.path.join(data_folder, mask_name)
                    else:
                        filepath = name
                        filepath_m = mask_name
                else:
                    mask_name = name.replace(f_ext,
                                             "_" + mask_identifier + f_ext)
                    if include_path:
                        filepath = os.path.join(data_folder, name)
                        filepath_m = os.path.join(data_folder, mask_name)
                    else:
                        filepath = name
                        filepath_m = mask_name
                filelist.append((filepath, filepath_m))
            elif mask_identifier and mask_identifier in name:
                continue
            elif not mask_identifier:
                filepath = os.path.join(data_folder,
                                        name) if include_path else name
            else:
                filepath = os.path.join(data_folder,
                                        name) if include_path else name

                name_m = name.replace(f_ext, "_" + mask_identifier + f_ext)
                filepath_m = os.path.join(data_folder,
                                          name_m) if include_path else name
            filelist.append((filepath, filepath_m))
        else:
            continue  # Not an image file

    if splits is not None:
        if len(splits) == 2:
            cut_idx = int(splits[0] * len(filelist))
            train_lst = filelist[
                        1:cut_idx
                        ]  # Split the list previously created
            test_lst = filelist[cut_idx:]
        elif len(splits) == 3:  # Train/validation/test splits
            cut_idx_1 = int(len(filelist) * splits[0])
            cut_idx_2 = int(len(filelist) * (splits[0] + splits[1]))
            train_lst = filelist[
                        1:cut_idx_1
                        ]  # Split the list previously created
            validation_lst = filelist[cut_idx_1:cut_idx_2]
            test_lst = filelist[cut_idx_2:]
        else:
            print("Wrong splits dimensions")
            return None

        train_lst.insert(
            0, (image_identifier, mask_identifier)
        )  # Insert headers.
        test_lst.insert(0, (image_identifier, mask_identifier))
        train_name = csvname.replace(".csv", "_train.csv")
        test_name = csvname.replace(".csv", "_test.csv")
        train_path = os.path.join(data_folder, train_name)
        test_path = os.path.join(data_folder, test_name)

        if len(splits) == 3:
            validation_lst.insert(0, (image_identifier, mask_identifier))
            validation_name = csvname.replace(".csv", "_validation.csv")
            validation_path = os.path.join(data_folder, validation_name)
            with open(validation_path, "w") as fp:  # Save validation CSV file
                writer = csv.writer(fp, delimiter=",")
                writer.writerows(validation_lst)
            splits_path = [train_path, validation_path, test_path]

        with open(train_path, "w") as fp:  # Save train CSV file
            writer = csv.writer(fp, delimiter=",")
            writer.writerows(train_lst)

        with open(test_path, "w") as fp:  # Save test CSV file
            writer = csv.writer(fp, delimiter=",")
            writer.writerows(test_lst)

        if len(splits) == 2:
            splits_path = [train_path, test_path]

    csv_path = os.path.join(data_folder, csvname)
    with open(csv_path, "w") as fp:  # Save all files CSV
        writer = csv.writer(fp, delimiter=",")
        writer.writerows(filelist)

    print("Saved: ", csv_path)
    if splits is not None:
        print("Saved: ", splits_path)
        return csv_path, splits_path

    return csv_path


def simple_csv(data_folder, csv_name="UNetSP.csv", image_identifier=None,
               mask_identifier=None, ext='.nii.gz'):
    filelist = [((image_identifier if image_identifier else "image"),
                 (mask_identifier if mask_identifier else "mask"),)]

    files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if
             f.endswith(ext)]
    if image_identifier:
        files = [f for f in files if image_identifier in f]

    for path in files:
        if mask_identifier:
            filelist.append((path, path.replace(image_identifier,
                                                mask_identifier)))
        else:
            filelist.append((path, ''))

    csv_path = os.path.join(data_folder, csv_name)
    with open(csv_path, "w") as fp:  # Save all files CSV
        writer = csv.writer(fp, delimiter=",")
        writer.writerows(filelist)

    print("Saved: ", csv_path)
    return csv_path


def folder_to_csv(
        data_folder,
        csvname="UNetSP.csv",
        image_extension=".nii.gz",
        include_path=True,
):
    """Create csv file of images in data_folder. In that pred_folder,
    a "UNetSP.csv" will be created.

    :param data_folder: Input pred_folder
    """
    print("Generating CSV file. Saving as: ", csvname)

    filelist = [("image", "mask")]  # CSV header

    for name in os.listdir(data_folder):
        f_ext = os.path.splitext(name)[1]
        if f_ext in image_extension:
            mask_name = name
            if include_path:  # attach file path
                filepath = os.path.join(
                    data_folder, name
                )  # Reconstruct file path.
                filepath_m = os.path.join(
                    data_folder, mask_name
                )  # Reconstruct file path.
            else:  # only file names
                filepath = name
                filepath_m = mask_name
            names = (filepath, filepath_m)
            filelist.append(names)
        else:
            continue  # Not an image file

    csv_path = os.path.join(data_folder, csvname)
    with open(csv_path, "w") as fp:  # Save all files CSV
        writer = csv.writer(fp, delimiter=",")
        writer.writerows(filelist)

    print("Saved: ", csv_path)
    return csv_path


def get_largest_cc(image):
    """
    Retains only the largest connected component of a binary image, and returns it.
    """
    image = sitk.Cast(image, sitk.sitkUInt32)

    connectedComponentFilter = sitk.ConnectedComponentImageFilter()
    objects = connectedComponentFilter.Execute(image)

    # If there is more than one connected component
    if connectedComponentFilter.GetObjectCount() > 1:
        objectsData = sitk.GetArrayFromImage(objects)

        # Detect the largest connected component
        maxLabel = 1
        maxLabelCount = 0
        for i in range(1, connectedComponentFilter.GetObjectCount() + 1):
            componentData = objectsData[objectsData == i]

            if len(componentData.flatten()) > maxLabelCount:
                maxLabel = i
                maxLabelCount = len(componentData.flatten())

        # Remove all the values, exept the ones for the largest connected component

        dataAux = np.zeros(objectsData.shape, dtype=np.uint8)

        # Fuse the labels

        dataAux[objectsData == maxLabel] = 1

        # Save edited image
        output = sitk.GetImageFromArray(dataAux)
        output.SetSpacing(image.GetSpacing())
        output.SetOrigin(image.GetOrigin())
        output.SetDirection(image.GetDirection())
    else:
        output = image

    return output


def boxplots(statsDict, savePath="sitk_img", name=None):
    """
    Generate boxplots of variable dictionary of stats. Example of use:
    statDict = {'dice' : {'unet' : np.random.normal(0,10,10),
                          'levelset' : np.random.normal(0,10,10)},
                'hausdorff' : {'unet' : np.random.normal(0,10,10),
                               'levelset' : np.random.normal(0,10,10)}}
    boxplotsv2(statDict, '')

    :param statsDict:
    :param savePath:
    :return:
    """
    for measure, stats in statsDict.items():
        dice, ax = plt.subplots()
        dice.canvas.draw()
        plt.boxplot(list(statsDict[measure].values()), showmeans=True)
        ax.set_xlabel("Method")
        plt.title(measure)
        ax.set_ylabel(measure)

        labels = list(statsDict[measure].keys())
        ax.set_xticklabels(labels)
        # plt.xticks(rotation=30, horizontalalignment='right')

        if name is None:
            svPath = os.path.join(savePath, measure + ".png")
        else:
            svPath = os.path.join(savePath, measure + name + ".png")
        plt.savefig(svPath)
        print("Output path: ", svPath)


def getStats(img1, img2, spacing):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    # print('Tipos: ', img1.dtype, img2.dtype)
    dice = metric.binary.dc(img1, img2)
    hausdorff = metric.binary.hd(
        img1.astype(np.float32), img2.astype(np.float32), spacing
    )
    asd = metric.binary.asd(
        img1.astype(np.float32), img2.astype(np.float32), spacing
    )

    return [dice, hausdorff, asd]


def loadStats(statsPath, methods):
    statDict = dict()

    # Initialize stats
    statDict["dice"] = dict()
    statDict["hausdorff"] = dict()
    statDict["asd"] = dict()

    for method in methods:
        statDict["dice"][method] = []
        statDict["hausdorff"][method] = []
        statDict["asd"][method] = []

    for i, path in enumerate(statsPath):
        with open(path, "r") as f:
            x = f.readlines()

        for item in x:
            item = item.replace("[", "").replace("]", "").split(",")
            statDict["dice"][methods[i]].append(float(item[0]))
            statDict["hausdorff"][methods[i]].append(float(item[1]))
            statDict["asd"][methods[i]].append(float(item[2]))

    return statDict


def hard_segm_from_tensor(probMap, keep_dims=False):
    """
    Given probMap in the form of a Theano tensor: ( shape = (N, numClasses, H, W, D), float32) or
    ( shape = (numClasses, H, W, D), float32), where the values indicates the probability for every class,
     it returns a hard segmentation per image sample where the value corresponds to the segmentation label with
     the highest probability.

    :param probMap: probability maps containing the per-class prediction/
    :param labels: List containing the real label value. If it is provided, then the labels in the output hard segmentation
           will contain the value provided in 'labels'. The mapping is lineal: the voxels whose probability is maximum in
           the c-th channel of the input probMap will be assigned label[c].
    :param keep_dims: Preserve dimensions after applying argmax

    :return: if a 5D tensor is provided (N, numClasses, H, W, D), it returns a 4D tensor with shape = (N, H, W, D).
             if a 4D tensor is provided, it returns a 3D tensor with shape = H, W, D
    """
    if len(list(probMap.shape)) == 5:
        probMap = (torch.argmax(probMap, dim=1)).type(torch.float)
        probMap = probMap.unsqueeze(1) if keep_dims else probMap
    else:
        probMap = (torch.argmax(probMap, dim=0)).type(torch.float)
        probMap = probMap.unsqueeze(0) if keep_dims else probMap
    return probMap


def salt_and_pepper(
        img, noise_probability=1, noise_density=0.2, salt_ratio=0.1
):
    batch_size = img.shape[0]
    output = np.copy(img).astype(np.uint8)
    noise_density = np.random.uniform(0, noise_density)
    for i in range(batch_size):
        r = random.uniform(0, 1)  # Random number
        if noise_probability >= r:  # Inside the probability
            blackDots = (
                    np.random.uniform(0, 1, output[i, :, :, :].shape)
                    > noise_density * (1 - salt_ratio)
            ).astype(np.uint8)
            whiteDots = 1 - (
                    np.random.uniform(0, 1, output[i, :, :, :].shape)
                    > noise_density * salt_ratio
            ).astype(np.uint8)
            output[i, :, :, :] = np.logical_and(output[i, :, :, :], blackDots)
            output[i, :, :, :] = np.logical_or(output[i, :, :, :], whiteDots)
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


def addNoiseToMasksBorder(img, noiseProb=0.4, depth=0.1):
    """
    Given a mask, add noise to its border, using distance transform.

    :param prob: probability of adding noise (dark pixels).
    :param depth: how many pixels can be switched
    :return: image with noise
    """
    batch_size = img.shape[0]
    output = np.zeros(img.shape)
    for i in range(batch_size):
        imgAux = sitk.GetImageFromArray(img[i, :, :, :])
        imgAux = sitk.Cast(imgAux, sitk.sitkUInt8)
        distMap = sitk.SignedMaurerDistanceMap(imgAux)

        saltRatio = 0.1
        imgSize = imgAux.GetSize()

        affectedArea = (
                distMap < -depth
        )  # Area inside the mask not covered by the noise

        outputAux = imgAux - affectedArea

        blackDots = (
                np.random.uniform(0, 1, size=imgSize) > noiseProb * (
                1 - saltRatio)
        ).astype(np.uint8)
        blackDots = sitk.GetImageFromArray(blackDots)
        blackDots.SetSpacing(outputAux.GetSpacing())
        blackDots.SetOrigin(outputAux.GetOrigin())
        blackDots.SetDirection(outputAux.GetDirection())

        outputAux = outputAux & blackDots
        outputAux = outputAux | affectedArea

        output[i, :, :, :] = sitk.GetArrayFromImage(outputAux)
    return output


def str2bool(v):
    return not v.lower() in ("false", "0", "no", "")


def calculateRemainingTime(ep_time, n_epoch, epochs):
    """
    Calculate training remaining time

    :param ep_time: Time per epoch (last)
    :param n_epoch: Number of epoch.
    :param epochs: Total of epochs.
    """
    ep_time = timeit.default_timer() - ep_time
    time1 = int(ep_time * (epochs + 1 - n_epoch))
    time_h = time1 // 3600
    time_m = (time1 - time_h * 3600) // 60
    print(
        "({}%) Remaining time (HH:MM): {}:{}\n".format(
            int(100 * n_epoch / float(epochs)), time_h, time_m
        )
    )


def autoName(model=None):
    tday = datetime.today()
    dateName = datetime.strftime(tday, "%y%m%d%H%M")

    if model is None:
        return dateName
    else:
        return dateName + "_" + model


def display_status(phase):
    if phase == "train":
        print("  Training...")
    elif phase == "test":
        print("  Testing...")
    elif phase == "val":
        print("  Validation...")
    elif phase == "test":
        print("  Predicting...")
    else:
        print(phase)


def write_stats(outFolder, statsArray):
    """Create/overwrite in out_folder a stats.txt file with with the content of statsArray

    :param outFolder: output pred_folder.
    :param statsArray: array with stats.
    :return:
    """
    statsfile = open(os.path.join(outFolder, "stats.txt"), "w")
    for item in statsArray:
        statsfile.write("%s\n" % item)
    statsfile.close()


def keep_max_pixel_value(outputnp, targetnp):
    """wrapper for numpy np.maximum.reduce([A,B,C])

    :param outputnp:
    :param targetnp:
    :return:
    """

    return np.maximum.reduce([outputnp, targetnp])


def get_stats_from_csv(csv_file, out_folder=None):
    if os.path.isfile(csv_file):
        stats_array = []
        dataset = BrainSegmentationDataset(csv_file=csv_file, root_dir="")
        loader = DataLoader(dataset=dataset, batch_size=1)

        for sample in loader:  # for each batch in the split
            p_mask = sample["image"]
            gt_mask = sample["segmentation"]
            filepath = sample["filepath"]

            s_img = sitk.ReadImage(filepath[0])
            spacing = s_img.GetSpacing()

            p_mask = sitk.GetImageFromArray(p_mask[0, :, :, :].numpy())
            gt_mask = sitk.GetImageFromArray(gt_mask[0, :, :, :].numpy())

            stats_array.append(
                getStats(
                    sitk.GetArrayFromImage(p_mask),
                    sitk.GetArrayFromImage(gt_mask),
                    spacing,
                )
            )

        if out_folder is None:
            out_folder, _ = os.path.split(csv_file)

        statsfile = open(os.path.join(out_folder, "stats.txt"), "w")
        for item in stats_array:
            statsfile.write("%s\n" % item)
        print("Stats file generated in ",
              os.path.join(out_folder, "stats.txt"))
    else:
        print("The csv file doesn't exists.")


def wilcoxon(stats_dict):
    """
    stats = {'dice' : {'unet' : np.random.normal(0,10,10),
                          'levelset' : np.random.normal(0,10,10)},
                'hausdorff' : {'unet' : np.random.normal(0,10,10),
                               'levelset' : np.random.normal(0,10,10)}}
    :param stats_dict:
    :return:
    """
    for measure, stats_dict in stats_dict.items():
        methods = stats_dict.keys()  # Method (U-Net, Level sets, etc)
        values = stats_dict.values()

        print("Measure: ", measure)

        wcx = np.empty((0, 1), dtype=float)

        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                # print(methods[i],len(values[i]), methods[j], len(values[j]))
                if len(values[i]) == len(values[j]):
                    wilcox = sps.wilcoxon(values[i], values[j])
                    print(
                        "{:45}".format(methods[i] + " vs " + methods[j]),
                        ": ",
                        wilcox[1],
                    )
                    wcx = np.append(wcx, wilcox[1])
                else:
                    print((i, j), " have different elems!.")

        print(
            "{:>45} : {} de {}".format(
                "Menores a 0.05", len(np.where(wcx < 0.05)[0]), len(wcx)
            )
        )


def padVolumeToMakeItMultipleOf(
        v, multipleOf=None, mode="minimum", return_padding=False
):
    if not multipleOf:
        multipleOf = [3, 3, 3]

    padding = (
        (
            0,
            0
            if v.shape[0] % multipleOf[0] == 0
            else multipleOf[0] - (v.shape[0] % multipleOf[0]),
        ),
        (
            0,
            0
            if v.shape[1] % multipleOf[1] == 0
            else multipleOf[1] - (v.shape[1] % multipleOf[1]),
        ),
        (
            0,
            0
            if v.shape[2] % multipleOf[2] == 0
            else multipleOf[2] - (v.shape[2] % multipleOf[2]),
        ),
    )

    if return_padding:
        return np.pad(v, padding, mode), padding
    else:
        return np.pad(v, padding, mode)


def unpad(x, pad_width):
    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return x[tuple(slices)]


def fixed_pad(
        v,
        final_img_size=None,
        mode="constant",
        constant_values=(0, 0),
        return_padding=False,
):
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


def diff_sitk(A, B):
    """Perform A-B with A and B SimpleITK images."""
    A = sitk.Cast(A, sitk.sitkInt8)
    B = sitk.Cast(B, sitk.sitkInt8)
    result = sitk.Cast(sitk.And(sitk.Not(A), B), sitk.sitkFloat32)
    result.SetOrigin(A.GetOrigin())
    result.SetDirection(A.GetDirection())
    result.SetSpacing(A.GetSpacing())
    return result


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


def print_params(config, sections=[]):
    print("--------\n Model parameters: ")
    for sec in sections:
        print("[{}]".format(sec))
        for item in config.items(sec):
            print("  {}: {}".format(item[0], item[1]))

    print("\n--------")
    return None


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


def str_to_num(text, max_n=11):
    """
    Get a number from a string, based on the characters numeric code and a custom maximum set number.

    :param text: Input string.
    :param max_n: (One plus) the maximum number allowed.
    :return: Output number, in the range [0, max_n).
    """
    s = 0
    for c in text:
        s += ord(c)
    return s % max_n


def crop_str_from_n_ocurrence(string, chr="_", n=2):
    """
    Get a substring finding the nth occurrence of chr and slicing up to that position.
    Example: "image_to_process_training.png" with chr='_' and n=2 -> "image_to"

    :param string: Input string.
    :param chr: Separator character to find.
    :param n: Number of separators in the output substring.
    :return: Substring
    """
    idxs = [pos for pos, char in enumerate(string) if char == chr]
    if len(idxs) == 0:  # char not found
        return string
    idx = n if n <= len(idxs) else len(idxs)
    return string[: idxs[idx - 1]]


def set_cfg_params(cfg_file=None, default_dict=None):
    """
    From a .ini ConfigParser file, create a dictionary with its image in the corresponding image types. Currently int,
    float, bool and string (default) types are supported as prefixes in the .ini files.
    The first two chars of each variable name will identify the file type (i_, f_, b_ and s_ are supported).

    :param cfg_file: Path of the cfg file
    :param default_dict: Dictionary with the necessary and minimum parameters initialized with None values or similar
    (useful in the case the configuration file does not provide all the required params).
    :return: A dictionary with the parameters set in the .ini file (and the default ones if not changed and provided).
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


def get_random_string(length):
    letters = string.ascii_letters
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def view(tensor):
    sitk.Show(sitk.GetImageFromArray(tensor.cpu().detach().numpy()))
