# This file is part of the
#   ctunet Project (https://github.com/vfmatzkin/ctunet).
# Copyright (c) 2021, Franco Matzkin
# License: MIT
#   Full Text: https://github.com/vfmatzkin/ctunet/blob/main/LICENSE

""" ProblemHandler class and some usages.

A ProblemHandler simply defines what datasets to use and how to write a
model prediction. This way, how to read the data and prepare the train/test
samples keeps defined in Dataset subclasses. """

from abc import ABC, abstractmethod

import torch.nn as nn
from torch.nn.functional import softmax

from .dataset_classes import *


class ProblemHandler:
    """ This class defines the employed datasets and how to write the output.

    This is an abstract class so in order to use it, you must create a
    subclass and implement the write_predictions method. Note that its
    arguments correspond to a batch of pytorch images, so they must be
    saved one by one.

    :param train_dataset_class: Dataset used while training.
    :param test_dataset_class: Dataset used while testing.
    """

    def __init__(self, train_dataset_class, test_dataset_class):
        self.train_dataset_class = None if train_dataset_class is None else \
            train_dataset_class
        self.test_dataset_class = None if test_dataset_class is None else \
            test_dataset_class

    @abstractmethod
    def write_predictions(self, predictions, input_filepaths,
                          output_folder_name, input_imgs):
        pass

    @staticmethod
    def comp_losses_metrics(model, prediction, target, idx, n_imgs):
        """ Compute the losses and required metrics.

        Calculate the losses depending on the model phase, and save it in
        self.pt_loss. It contains two basic losses and one metric.

        :param model: Model instance. It contains all the training related
        information.
        :param prediction: Model prediction.
        :param target: Target image.
        :param idx: Batch index
        :param n_imgs: Number of imgs in the batch
        :return:
        """
        model.pt_loss = []  # List that will contain all the weighted losses.

        # /!\ Loss functions /!\
        # Binary cross entropy between model output and target
        if model.params["ce_lambda"] != 0:
            if "ce" not in model.losses_and_metrics:
                model.losses_and_metrics["ce"] = []  # For visualization

            target_am = torch.argmax(target, 1) if (len(target.shape)) == 5 \
                else target  # If it's encoded, decode it.
            ce_loss = nn.CrossEntropyLoss()(prediction, target_am)
            model.pt_loss.append(model.params["ce_lambda"] * ce_loss)
            model.losses_and_metrics["ce"].append(float(model.pt_loss[-1]))

        if model.params["dice_lambda"] != 0:  # Dice loss
            if "dice_loss" not in model.losses_and_metrics:
                model.losses_and_metrics["dice_loss"] = []

            dice_loss = utils.dice_loss()(prediction, target)
            model.pt_loss.append(model.params["dice_lambda"] * dice_loss)
            model.losses_and_metrics["dice_loss"].append(
                float(model.pt_loss[-1])
            )

        # /!\ Metrics /!\
        if model.params["save_dice_plots"] is True:  # Dice coefficient
            if "dice_coef" not in model.losses_and_metrics:
                model.losses_and_metrics["dice_coef"] = []
            dice_coeff = utils.dice_coeff(prediction, target)
            model.losses_and_metrics["dice_coef"].append(dice_coeff)

        # I'll convert the list into a PyTorch tensor
        model.pt_loss = sum(model.pt_loss)

        if "epoch_loss" not in model.losses_and_metrics:  # Sum of all losses
            model.losses_and_metrics["epoch_loss"] = []
        model.losses_and_metrics["epoch_loss"].append(float(model.pt_loss))

        print(
            "    Batch {}/{} ({:.0f}%)\tLoss: {:.6f}".format(
                idx + 1, n_imgs, 100.0 * (idx + 1) / n_imgs,
                float(model.pt_loss)
            )
        )


class ImageTargetProblem(ProblemHandler, ABC):
    """ Image-Target Problem. Basic abstract extension of ProblemHandler.

    This is used as a generic class for nifti images, leaving only
    undefined the datasets. The write_predictions method is able to write
    multi-channel batches of images.
    """

    def __init__(self, train_dataset, test_dataset):
        super(ImageTargetProblem, self).__init__(train_dataset, test_dataset)

    def write_predictions(self, predictions, input_filepaths,
                          output_folder_name, input_imgs):
        """ Get predictions of a batch of images and save them as sitk images.

        :param input_filepaths: Paths of the input images in the batch.
        :param predictions: prediction of the batch.
        :param output_folder_name: Name of the pred_folder with the outputs.
        :param input_imgs: Input images.
        :return:
        """
        print(" Saving prediction for...")

        # For each element in the batch
        for prediction, input_filepath, input_img in zip(predictions,
                                                         input_filepaths,
                                                         input_imgs):
            path, name = os.path.split(input_filepath)
            print("  " + name + "..", end='')
            out_folder = utils.veri_folder(
                os.path.join(path, "pred_" + output_folder_name)
            )

            sitk_orig_img = sitk.ReadImage(input_filepath)  # Input image
            origin, direction, spacing = utils.get_sitk_metadata(sitk_orig_img)

            np_out = prediction.cpu().detach()  # Model prediction
            np_out = utils.hard_segm_from_tensor(np_out).numpy()
            sitk_out = utils.get_sitk_img(np_out, origin, direction, spacing)
            if type(sitk_out) == list:
                for i, sitk_out_img in enumerate(sitk_out):  # Several channels
                    name_c = name.replace('.nii.gz', f'_c{i}.nii.gz')
                    sitk.WriteImage(sitk_out_img,
                                    os.path.join(out_folder, name_c))
            else:
                sitk.WriteImage(sitk_out, os.path.join(out_folder, name))

            # Write input image
            sitk.WriteImage(sitk_orig_img,
                            os.path.join(out_folder,
                                         name.replace('.nii.gz', '_i.nii.gz')))


class FlapRec(ImageTargetProblem):
    """ Basic Flap Reconstruction problem.

    Simulate flaps only while training. In test time load only the images.
    """

    def __init__(self):
        super(FlapRec, self).__init__(FlapRecTrainDataset, NiftiImageDataset)


class FlapRecWithShapePrior(ImageTargetProblem):
    """ Flap Reconstruction with Shape Priors problem.

    Simulate flaps only while training. In test time load only the images.
    It will always load a Shape Prior (typically an atlas) and concatenate
    it to the input.

    """

    def __init__(self):
        super(FlapRecWithShapePrior, self).__init__(
            FlapRecWShapePriorTrainDataset, NiftiImageWithAtlasDataset
        )


class FlapRecWithShapePriorDoubleOut(ImageTargetProblem):
    """ Flap Reconstruction with Shape Priors problem with double output.

    Simulate flaps only while training. In test time load only the images.
    It will always load a Shape Prior (typically an atlas) and concatenate
    it to the input.
    The model output will be a two-channel binary image, consisting in the
    predicted bone flap in the first channel, and the full skull (input+flap)
    in the second channel.

    """

    def __init__(self, with_sp=True):
        if with_sp:
            super(FlapRecWithShapePriorDoubleOut, self).__init__(
                FlapRecWShapePrior2OTrainDataset, NiftiImageWithAtlasDataset
            )
        else:
            super(FlapRecWithShapePriorDoubleOut, self).__init__(
                FlapRec2OTrainDataset, NiftiImageDataset
            )

    @staticmethod
    def comp_losses_metrics(model, prediction, target, idx, n_imgs):
        """ Compute the losses and required metrics.

        Calculate the losses depending on the model phase, and save it in
        self.pt_loss. It contains two basic losses and one metric.

        :param model: Model instance. It contains all the training related
        information.
        :param prediction: Model prediction.
        :param target: Target image.
        :param idx: Batch index
        :param n_imgs: Number of imgs in the batch
        :return:
        """
        model.pt_loss = []  # List that will contain all the weighted losses.

        full_skull_p, flap_p = prediction
        full_skull_t, flap_t = target

        if model.params["dice_lambda"] or model.params["save_dice_plots"]:
            full_skull_p_sm = softmax(full_skull_p, dim=1)
            flap_p_sm = softmax(flap_p, dim=1)

        lm = model.losses_and_metrics

        # /!\ Loss functions /!\
        # Binary cross entropy between model output and target
        if model.params["ce_lambda"] != 0:
            if "ce_sk" not in lm:
                lm["ce_sk"] = []  # For visualization
            if "ce_fl" not in lm:
                lm["ce_fl"] = []

            full_skull_t_am = torch.argmax(full_skull_t, 1)
            flap_t_am = torch.argmax(flap_t, 1)

            # CE Loss of the skull
            ce_loss_s = nn.CrossEntropyLoss()(full_skull_p, full_skull_t_am)
            model.pt_loss.append(model.params["ce_lambda"] * ce_loss_s)
            lm["ce_sk"].append(float(model.pt_loss[-1]))

            # CE Loss of the flap
            ce_loss_f = nn.CrossEntropyLoss()(flap_p, flap_t_am)
            model.pt_loss.append(model.params["ce_lambda"] * ce_loss_f)
            lm["ce_fl"].append(float(model.pt_loss[-1]))

        if model.params["dice_lambda"] != 0:  # Dice loss
            if "dice_loss_sk" not in lm:
                lm["dice_loss_sk"] = []
            if "dice_loss_fl" not in lm:
                lm["dice_loss_fl"] = []

            # Dice Loss of the skull
            dice_loss_s = utils.dice_loss()(full_skull_p_sm, full_skull_t)
            model.pt_loss.append(model.params["dice_lambda"] * dice_loss_s)
            lm["dice_loss_sk"].append(float(model.pt_loss[-1]))

            # Dice Loss of the flap
            dice_loss_f = utils.dice_loss()(flap_p_sm, flap_t)
            model.pt_loss.append(model.params["dice_lambda"] * dice_loss_f)
            lm["dice_loss_fl"].append(float(model.pt_loss[-1]))

        # /!\ Metrics /!\
        if model.params["save_dice_plots"] is True:  # Dice coefficient
            if "dice_coef_sk" not in lm:
                lm["dice_coef_sk"] = []
            if "dice_coef_fl" not in lm:
                lm["dice_coef_fl"] = []
            dice_coeff_sk = utils.dice_coeff(full_skull_p_sm, full_skull_t)
            lm["dice_coef_sk"].append(dice_coeff_sk)
            dice_coeff_fl = utils.dice_coeff(flap_p_sm, flap_t)
            lm["dice_coef_fl"].append(dice_coeff_fl)

        # I'll convert the list into a PyTorch tensor
        model.pt_loss = sum(model.pt_loss)

        if "epoch_loss" not in lm:  # Sum of all losses
            lm["epoch_loss"] = []
        lm["epoch_loss"].append(float(model.pt_loss))

        print(
            "    Batch {}/{} ({:.0f}%)\tLoss: {:.6f}".format(
                idx + 1, n_imgs, 100.0 * (idx + 1) / n_imgs,
                float(model.pt_loss)
            )
        )

    def write_predictions(self, predictions, input_filepaths,
                          output_folder_name, input_imgs):
        """ Get predictions of a batch of images and save them as sitk images.

        :param input_filepaths: Paths of the input images in the batch.
        :param predictions: prediction of the batch.
        :param output_folder_name: Name of the pred_folder with the outputs.
        :param input_imgs: Input images.
        :return:
        """
        print(" Saving prediction for...")

        encoded_full_skulls, encoded_flaps = predictions

        # For each element in the batch
        for pred_sk, pred_fl, inp_path in zip(encoded_full_skulls,
                                              encoded_flaps,
                                              input_filepaths):
            path, name = os.path.split(inp_path)
            print("  " + name + "..", end='')
            out_folder = utils.veri_folder(
                os.path.join(path, "pred_" + output_folder_name)
            )

            sitk_orig_img = sitk.ReadImage(inp_path)  # Input image
            origin, direction, spacing = utils.get_sitk_metadata(sitk_orig_img)

            for pred, nme in zip([pred_sk, pred_fl], ['sk', 'fl']):
                np_out = pred.cpu().detach()  # Model prediction
                np_out = utils.hard_segm_from_tensor(np_out).numpy()
                sitk_out = utils.get_sitk_img(np_out, origin, direction,
                                              spacing)
                o_name = name.replace('.nii.gz', '_' + nme + '.nii.gz')
                sitk.WriteImage(sitk_out, os.path.join(out_folder, o_name))

                # Write input image
                sitk.WriteImage(sitk_orig_img,
                                os.path.join(out_folder,
                                             name.replace('.nii.gz',
                                                          '_i.nii.gz')))


class FlapRecDoubleOut(FlapRecWithShapePriorDoubleOut):
    def __init__(self):
        super(FlapRecDoubleOut, self).__init__(with_sp=False)


class DenoisingAE(ImageTargetProblem):
    """ Denoising Autoencoder.

    For training, it will only add salt and pepper noise.
    """

    def __init__(self):
        super(DenoisingAE, self).__init__(
            BinaryDenoisingAEDatasetv2, NiftiImageDataset
        )
