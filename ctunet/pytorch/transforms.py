import random

import SimpleITK as sitk
import numpy as np
import torch
import torchio as tio
from ctunet.utilities import shape_3d

from .. import utilities as utils
from torch.nn.functional import one_hot
from torchvision import transforms

composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])


class SaltAndPepper(object):
    def __init__(self, noise_probability=1, noise_density=0.2,
                 salt_ratio=0.1, keyws=('image', 'mask'),
                 apply_to=(True, False)):
        self.noise_probability = noise_probability
        self.noise_density = noise_density
        self.salt_ratio = salt_ratio
        self.keyws = keyws
        self.apply_to = apply_to

    def __call__(self, sample):
        for i, keyw in enumerate(self.keyws):
            if not self.apply_to[i]:
                continue
            img = sample[keyw]
            is_batch = len(img.shape) == 4
            batch_size = img.shape[0] if is_batch else 1
            output = np.copy((img if is_batch else img.unsqueeze(0))).astype(
                np.uint8)
            self.noise_density = np.random.uniform(0, self.noise_density)
            for i in range(batch_size):
                r = random.uniform(0, 1)  # Random number
                if self.noise_probability >= r:  # Inside the probability
                    black_dots = (
                            np.random.uniform(0, 1, output[i, :, :, :].shape)
                            > self.noise_density * (1 - self.salt_ratio)
                    ).astype(np.uint8)
                    white_dots = 1 - (
                            np.random.uniform(0, 1, output[i, :, :, :].shape)
                            > self.noise_density * self.salt_ratio
                    ).astype(np.uint8)
                    output[i, :, :, :] = np.logical_and(output[i, :, :, :],
                                                        black_dots)
                    output[i, :, :, :] = np.logical_or(output[i, :, :, :],
                                                       white_dots)
            output = torch.FloatTensor(output)
            return output if is_batch else output[0]


class FlapRecTransform:
    def __new__(cls, img=None):
        if img is not None:
            return cls.apply_transform(img)

    def apply_transform(self, sample):
        sample['image'], sample['target'] = skull_random_hole(
            sample['image'], prob=0.8, return_extracted=True)
        sample['image'] = SaltAndPepper(sample['image'],
                                        noise_probability=.01,
                                        noise_density=.05)
        sample['target'] = torch.FloatTensor(sample['target'])
        sample['image'] = torch.FloatTensor(sample['image'])
        sample['target'] = utils.one_hot_encoding(
            sample['target'])  # Encode the target
        sample['image'] = sample['image'].unsqueeze(
            1)  # Add the channel dimension

        return sample


def flap_rec_transform(sample):
    sample['image'], sample['target'] = skull_random_hole(
        sample['image'], prob=.8, return_extracted=True)
    sample['image'] = SaltAndPepper(sample['image'],
                                    noise_probability=.01,
                                    noise_density=.05)
    # sample['target'] = torch.tensor(sample['target'], dtype=torch.float32)
    sample['target'] = sample['target'].clone().detach().float()  # TODO gpuvar
    sample['image'] = torch.FloatTensor(sample['image'])
    sample['target'] = utils.one_hot_encoding(sample['target']).squeeze(
        0)  # Encode the target

    return sample


def cranioplasty_transform(sample, return_full=False):
    """ Transform that combines several augmentation methods for flap reconstr.

    It uses the following augmentation additively with a probability of add:
        - random_blank_patch: It extracts a random sphere/cube/autoimplant
        flap from a random location in the surface.
        - salt and pepper noise: For simulating threshold noise / artifacts.
        - OldaKodym'sh DefectGenerator: It simulates complex craniectomy
        defects. https://github.com/OldaKodym/BUT_autoimplant_public
        - Erosions/Dilations prior to the craniectomy generation.

    :param sample: Full skull binary image.
    :param return_full: If true, the target image will contain two channels:
    the bone flap and the full skull. Otherwise, it will be only the flap.
    :return:
    """
    full_skull = sample['image']

    # EROSIONS/DILATIONS
    full_skull = erode_dilate(full_skull, p=0.3)

    # RANDOM HORIZONTAL FLIP IN THE S PLANE TODO ADD PROB
    full_skull = tio.RandomFlip(('S',), .5)(full_skull.unsqueeze(0))[0]

    # SLIGHT DEFROMATION
    red = tio.RandomElasticDeformation(7, locked_borders=2,
                                       image_interpolation='nearest', p=0.5)
    full_skull = red(full_skull.unsqueeze(0))[0]

    # ZOOM/DISPLACEMENT/ROTATION
    aff = tio.RandomAffine(scales=(0.9, 1.1), translation=(10, 10, 15),
                           degrees=15, image_interpolation='nearest',
                           isotropic=False, p=0.5)
    full_skull = aff(full_skull.unsqueeze(0))[0]

    # BONE FLAP EXTRACTION
    incomp_skull, flap = skull_random_hole(full_skull.clone(), prob=0.9,
                                           return_extracted=True)

    # SALT AND PEPPER NOISE
    incomp_skull = SaltAndPepper(incomp_skull,
                                 noise_probability=1,
                                 noise_density=.05)
    sample['image'] = torch.FloatTensor(incomp_skull).unsqueeze(0)  # Channel
    if return_full:
        flap = one_hot(flap.long(), 2).movedim(3, 0)
        full_skull = one_hot(full_skull.long(), 2).movedim(3, 0)

        # sample['target'] = torch.cat((full_skull[0, 0:1],
        #                               flap[0, 1:2],
        #                               full_skull[0, 1:2]))
        sample['target'] = full_skull, flap
    else:
        sample['target'] = one_hot(flap, 2).movedim(3, 0)

    return sample


def salt_and_pepper_ae(sample):
    if 'target' not in sample:
        sample['target'] = sample['image'].clone()
    sample['image'] = SaltAndPepper(sample['image'],
                                    noise_probability=.8,
                                    noise_density=.3)
    sample['image'] = torch.from_numpy(sample['image']).float()
    return sample


def skull_random_hole(img, prob=1, return_extracted=False):
    """ Simulate craniectomies placing random binary shapes.

    Given a batch of 3D images (PyTorch tensors), crop a random cube or box
    placed in a random position of the image with the sizes given in d.

    :param img: Input image.
    :param prob: probability of adding the noise (by default flip a coin).
    :param return_extracted: Return extracted bone flap.
    """
    if not type(img) == torch.Tensor:
        raise TypeError(f"Expected 'torch.Tensor'. Got {type(img)}.")
    is_batch = len(img.shape) == 4
    batch_size = img.shape[0] if is_batch else 1
    output = np.copy((img if is_batch else img.unsqueeze(0))).astype(np.uint8)
    if return_extracted:
        flap = np.copy((img if is_batch else img.unsqueeze(0))).astype(
            np.uint8)
    for i in range(batch_size):
        np_img = output[i, :, :, :]
        if not return_extracted:
            output[i] = random_blank_patch(np_img, prob, return_extracted)
        else:
            output[i], flap[i] = random_blank_patch(np_img, prob,
                                                    return_extracted)
    output = torch.ByteTensor(output)
    if not is_batch:
        output = output[0]
        if return_extracted:
            flap = flap[0]

    if not return_extracted:
        return output
    else:
        flap = torch.ByteTensor(flap)
        return output, flap


def random_blank_patch(image, prob=1, return_extracted=False, p_type="random",
                       apply_plane_cond=False):
    r = random.uniform(0, 1)  # Random number
    if prob >= r:  # Inside the probability -> crop
        image_size = image.shape

        # Select a nonzero pixel
        # TODO Check which selection method is faster
        pixels = np.argwhere(image > 0)
        if pixels.shape[0]:  # The image is not empty
            while pixels.shape[0]:
                center = pixels[np.random.choice(pixels.shape[0])]
                if apply_plane_cond:
                    plane_cond = (
                            center[1] * (
                            3 / 7 * image_size[0] / image_size[1]) +
                            center[0]
                            > 0.65 * image_size[0]
                    )  # Plane
                    if plane_cond:  # white pixel
                        break
                else:
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
                extracted = np.logical_and(image,
                                           1 - shape_np).astype(np.uint8)
                return masked_out, extracted
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


def erode_dilate(inp_img, p=1):
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

    out_img = np.random.choice([dilate])(img)
    # out_img = np.random.choice([erode, dilate])(img) TODO FIX (PREVENT ERODING ALL)

    if is_array:
        return sitk.GetArrayFromImage(out_img)
    elif is_tensor:
        return torch.tensor(sitk.GetArrayFromImage(out_img))
    else:
        return out_img
