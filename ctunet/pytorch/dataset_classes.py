# In this file I'll put the PyTorch Dataset definitions, needed for proper
# loading of images. It's recommended to declare image augmentation here,
# using inheritance and keeping a simple instancing interface.

# In the normal usage, the only arguments that can be left unfilled are
# csv_file, root_dir, and single_file, which will be eventually used in the
# Model class.

import os

import SimpleITK as sitk
import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import flap_rec_transform, cranioplasty_transform, \
    salt_and_pepper_ae
from .. import utilities as utils

src = os.path.expanduser('~/headctools/assets/atlas/reg')
ATLASES = {
    (224, 304, 304): os.path.join(src, 'atlas_304_224.nii.gz'),
    (224, 512, 512): os.path.join(src, 'atlas_skull_512_224.nii.gz')
}

def load_atlas_and_append_at_axis(image, axis=0, im_size=None):
    # If not provided, grab the last three dims
    im_size = image.shape[-3:] if not im_size else im_size
    if im_size not in ATLASES:
        avail_sizes = ', '.join([str(t) for t in ATLASES.keys()])
        raise FileNotFoundError(f"The input images have a size ({im_size}) "
                                f"different than the available atlases sizes: "
                                f"{avail_sizes}.")
    atlas_path = ATLASES[tuple(im_size)]
    if os.path.exists(atlas_path):
        atlas_path = torch.tensor(
            sitk.GetArrayFromImage(sitk.ReadImage(atlas_path)),
            dtype=torch.float).unsqueeze(axis)
        image = torch.cat((image, atlas_path), axis)
    else:
        raise Exception(f"Atlas not found {atlas_path}.")

    return image


class BrainSegmentationDataset(Dataset):
    """
    This dataset loads the CT scans and its masks
    """

    def __init__(self, csv_file=None, root_dir="", transform=None,
                 single_file=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if single_file is not None:
            self.files_frame = pd.DataFrame(data={'image': [single_file],
                                                  'mask': ['']})
        else:
            self.files_frame = pd.read_csv(csv_file, )
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.files_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.files_frame.iloc[idx, 0])
        seg_name = os.path.join(self.root_dir,
                                self.files_frame.iloc[idx, 1])

        image = sitk.GetArrayFromImage(sitk.ReadImage(img_name))
        segmentation = sitk.GetArrayFromImage(
            sitk.ReadImage(seg_name)) if os.path.exists(seg_name) else image

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        target = utils.one_hot_encoding(
            torch.tensor(segmentation, dtype=torch.float32).unsqueeze(
                0)).squeeze(0)
        sample = {'image': image,
                  'target': target,
                  'filepath': img_name}

        return sample


class NiftiImageWithAtlasDataset(Dataset):
    """ This dataset loads the CT scans only.

    This means if used as it's defined, the torch model will receive only a
    sample['image'] as input and the sample['target'] will remain undefined,
    This is useful for self-supervised models or AE that define the target
    based in the input.

    Since you have to define sample['target'], in order to use this class
    you should inherit it first and pass a transform that defines inside of
    it sample['target'] from sample['image'].

    Also, a skull atlas is concatenated in the input images by default,
    so that the model inputs are the broken image and a full skull atlas.

    """

    def __init__(self, csv_file=None, root_dir="", transform=None,
                 append_atlas=True, single_file=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        if single_file is not None:
            self.files_frame = pd.DataFrame(data={'image': [single_file],
                                                  'mask': ['']})
        else:
            self.files_frame = pd.read_csv(csv_file, )
        self.root_dir = root_dir
        self.transform = transform
        self.append_atlas = append_atlas

    def __len__(self):
        return len(self.files_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.files_frame.iloc[idx, 0])
        sample = {'image': torch.tensor(
            sitk.GetArrayFromImage(sitk.ReadImage(img_name)),
            dtype=torch.float),
            'filepath': img_name}

        im_size = sample['image'].shape
        sample['image'] = sample['image'].unsqueeze(0)  # Batch dimension
        if self.transform:
            sample = self.transform(sample)

        if self.append_atlas:
            sample['image'] = load_atlas_and_append_at_axis(sample['image'], 0,
                                                            im_size)

        return sample


class NiftiImageDataset(NiftiImageWithAtlasDataset):
    """ This dataset loads the CT scans only.

    This means if used as it's defined, the torch model will receive only a
    sample['image'] as input and the sample['target'] will remain undefined,
    This is useful for self-supervised models or AE that define the target
    based in the input.

    Since you have to define sample['target'], in order to use this class
    you should inherit it first and pass a transform that defines inside of
    it sample['target'] from sample['image'].

    """

    def __init__(self, csv_file=None, root_dir="", transform=None,
                 single_file=None):
        super(NiftiImageDataset, self).__init__(csv_file, root_dir, transform,
                                                append_atlas=False,
                                                single_file=single_file)


class FlapRecTrainDataset(NiftiImageDataset):
    """ Flap reconstruction train dataset.

    It implements the NiftiImageDataset with the flap_rec_transform transform.
    This means it will take binary skull images, extract a bone flap and set
    the sample['image'] as the broken skull and sample['target'] as the
    extracted bone flap.

    """

    def __init__(self, csv_file=None, root_dir="", single_file=None):
        super(FlapRecTrainDataset, self).__init__(csv_file, root_dir,
                                                  flap_rec_transform,
                                                  single_file)


class FlapRecWShapePrior2OTrainDataset(NiftiImageDataset):
    """ Dataset for applying a cranioplasty transform in only some cases.

    The transforms will be done to the images with the id supplied,
    otherwise it will be assumed that the transform was applied previously
    and its provided as 'mask' in the csv. This is useful when the
    generation process takes too much time.

    By default the target is a two channel image containing the extracted
    bone flap and the full skull. See the complementary_output parameter.
    The difference with FlapRecWShapePriorTrainDataset lies in the target
    image: in this case, it will only be a two-channel target image,
    thought for predicting both the bone flap and the full skull
    simultaneously.

    :param csv_file: Input CSV. It must contain two path strings per line,
    indicating the skull path in the first place.
    :param root_dir: Root path. Useful when the CSV file contains only the
    filenames.
    :param already_augmented_id: Identifier skulls with previously extracted
    flaps.
    :param fr_transform: Transform to apply.
    :param append_atlas: Append atlas to the images.
    :param single_file: For passing just a single file and not a CSV for
    inference.
    :param append_full: If true, the target img will also contain the full
    skull (flap + skull with hole) in other channel.
    """

    def __init__(self, csv_file=None, root_dir="",
                 already_augmented_id='nfg',
                 fr_transform=cranioplasty_transform,
                 append_atlas=True,
                 single_file=None,
                 append_full=True):
        super(FlapRecWShapePrior2OTrainDataset, self).__init__(csv_file,
                                                               root_dir,
                                                               single_file)
        self.already_augmented_id = already_augmented_id
        self.transform = fr_transform
        self.append_atlas = append_atlas
        self.append_full = append_full

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.files_frame.iloc[idx, 0])

        # We will apply the transform based on the file path
        image = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(img_name)),
                             dtype=torch.float)
        im_size = image.shape

        # In this case I'll always apply the flap extraction
        if self.already_augmented_id not in os.path.split(img_name)[1]:
            sample = {'image': image, 'filepath': img_name}
            sample = self.transform(sample, self.append_full)
        else:  # The flap is already extracted
            flap_path = os.path.join(self.root_dir,
                                     self.files_frame.iloc[idx, 1])
            flap = torch.from_numpy(
                sitk.GetArrayFromImage(sitk.ReadImage(flap_path)))
            full_skull = image + flap
            full_skull = utils.one_hot_encoding(full_skull.unsqueeze(0))
            flap = utils.one_hot_encoding(flap.unsqueeze(0))

            target = full_skull.squeeze(0), flap.squeeze(0)
            sample = {'image': image.unsqueeze(0),
                      'target': target,
                      'filepath': img_name}

        if self.append_atlas:
            sample['image'] = load_atlas_and_append_at_axis(sample['image'], 0,
                                                            im_size)

        return sample


class FlapRec2OTrainDataset(FlapRecWShapePrior2OTrainDataset):
    """ Flap Reconstruction without Shape Priors and with double output.

    It inherits the Shape Prior class, but discarding its file path, avoiding
    the atlas concatenation.

    """

    def __init__(self, csv_file=None, root_dir="", single_file=None):
        super(FlapRec2OTrainDataset, self).__init__(csv_file, root_dir,
                                                    single_file=single_file,
                                                    append_atlas='')


class FlapRecWShapePriorTrainDataset(FlapRecWShapePrior2OTrainDataset):
    """ Dataset for applying a cranioplasty transform in only some cases.

    The difference with FlapRecWShapePrior2OTrainDataset lies in the target
    image: in this case, it will only be a single target image, thought for
    predicting the bone flap only.

    :param csv_file: Input CSV. It must contain two path strings per line,
    indicating the skull path in the first place.
    :param root_dir: Root path. Useful when the CSV file contains only the
    filenames.
    :param full_skull_fileid: Identifier of full skulls,
    :param fr_transform: Transform to apply.
    :param append_atlas: Append atlas to the images.
    :param single_file: For passing just a single file and not a CSV for
    inference.
    """

    def __init__(self, csv_file=None, root_dir="",
                 full_skull_fileid='complete_skull',
                 fr_transform=cranioplasty_transform,
                 append_atlas='~/headctools/assets/atlas/reg/atlas_304_224.nii'
                              '.gz', single_file=None):
        super(FlapRecWShapePriorTrainDataset, self).__init__(csv_file,
                                                             root_dir,
                                                             full_skull_fileid,
                                                             fr_transform,
                                                             append_atlas,
                                                             single_file,
                                                             append_full=False)


class BinaryDenoisingAEDataset(NiftiImageDataset):
    """ Binary Denoising AutoEncoder train dataset.

    It adds salt and pepper noise to the images. The target is the input image.

    """

    def __init__(self, csv_file=None, root_dir="", single_file=None):
        super(BinaryDenoisingAEDataset, self).__init__(csv_file, root_dir,
                                                       salt_and_pepper_ae,
                                                       single_file)


class BinaryDenoisingAEDatasetv2(NiftiImageDataset):
    """ TODO Make this class a subclass of BinaryDenoisingAEDataset

    also change the name. this only works in already augmented imgs

    """

    def __init__(self, csv_file=None, root_dir="",
                 already_augmented_id='nfg',
                 transform=salt_and_pepper_ae,
                 single_file=None,
                 append_full=True):
        super(BinaryDenoisingAEDatasetv2, self).__init__(csv_file,
                                                         root_dir,
                                                         transform,
                                                         single_file)
        self.already_augmented_id = already_augmented_id
        self.append_full = append_full

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.files_frame.iloc[idx, 0])

        # We will apply the transform based on the file path
        image = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(img_name)),
                             dtype=torch.float)

        # In this case I'll always apply the flap extraction
        flap_path = os.path.join(self.root_dir,
                                 self.files_frame.iloc[idx, 1])
        flap = torch.from_numpy(
            sitk.GetArrayFromImage(sitk.ReadImage(flap_path)))
        full_skull = image + flap
        full_skull = utils.one_hot_encoding(full_skull.unsqueeze(0))

        target = full_skull.squeeze(0)
        sample = {'image': image.unsqueeze(0),
                  'target': target,
                  'filepath': img_name}

        sample = self.transform(sample)

        return sample
