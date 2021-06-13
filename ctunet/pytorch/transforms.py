import torch

from .. import utilities as utils


class FlapRecTransform:
    def __new__(cls, img=None):
        if img is not None:
            return cls.apply_transform(img)

    def apply_transform(self, sample):
        sample['image'], sample['target'] = utils.skull_random_hole(
            sample['image'], prob=0.8, return_extracted=True)
        sample['image'] = utils.salt_and_pepper(sample['image'],
                                                noise_probability=.01,
                                                noise_density=.05)
        sample['target'] = torch.tensor(sample['target'], dtype=torch.float32)
        sample['image'] = torch.tensor(sample['image'], dtype=torch.float32)
        sample['target'] = utils.one_hot_encoding(
            sample['target'])  # Encode the target
        sample['image'] = sample['image'].unsqueeze(
            1)  # Add the channel dimension

        return sample


def flap_rec_transform(sample):
    sample['image'] = sample['image'].unsqueeze(0)
    sample['image'], sample['target'] = utils.skull_random_hole(
        sample['image'], prob=.8, return_extracted=True)
    sample['image'] = utils.salt_and_pepper(sample['image'],
                                            noise_probability=.01,
                                            noise_density=.05)
    sample['target'] = torch.tensor(sample['target'], dtype=torch.float32)
    sample['image'] = torch.tensor(sample['image'], dtype=torch.float32)
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
    full_skull = utils.erode_dilate(full_skull, p=0.7)

    # BONE FLAP EXTRACTION
    full_skull = full_skull.unsqueeze(0)
    incomp_skull, flap = utils.skull_random_hole(full_skull.clone(), prob=0.8,
                                                 return_extracted=True)

    # SALT AND PEPPER NOISE
    incomp_skull = utils.salt_and_pepper(incomp_skull,
                                         noise_probability=.01,
                                         noise_density=.05)

    sample['image'] = torch.tensor(incomp_skull, dtype=torch.float32)
    if return_full:
        flap = utils.one_hot_encoding(flap)
        full_skull = utils.one_hot_encoding(full_skull)
        sample['target'] = torch.cat((full_skull[0, 0:1],
                                      flap[0, 1:2],
                                      full_skull[0, 1:2]))
    else:
        flap = torch.from_numpy(flap).float()
        sample['target'] = utils.one_hot_encoding(flap).squeeze(0)

    return sample


def salt_and_pepper_ae(sample):
    if 'target' not in sample:
        sample['target'] = sample['image'].clone()
    sample['image'] = utils.salt_and_pepper(sample['image'],
                                            noise_probability=.8,
                                            noise_density=.3)
    sample['image'] = torch.from_numpy(sample['image']).float()
    return sample
