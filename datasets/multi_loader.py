"""
Custom DomainUniformConcatDataset
"""
import numpy as np

from torch.utils.data import Dataset
import torch
from config import cfg


np.random.seed(cfg.RANDOM_SEED)


class DomainUniformConcatDataset(Dataset):
    """
    DomainUniformConcatDataset

    Sample images uniformly across the domains
    If bs_mul is n, this outputs # of domains * n images per batch
    """
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, args, datasets):
        """
        This dataset is to return sample image (source)
        and augmented sample image (target)
        Args:
            args: input config arguments
            datasets: list of datasets to concat
        """
        super(DomainUniformConcatDataset, self).__init__()
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.offsets = self.cumsum(datasets)
        self.length = np.sum(self.lengths)

        print("# domains: {}, Total length: {}, 1 epoch: {}, offsets: {}".format(
            str(len(datasets)), str(self.length), str(len(self)), str(self.offsets)))


    def __len__(self):
        """
        Returns:
            The number of images in a domain that has minimum image samples
        """
        return min(self.lengths)


    def _get_batch_from_dataset(self, dataset, idx):
        """
        Get batch from dataset
        New idx = idx + random integer
        Args:
            dataset: dataset class object
            idx: integer

        Returns:
            One batch from dataset
        """
        p_index = idx + np.random.randint(len(dataset))
        if p_index > len(dataset) - 1:
            p_index -= len(dataset)

        return dataset[p_index]


    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            images corresonding to the index from each domain
        """
        imgs = []
        masks = []
        img_names = []
        mask_auxs = []

        for dataset in self.datasets:
            img, mask, img_name, mask_aux = self._get_batch_from_dataset(dataset, idx)
            imgs.append(img)
            masks.append(mask)
            img_names.append(img_name)
            mask_auxs.append(mask_aux)
        imgs, masks, mask_auxs = torch.stack(imgs, 0), torch.stack(masks, 0), torch.stack(mask_auxs, 0)

        return imgs, masks, img_names, mask_auxs

