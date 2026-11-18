import os

import numpy as np
import torch

# CIFAR-10 normalisation used by the loaders; "white" = (1 - mean) / std per channel.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


class PatchBackdoor:
    """Solid white square trigger in the bottom-right corner, single target class.

    Replaces FigRandBackdoor (3x3 random binary pattern at a mid-grey value with per-sample
    label shift y -> (y+5)%10), which the model never learned as a trigger (audit 2026-11-18:
    ASR identical with the real, random or no trigger). Deterministic: draws no random numbers.
    """

    def __init__(self, target_label, size=4, dataloader=None, save_folder='', save_name='backdoor'):
        if dataloader is None:
            raise RuntimeError('dataloader must be provided.')
        self.name = f'PatchBackdoor{size}'
        self.target_label = int(target_label)
        self.target_class_num = dataloader.target_class_num
        self.image_size = dataloader.raw_data_shape
        self.input_data_shape = dataloader.input_data_shape
        self.size = size
        self.x_top, self.y_top, self.x_len, self.y_len = -size, -size, size, size
        c, h, w = self.image_size
        self.mask = torch.zeros(1, h, w)
        self.mask[:, h - size:, w - size:] = 1.0
        self.white = torch.tensor([(1 - m) / s for m, s in zip(CIFAR_MEAN, CIFAR_STD)]).view(c, 1, 1)
        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            np.save(os.path.join(save_folder, save_name + '.npy'), self)

    def apply(self, batch_x):
        return (1 - self.mask) * batch_x + self.mask * self.white

    def add_backdoor(self, dataset, attack_portion=0.8):
        for [batch_x, batch_y] in dataset:
            n = int(len(batch_y) * attack_portion)
            batch_x[:n] = self.apply(batch_x[:n])
            batch_y[:n] = self.target_label

    @staticmethod
    def choose_target(training_batches, num_classes):
        """Class with the fewest samples in the forget client's training data (ties -> lowest id)."""
        counts = np.zeros(num_classes, dtype=np.int64)
        for _, by in training_batches:
            counts += np.bincount(by.cpu().numpy(), minlength=num_classes)
        return int(np.argmin(counts))
