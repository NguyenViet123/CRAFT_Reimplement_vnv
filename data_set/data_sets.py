from .module.base_dataset import BaseDataset
import os
import cv2
import numpy as np
from .module.utils import random_scale
import re
import itertools


class Synth80k(BaseDataset):

    def __init__(self, folder, target_size=768, viz=False, debug=False):
        super(Synth80k, self).__init__(target_size, viz, debug)
        self.folder = folder
        self.charbox = gt['charBB'][0]
        self.image = gt['imnames'][0]
        self.imgtxt = gt['txt'][0]

    def __getitem__(self, index):
        return self.pull_item(index)

    def __len__(self):
        return len(self.imgtxt)

    def get_imagename(self, index):
        return self.image[index][0]

    def load_image_gt_and_confidencemask(self, index):
        '''
        根据索引加载ground truth
        :param index:索引
        :return:bboxes 字符的框，
        '''
        img_path = os.path.join(self.folder, self.image[index][0])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _charbox = self.charbox[index].transpose((2, 1, 0))
        image = random_scale(image, _charbox, self.target_size)

        words = [re.split(' \n|\n |\n| ', t.strip()) for t in self.imgtxt[index]]
        words = list(itertools.chain(*words))
        words = [t for t in words if len(t) > 0]
        character_bboxes = []
        total = 0
        confidences = []
        for i in range(len(words)):
            bboxes = _charbox[total:total + len(words[i])]
            assert (len(bboxes) == len(words[i]))
            total += len(words[i])
            bboxes = np.array(bboxes)
            character_bboxes.append(bboxes)
            confidences.append(1.0)

        return image, character_bboxes, words, np.ones((image.shape[0], image.shape[1]), np.float32), confidences


if __name__ == '__main__':
