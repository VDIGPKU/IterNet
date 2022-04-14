import torch
import torch.nn as nn
from fastai.vision import *

from .model_vision import BaseIterVision
from .model_language import BCNLanguage
from .model_alignment import BaseAlignment

class IterNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.iter_size = ifnone(config.model_iter_size, 1)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.vision = BaseIterVision(config)
        self.language = BCNLanguage(config)
        self.alignment = BaseAlignment(config)
        self.deep_supervision = ifnone(config.model_deep_supervision, True)

    def forward(self, images, *args):
        list_v_res = self.vision(images)
        if not isinstance(list_v_res, (list, tuple)):
            list_v_res = [list_v_res]
        all_l_res, all_a_res = [], []

        for v_res in list_v_res:
            a_res = v_res
            for _ in range(self.iter_size):
                tokens = torch.softmax(a_res['logits'], dim=-1)
                lengths = a_res['pt_lengths']
                lengths.clamp_(2, self.max_length)  # TODO:move to langauge model
                l_res = self.language(tokens, lengths)
                all_l_res.append(l_res)
                a_res = self.alignment(l_res['feature'], v_res['feature'])
                all_a_res.append(a_res)
        if self.training and self.deep_supervision:
            return all_a_res, all_l_res, list_v_res
        else:
            return a_res, all_l_res[-1], list_v_res[-1]
