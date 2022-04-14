import logging
import torch.nn as nn
from fastai.vision import *

from modules.attention import *
from modules.backbone import ResTranformer
from modules.model import Model
from modules.resnet import resnet45


class BaseVision(Model):
    def __init__(self, config):
        super().__init__(config)
        self.loss_weight = ifnone(config.model_vision_loss_weight, 1.0)
        self.out_channels = ifnone(config.model_vision_d_model, 512)

        if config.model_vision_backbone == 'transformer':
            self.backbone = ResTranformer(config)
        else: self.backbone = resnet45()
        
        if config.model_vision_attention == 'position':
            mode = ifnone(config.model_vision_attention_mode, 'nearest')
            self.attention = PositionAttention(
                in_channels=self.out_channels,
                max_length=config.dataset_max_length + 1,  # additional stop token
                mode=mode,
            )
        elif config.model_vision_attention == 'attention':
            self.attention = Attention(
                in_channels=self.out_channels,
                max_length=config.dataset_max_length + 1,  # additional stop token
                n_feature=8*32,
            )
        else:
            raise Exception(f'{config.model_vision_attention} is not valid.')
        self.cls = nn.Linear(self.out_channels, self.charset.num_classes)

        if config.model_vision_checkpoint is not None:
            logging.info(f'Read vision model from {config.model_vision_checkpoint}.')
            self.load(config.model_vision_checkpoint)

    def _forward(self, b_features):
        attn_vecs, attn_scores = self.attention(b_features)  # (N, T, E), (N, T, H, W)
        logits = self.cls(attn_vecs) # (N, T, C)
        pt_lengths = self._get_length(logits)

        return {'feature': attn_vecs, 'logits': logits, 'pt_lengths': pt_lengths,
                'attn_scores': attn_scores, 'loss_weight':self.loss_weight, 'name': 'vision', 'b_features':b_features}

    def forward(self, images, *args, **kwargs):
        features = self.backbone(images, **kwargs)  # (N, E, H, W)
        return self._forward(features)
        

class BaseIterVision(BaseVision):
    def __init__(self, config):
        super().__init__(config)
        assert config.model_vision_backbone == 'transformer'
        self.iter_size = ifnone(config.model_vision_iter_size, 1)
        self.share_weights = ifnone(config.model_vision_share_weights, False)
        self.share_cnns = ifnone(config.model_vision_share_cnns, False)
        self.add_transformer = ifnone(config.model_vision_add_transformer, False)
        self.simple_trans = ifnone(config.model_vision_simple_trans, False)
        self.deep_supervision = ifnone(config.model_vision_deep_supervision, True)
        self.backbones = nn.ModuleList()
        self.trans = nn.ModuleList()
        for i in range(self.iter_size-1):
            B = None if self.share_weights else ResTranformer(config)
            if self.share_cnns:
                del B.resnet
            self.backbones.append(B)
            output_channel = self.out_channels
            if self.add_transformer:
                self.split_sizes = [output_channel]
            elif self.simple_trans:
                # self.split_sizes=[output_channel//16] + [0] * 5
                # self.split_sizes= [output_channel//16, output_channel//16, output_channel//8, output_channel//4, output_channel//2] + [0]
                self.split_sizes= [output_channel//16, output_channel//16, 0, output_channel//4, output_channel//2, output_channel]
            else:
                self.split_sizes=[output_channel//16, output_channel//16, output_channel//8, output_channel//4, output_channel//2, output_channel]
            self.trans.append(nn.Conv2d(output_channel, sum(self.split_sizes), 1))
            torch.nn.init.zeros_(self.trans[-1].weight)
        
        if config.model_vision_checkpoint is not None:
            logging.info(f'Read vision model from {config.model_vision_checkpoint}.')
            self.load(config.model_vision_checkpoint)
        cb_init = ifnone(config.model_vision_cb_init, True)
        if cb_init:
            self.cb_init()

    def load(self, source, device=None, strict=False):
        state = torch.load(source, map_location=device)
        msg = self.load_state_dict(state['model'], strict=strict)
        print(msg)

    def cb_init(self):
        model_state_dict = self.backbone.state_dict()

        for m in self.backbones:
            if m:
                print('cb_init')
                msg = m.load_state_dict(model_state_dict, strict=False)
                print(msg)

    def forward_test(self, images, *args):
        l_feats = self.backbone.resnet(images)
        b_feats = self.backbone.forward_transformer(l_feats)
        cnt = len(self.backbones)
        if cnt == 0:
            v_res = super()._forward(b_feats)
        for B,T in zip(self.backbones, self.trans):
            cnt -= 1
            extra_feats = T(b_feats).split(self.split_sizes, dim=1)
            if self.share_weights:
                v_res = super().forward(images, extra_feats=extra_feats)
            else:
                if self.add_transformer:
                    if not self.share_cnns:
                        l_feats = B.resnet(images)
                    b_feats = B.forward_transformer(extra_feats[-1] + l_feats)
                else:
                    b_feats = B(images, extra_feats=extra_feats)
                v_res = super()._forward(b_feats) if cnt==0 else None
        return v_res

    def forward_train(self, images, *args):
        l_feats = self.backbone.resnet(images)
        b_feats = self.backbone.forward_transformer(l_feats)
        v_res = super()._forward(b_feats)
        # v_res = super().forward(images)
        all_v_res = [v_res]
        for B,T in zip(self.backbones, self.trans):
            extra_feats = T(v_res['b_features']).split(self.split_sizes, dim=1)
            if self.share_weights:
                v_res = super().forward(images, extra_feats=extra_feats)
            else:
                if self.add_transformer:
                    if not self.share_cnns:
                        l_feats = B.resnet(images)
                    b_feats = B.forward_transformer(extra_feats[-1] + l_feats)
                else:
                    b_feats = B(images, extra_feats=extra_feats)
                v_res = super()._forward(b_feats)
            all_v_res.append(v_res)
        return all_v_res

    def forward(self, images, *args):
        if self.training and self.deep_supervision:
            return self.forward_train(images, *args)
        else:
            return self.forward_test(images, *args)