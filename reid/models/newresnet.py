from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import softplus

from reid.models.baseline import Baseline
from utlis import  ChannelCompress, to_edge

__all__ = ['ft_net']

##################################################################################
# Initialization function
##################################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

##################################################################################
# framework
##################################################################################
class ft_net(nn.Module):
    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)


    def __init__(self, args, num_classes, num_features):
        super(ft_net, self).__init__()

        # Load basic config to initialize encoders, decoders and discriminators
        self.args = args

        self.IR_backbone = Baseline(num_classes, num_features)
        self.IR_Bottleneck = VIB(in_ch=2048, z_dim=self.args.z_dim, num_class= num_classes)
        #self.IR_MIE = MIEstimator(size1=2048, size2=int(2 * self.args.z_dim))

        self.RGB_backbone = Baseline(num_classes, num_features)
        self.RGB_Bottleneck = VIB(in_ch=2048, z_dim=self.args.z_dim, num_class= num_classes)
        #self.RGB_MIE = MIEstimator(size1=2048, size2=int(2 * self.args.z_dim))

        self.shared_backbone = Baseline(num_classes, num_features)
        self.shared_Bottleneck = VIB(in_ch=2048, z_dim=self.args.z_dim, num_class= num_classes)
        #self.shared_MIE = MIEstimator(size1=2048, size2=int(2 * self.args.z_dim))

    def forward(self, x):
        # visible branch
        v_observation = self.RGB_backbone(x)
        v_representation = self.RGB_Bottleneck(v_observation[1])

        # modal-shared branch
        x_grey = to_edge(x)
        i_ms_input = torch.cat([x_grey, x_grey, x_grey], dim=1)

        i_ms_observation = self.shared_backbone(i_ms_input)
        v_ms_observation = self.shared_backbone(x)

        i_ms_representation = self.shared_Bottleneck(i_ms_observation[1])
        v_ms_representation = self.shared_Bottleneck(v_ms_observation[1])

        # infrared branch
        i_observation = self.IR_backbone(i_ms_input)
        i_representation = self.IR_Bottleneck(i_observation[1])

        return i_observation, i_representation, i_ms_observation, i_ms_representation, \
               v_observation, v_representation, v_ms_observation, v_ms_representation

    def optims(self):
        conv_params = []

        conv_params += list(self.IR_backbone.parameters())
        conv_params += list(self.IR_Bottleneck.bottleneck.parameters())
        conv_params += list(self.IR_Bottleneck.classifier.parameters())

        conv_params += list(self.RGB_backbone.parameters())
        conv_params += list(self.RGB_Bottleneck.bottleneck.parameters())
        conv_params += list(self.RGB_Bottleneck.classifier.parameters())

        conv_params += list(self.shared_backbone.parameters())
        conv_params += list(self.shared_Bottleneck.bottleneck.parameters())
        conv_params += list(self.shared_Bottleneck.classifier.parameters())

        conv_optim = torch.optim.Adam([p for p in conv_params if p.requires_grad], lr=self.args.lr, weight_decay=5e-4)

        return conv_optim

##################################################################################
# Variational Information Bottleneck
##################################################################################
class VIB(nn.Module):
    def __init__(self, in_ch=2048, z_dim=256, num_class=395):
        super(VIB, self).__init__()
        self.in_ch = in_ch
        self.out_ch = z_dim * 2
        self.num_class = num_class
        self.bottleneck = ChannelCompress(in_ch=self.in_ch, out_ch=self.out_ch)
        # classifier of VIB, maybe modified later.
        classifier = []
        classifier += [nn.Linear(self.out_ch, self.out_ch // 2)]
        classifier += [nn.BatchNorm1d(self.out_ch // 2)]
        classifier += [nn.LeakyReLU(0.1)]
        classifier += [nn.Dropout(0.5)]
        classifier += [nn.Linear(self.out_ch // 2, self.num_class)]
        classifier = nn.Sequential(*classifier)
        self.classifier = classifier
        self.classifier.apply(weights_init_classifier)

    def forward(self, v):
        z_given_v = self.bottleneck(v)
        p_y_given_z = self.classifier(z_given_v)
        return p_y_given_z, z_given_v

##################################################################################
# Mutual Information Estimator
##################################################################################
class MIEstimator(nn.Module):
    def __init__(self, size1=2048, size2=512):
        super(MIEstimator, self).__init__()
        self.size1 = size1
        self.size2 = size2
        self.in_ch = size1 + size2
        add_block = []
        add_block += [nn.Linear(self.in_ch, 2048)]
        add_block += [nn.BatchNorm1d(2048)]
        add_block += [nn.ReLU()]
        add_block += [nn.Linear(2048, 512)]
        add_block += [nn.BatchNorm1d(512)]
        add_block += [nn.ReLU()]
        add_block += [nn.Linear(512, 1)]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.block = add_block

    # Gradient for JSD mutual information estimation and EB-based estimation
    def forward(self, x1, x2, x1_shuff):
        """
        :param x1: observation
        :param x2: representation
        """
        pos = self.block(torch.cat([x1, x2], 1))  # Positive Samples
        neg = self.block(torch.cat([x1_shuff, x2], 1))

        return -softplus(-pos).mean() - softplus(neg).mean(), pos.mean() - neg.exp().mean() + 1