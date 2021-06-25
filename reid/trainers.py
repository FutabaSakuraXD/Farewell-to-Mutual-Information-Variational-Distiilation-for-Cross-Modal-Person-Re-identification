from __future__ import print_function, absolute_import
import time
import torch

from utlis import SinkhornDistance
from .utils.meters import AverageMeter
from torch.nn.functional import kl_div


class BaseTrainer(object):
    def __init__(self, args, model, ce_loss, rank_loss, associate_loss, trainvallabel):
        super(BaseTrainer, self).__init__()
        self.model = model

        self.args = args

        self.CE_Loss = ce_loss
        self.rank_loss = rank_loss

        self.softmax = torch.nn.Softmax(dim=1)
        self.KLD = torch.nn.KLDivLoss()
        self.W_dist = SinkhornDistance().cuda()

        self.associate_loss = associate_loss

        self.trainvallabel = trainvallabel

    def train(self, epoch, data_loader, conv_optim, print_freq=24):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_total = AverageMeter()

        losses_triple = AverageMeter()
        losses_celoss = AverageMeter()

        losses_cml = AverageMeter()
        losses_vsd = AverageMeter()
        losses_vcd = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, sub, label = self._parse_data(inputs)

            # Calc the loss
            ce_loss, triplet_Loss, conventional_ML, vsd_loss, vcd_loss = self._forward(inputs, label, sub, epoch)
            L = self.args.CE_loss * ce_loss + \
                self.args.Triplet_loss * triplet_Loss + \
                self.args.CML_loss * conventional_ML + \
                self.args.VSD_loss * vsd_loss + \
                self.args.VCD_loss * vcd_loss

            conv_optim.zero_grad()
            L.backward()
            conv_optim.step()

            losses_total.update(L.data.item(), label.size(0))

            losses_celoss.update(ce_loss.item(), label.size(0))
            losses_triple.update(triplet_Loss.item(), label.size(0))

            losses_cml.update(conventional_ML.item(), label.size(0))
            losses_vcd.update(vcd_loss.item(), label.size(0))
            losses_vsd.update(vsd_loss.item(), label.size(0))

            # losses_sharedMI.update(shared_MI.item(), label.size(0))
            # losses_specificMI.update(specific_MI.item(), label.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.2f} ({:.2f})\t'
                      'Total Loss {:.2f} ({:.2f})\t'
                      'IDE Loss {:.2f} ({:.2f})\t'
                      'Triple Loss {:.2f} ({:.2f})\t'
                      'CML Loss {:.3f} ({:.3f})\t'
                      'VSD Loss {:.3f} ({:.3f})\t'
                      'VCD Loss {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              losses_total.val, losses_total.avg,
                              losses_celoss.val, losses_celoss.avg,
                              losses_triple.val, losses_triple.avg,
                              losses_cml.val, losses_cml.avg,
                              losses_vsd.val, losses_vsd.avg,
                              losses_vcd.val, losses_vcd.avg))
        return losses_triple.avg, losses_total.avg

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, cams = inputs
        inputs = imgs.cuda()
        pids = pids.cuda()
        sub = ((cams == 2).long() + (cams == 5).long()).cuda()
        label = torch.cuda.LongTensor(range(pids.size(0)))
        for i in range(pids.size(0)):
            label[i] = self.trainvallabel[pids[i].item()]
        return inputs, sub, label

    def _forward(self, inputs, label, sub, epoch):
        i_observation, i_representation, i_ms_observation, i_ms_representation, \
        v_observation, v_representation, v_ms_observation, v_ms_representation = self.model(inputs)

        # Classification loss
        ce_loss = 0.5 * (self.CE_Loss(i_observation[0], label) + self.CE_Loss(i_representation[0], label)) + \
                  0.5 * (self.CE_Loss(v_observation[0], label) + self.CE_Loss(v_representation[0], label)) + \
                  0.25 * (self.CE_Loss(i_ms_observation[0], label) + self.CE_Loss(i_ms_representation[0], label)) + \
                  0.25 * (self.CE_Loss(v_ms_observation[0], label) + self.CE_Loss(v_ms_representation[0], label))

        # Metric learning, notice rank loss are applied to v and z, respectively.
        triplet_Loss = 0.5 * ( self.rank_loss(i_observation[1], label, sub) + self.rank_loss(i_representation[1], label, sub)) + \
                       0.5 * (self.rank_loss(v_observation[1], label, sub) +  self.rank_loss(v_representation[1], label, sub)) + \
                       0.25 * (self.rank_loss(i_ms_observation[1], label, sub) + self.rank_loss(i_ms_representation[1], label, sub)) + \
                       0.25 * (self.rank_loss(v_ms_observation[1], label, sub) + self.rank_loss(v_ms_representation[1], label, sub))

        #associate_loss = 0.5 * (self.associate_loss(i_observation[1], label, sub) + self.associate_loss(i_representation[1], label, sub)) + \
        #                 0.5 * (self.associate_loss(v_observation[1], label, sub) + self.associate_loss(v_representation[1], label, sub)) + \
        #                 0.25 * (self.associate_loss(i_ms_observation[1], label, sub) + self.associate_loss(i_ms_representation[1], label, sub)) + \
        #                 0.25 * (self.associate_loss(v_ms_observation[1], label, sub) + self.associate_loss(v_ms_representation[1], label, sub))

        # Conventional mutual learning strategy, conducted only between observations of modal-specific branches.
        conventional_ML = self.W_dist(self.softmax(i_observation[0]), self.softmax(v_observation[0]))

        # Variational Self-Distillation, preserving sufficiency
        vsd_loss = kl_div(input=self.softmax(i_observation[0].detach() / self.args.temperature),
                          target=self.softmax(i_representation[0] / self.args.temperature)) + \
                   kl_div(input=self.softmax(v_observation[0].detach() / self.args.temperature),
                          target=self.softmax(v_representation[0] / self.args.temperature))

        vcd_loss = 0.5 * kl_div(input=self.softmax(v_ms_observation[0].detach()),
                                target=self.softmax(i_ms_representation[0])) + \
                   0.5 * kl_div(input=self.softmax(i_ms_observation[0].detach()),
                                target=self.softmax(v_ms_representation[0]))

        # mutual information estimation for modal-specific branches and modal-shared branch
        # shuff_order = np.random.permutation(self.args.batch_size)
        # specific_MI_I = self.model.module.IR_MIE(x1=i_observation[1], x2=i_representation[1],
        #                                         x1_shuff=i_observation[1][shuff_order, :])[0].mean()
        # specific_MI_V = self.model.module.IR_MIE(x1=v_observation[1], x2=v_representation[1],
        #                                         x1_shuff=v_observation[1][shuff_order, :])[0].mean()
        # shared_MI_I = self.model.module.shared_MIE(x1=i_ms_observation[1], x2=i_ms_representation[1],
        #                                           x1_shuff=i_ms_observation[1][shuff_order, :])[0].mean()
        # shared_MI_V = self.model.module.shared_MIE(x1=v_ms_observation[1], x2=v_ms_representation[1],
        #                                           x1_shuff=v_ms_observation[1][shuff_order, :])[0].mean()

        return ce_loss, triplet_Loss, conventional_ML, vsd_loss, vcd_loss#, associate_loss
