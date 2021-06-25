import math
import random
from bisect import bisect_right
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
############################################################################################
# Channel Compress
############################################################################################

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

class ChannelCompress(nn.Module):
    def __init__(self, in_ch=2048, out_ch=256):
        """
        reduce the amount of channels to prevent final embeddings overwhelming shallow feature maps
        out_ch could be 512, 256, 128
        """
        super(ChannelCompress, self).__init__()
        num_bottleneck = 1000
        add_block = []
        add_block += [nn.Linear(in_ch, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.ReLU()]

        add_block += [nn.Linear(num_bottleneck, 500)]
        add_block += [nn.BatchNorm1d(500)]
        add_block += [nn.ReLU()]
        add_block += [nn.Linear(500, out_ch)]

        # Extra BN layer, need to be removed
        #add_block += [nn.BatchNorm1d(out_ch)]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.model = add_block

    def forward(self, x):
        x = self.model(x)
        return x

############################################################################################
# Classification Loss
############################################################################################
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.0, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


############################################################################################
# gray_scale function
############################################################################################
def to_edge(x):
    x = x.data.cpu()
    out = torch.FloatTensor(x.size(0), x.size(2), x.size(3))
    for i in range(x.size(0)):
        item = x[i,:,:,:]
        #print(item.shape)
        r, g, b = item[0, :, :], item[1, :, :], item[2, :, :]
        xx = 0.2989 * r + 0.5870 * g + 0.1140 * b
        #print(xx.shape)
        out[i, :, :] = xx
    out = out.unsqueeze(1)
    return out.cuda()

############################################################################################
# Random Erasing
############################################################################################
class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

############################################################################################
# Warmup scheduler
############################################################################################
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
############################################################################################
# Rank loss
############################################################################################
class Rank_loss(nn.Module):

    ## Basic idea for cross_modality rank_loss 8

    def __init__(self, margin_1=1.0, margin_2=1.5, alpha_1=2.4, alpha_2=2.2, tval=1.0):
        super(Rank_loss, self).__init__()
        self.margin_1 = margin_1 # for same modality
        self.margin_2 = margin_2 # for different modalities
        self.alpha_1 = alpha_1 # for same modality
        self.alpha_2 = alpha_2 # for different modalities
        self.tval = tval

    def forward(self, x, targets, sub, norm = True):
        if norm:
            #x = self.normalize(x)
            x = torch.nn.functional.normalize(x, dim=1, p=2)
        dist_mat = self.euclidean_dist(x, x) # compute the distance
        loss = self.rank_loss(dist_mat, targets, sub)
        return loss #,dist_mat

    def rank_loss(self, dist, targets, sub):
        loss = 0.0
        for i in range(dist.size(0)):
            is_pos = targets.eq(targets[i])
            is_pos[i] = 0
            is_neg = targets.ne(targets[i])

            intra_modality = sub.eq(sub[i])
            cross_modality = ~ intra_modality

            mask_pos_intra = is_pos* intra_modality
            mask_pos_cross = is_pos* cross_modality
            mask_neg_intra = is_neg* intra_modality
            mask_neg_cross = is_neg* cross_modality

            ap_pos_intra = torch.clamp(torch.add(dist[i][mask_pos_intra], self.margin_1-self.alpha_1),0)
            ap_pos_cross = torch.clamp(torch.add(dist[i][mask_pos_cross], self.margin_2-self.alpha_2),0)

            loss_ap = torch.div(torch.sum(ap_pos_intra), ap_pos_intra.size(0)+1e-5)
            loss_ap += torch.div(torch.sum(ap_pos_cross), ap_pos_cross.size(0)+1e-5)

            dist_an_intra = dist[i][mask_neg_intra]
            dist_an_cross = dist[i][mask_neg_cross]

            an_less_intra = dist_an_intra[torch.lt(dist[i][mask_neg_intra], self.alpha_1)]
            an_less_cross = dist_an_cross[torch.lt(dist[i][mask_neg_cross], self.alpha_2)]

            an_weight_intra = torch.exp(self.tval*(-1* an_less_intra +self.alpha_1))
            an_weight_intra_sum = torch.sum(an_weight_intra)+1e-5
            an_weight_cross = torch.exp(self.tval*(-1* an_less_cross +self.alpha_2))
            an_weight_cross_sum = torch.sum(an_weight_cross)+1e-5
            an_sum_intra = torch.sum(torch.mul(self.alpha_1-an_less_intra,an_weight_intra))
            an_sum_cross = torch.sum(torch.mul(self.alpha_2-an_less_cross,an_weight_cross))

            loss_an =torch.div(an_sum_intra,an_weight_intra_sum ) +torch.div(an_sum_cross, an_weight_cross_sum)
            #loss_an = torch.div(an_sum_cross,an_weight_cross_sum )
            loss += loss_ap + loss_an
            #loss += loss_an
        return loss * 1.0/ dist.size(0)

    def normalize(self, x, axis=-1):
        x = 1.* x /(torch.norm(x, 2, axis, keepdim = True).expand_as(x)+ 1e-12)
        return x

    def euclidean_dist(self, x, y):
        m, n =x.size(0), y.size(0)

        xx = torch.pow(x,2).sum(1, keepdim= True).expand(m,n)
        yy = torch.pow(y,2).sum(1, keepdim= True).expand(n,m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min =1e-12).sqrt()

        return dist

############################################################################################
# Associate Loss
############################################################################################
class ASS_loss(nn.Module):
    def __init__(self, walker_loss=1.0, visit_loss=1.0):
        super(ASS_loss, self).__init__()
        self.walker_loss = walker_loss
        self.visit_loss = visit_loss
        self.ce = nn.CrossEntropyLoss()
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, feature, targets, sub):
        ## normalize
        feature = torch.nn.functional.normalize(feature, dim=1, p=2)
        loss = 0.0
        for i in range(feature.size(0)):
            cross_modality = sub.ne(sub[i])

            p_logit_ab, v_loss_ab = self.probablity(feature, cross_modality,  targets)
            p_logit_ba, v_loss_ba = self.probablity(feature, ~cross_modality, targets)
            n1 = targets[cross_modality].size(0)
            n2 = targets[~cross_modality].size(0)

            is_pos_ab = targets[cross_modality].expand(n1,n1).eq(targets[cross_modality].expand(n1,n1).t())

            p_target_ab = is_pos_ab.float()/torch.sum(is_pos_ab, dim=1).float().expand_as(is_pos_ab)

            is_pos_ba = targets[~cross_modality].expand(n2,n2).eq(targets[cross_modality].expand(n2,n2).t())
            p_target_ba = is_pos_ba.float()/torch.sum(is_pos_ba, dim=1).float().expand_as(is_pos_ba)

            p_logit_ab = self.logsoftmax(p_logit_ab)
            p_logit_ba = self.logsoftmax(p_logit_ba)

            loss += (- p_target_ab * p_logit_ab).mean(0).sum()+ (- p_target_ba * p_logit_ba).mean(0).sum()

            loss += 1.0*(v_loss_ab+v_loss_ba)

        return loss/feature.size(0)/4

    def probablity(self, feature, cross_modality, target):
        a = feature[cross_modality]
        b = feature[~cross_modality]

        match_ab = a.mm(b.t())

        p_ab = F.softmax(match_ab, dim=-1)
        p_ba = F.softmax(match_ab, dim=-1)
        p_aba = torch.log(1e-8+p_ab.mm(p_ba))

        visit_loss = self.new_visit(p_ab, target, cross_modality)

        return p_aba, visit_loss

    def new_visit(self, p_ab, target, cross_modality):
        p_ab = torch.log(1e-8 +p_ab)
        visit_probability = p_ab.mean(dim=0).expand_as(p_ab)
        n1 = target[cross_modality].size(0)
        n2 = target[~cross_modality].size(0)
        p_target_ab = target[cross_modality].expand(n1,n1).eq(target[~cross_modality].expand(n2,n2))
        p_target_ab = p_target_ab.float()/torch.sum(p_target_ab, dim=1).float().expand_as(p_target_ab)
        loss = (- p_target_ab * visit_probability).mean(0).sum()
        return loss

    def normalize(self, x, axis=-1):
        x = 1.* x /(torch.norm(x, 2, axis, keepdim = True).expand_as(x)+ 1e-12)
        return x

############################################################################################
# Wasserstein Distance
############################################################################################
class SinkhornDistance(nn.Module):
    def __init__(self, eps=0.01, max_iter=100, reduction='mean'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        C = C.cuda()
        n_points = x.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, n_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / n_points).squeeze()
        nu = torch.empty(batch_size, n_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / n_points).squeeze()

        u = torch.zeros_like(mu)
        u = u.cuda()
        v = torch.zeros_like(nu)
        v = v.cuda()
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu + 1e-8).cuda() - self.lse(self.M(C, u, v))).cuda() + u
            v = self.eps * (torch.log(nu + 1e-8).cuda() - self.lse(self.M(C, u, v).transpose(-2, -1))).cuda() + v
            err = (u - u1).abs().sum(-1).mean()
            err = err.cuda()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def lse(A):
        "log-sum-exp"
        # add 10^-6 to prevent NaN
        result = torch.log(torch.exp(A).sum(-1) + 1e-6)
        return result

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1