from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch

from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter


def extract_features(model, data_loader):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()
    filenames = []

    end = time.time()
    for i, (imgs, fnames, pids, cams) in enumerate(data_loader):
        data_time.update(time.time() - end)

        subs = ((cams == 2).long() + (cams == 5).long()).cuda()
        outputs = extract_cnn_feature(model=model, inputs=imgs, sub=subs)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid
            filenames.append(fname)

        batch_time.update(time.time() - end)
        end = time.time()


    return features, labels, filenames


def pairwise_distance(features1, features2, fnames1=None, fnames2=None, metric=None):
    x = torch.cat([features1[f].unsqueeze(0) for f in fnames1], 0)
    y = torch.cat([features2[f].unsqueeze(0) for f in fnames2], 0)

    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)

    # normalize
    x = torch.nn.functional.normalize(x, dim=1, p=2)
    y = torch.nn.functional.normalize(y, dim=1, p=2)

    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def evaluate_all(distmat, labels1, labels2, fnames1, fnames2, flag, cmc_topk=(1, 10, 20)):
    query_ids = [labels1[f] for f in fnames1]
    gallery_ids = [labels2[f] for f in fnames2]
    query_cams = [0 for f in fnames1]
    gallery_cams = [2 for f in fnames2]

    """Compute mean AP"""

    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.2%}'.format(mAP))
    if flag: # return mAP only
        return mAP

    """Compute all kinds of CMC scores"""

    cmc_configs = {
        'MM01': dict(separate_camera_set=False,
                      single_gallery_shot=False,
                      first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids, query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}
    # name:MM01
    # params{'separate_camera_set': False, 'single_gallery_shot': False, 'first_match_break': True}
    print('CMC Scores{:>12}'.format('MM01'))
    for k in cmc_topk:
        print('  top-{:<4}{:12.2%}'.format(k, cmc_scores['MM01'][k - 1]))
    # using Rank-1 as the main evaluation criterion
    return cmc_scores['MM01'][0]


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader1, data_loader2, metric=None, flag=False):
        features1, labels1, fnames1 = extract_features(self.model, data_loader1)
        features2, labels2, fnames2 = extract_features(self.model, data_loader2)
        distmat = pairwise_distance(features1, features2, fnames1, fnames2, metric=metric)
        return evaluate_all(distmat, labels1, labels2, fnames1, fnames2, flag)
