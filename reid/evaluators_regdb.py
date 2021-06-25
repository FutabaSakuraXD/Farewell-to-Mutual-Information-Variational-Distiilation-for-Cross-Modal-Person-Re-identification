from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch
import numpy as np
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
        outputs = extract_cnn_feature(model, imgs, subs)
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

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.2%}'.format(mAP))

    # return mAP
    if flag:
        return mAP
    # Compute all kinds of CMC scores
    cmc_configs = {
        'RegDB': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores{:>12}'.format('RegDB')
    )
    for k in cmc_topk:
        print('  top-{:<4}{:12.2%}'
              .format(k,cmc_scores['RegDB'][k - 1])
              )

    # Use the allshots cmc top-1 score for validation criterion
    return cmc_scores['RegDB'][0]


def eval_regdb(distmat, labels1, labels2, fnames1, fnames2, max_rank = 20, cmc_topk=(1, 10, 20)):
    q_pids = [labels1[f].numpy()  for f in fnames1]
    g_pids = [labels2[f].numpy() for f in fnames2]
    q_pids= np.array(q_pids)
    g_pids= np.array(g_pids)
    # q_pids = q_pids.numpy()
    # g_pids = g_pids.numpy()
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0. # number of valid query

    # only two cameras
    q_camids = np.ones(num_q).astype(np.int32)
    g_camids = 2* np.ones(num_g).astype(np.int32)

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(raw_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
        all_INP.append(inp)
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    print('Mean AP: {:4.2%}'.format(mAP))
    print('CMC Scores{:>12}'.format('RegDB')
    )
    for k in cmc_topk:
        print('  top-{:<4}{:12.2%}'
              .format(k,all_cmc[k - 1])
              )
    return all_cmc[0], all_cmc, mAP


class Evaluator(object):
    def __init__(self, model, regdb=True):
        super(Evaluator, self).__init__()
        self.model = model
        self.regdb=regdb

    def evaluate(self, data_loader1, data_loader2, metric=None, flag=False):
        features1, labels1, fnames1 = extract_features(model=self.model, data_loader=data_loader1)
        features2, labels2, fnames2 = extract_features(model=self.model, data_loader=data_loader2)
        distmat = pairwise_distance(features1, features2, fnames1, fnames2, metric=metric)

        if self.regdb:
            top1, all_cmc, mAP= eval_regdb(distmat, labels1, labels2, fnames1, fnames2)
            return top1, all_cmc, mAP
        else:
            return evaluate_all(distmat, labels1, labels2, fnames1, fnames2, flag)
