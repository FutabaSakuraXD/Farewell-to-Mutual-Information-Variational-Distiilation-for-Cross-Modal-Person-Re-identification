from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os
import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from reid import datasets
from reid.dist_metric import DistanceMetric
from reid.models import ft_net
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import CamRandomIdentitySampler as RandomIdentitySampler
from reid.utils.data.sampler import CamSampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from utlis import RandomErasing, WarmupMultiStepLR, CrossEntropyLabelSmooth, Rank_loss, ASS_loss


def get_data(name, split_id, data_dir, height, width, batch_size, num_instances,
             workers, combine_trainval, flip_prob, padding, re_prob, using_HuaWeiCloud, cloud_dataset_root):
    root = osp.join(data_dir, name)

    if using_HuaWeiCloud: root = cloud_dataset_root

    print(root)

    dataset = datasets.create(name, root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    trainvallabel = dataset.trainvallabel
    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)

    train_transformer = T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(p=flip_prob),
        T.Pad(padding),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        RandomErasing(probability=re_prob, mean=[0.485, 0.456, 0.406])
        ])

    test_transformer = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalizer,
    ])

    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=32, num_workers=workers,
        shuffle=False, pin_memory=True)

    query_loader = DataLoader(
        Preprocessor(list(set(dataset.query)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=32, num_workers=workers,
        sampler=CamSampler(list(set(dataset.query)), [2,5]),
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(list(set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=32, num_workers=workers,
        sampler=CamSampler(list(set(dataset.gallery)), [0,1,3,4], 4),
        shuffle=False, pin_memory=True)

    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
            transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=RandomIdentitySampler(train_set, num_instances),
        pin_memory=True, drop_last=True)

    return dataset, num_classes, train_loader, trainvallabel, val_loader, query_loader, gallery_loader

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir+'/log'))

    if args.height is None or args.width is None: args.height, args.width = (256, 128)

    # Dataset and loader
    dataset, num_classes, train_loader, trainvallabel, val_loader, query_loader, gallery_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
            args.width, args.batch_size, args.num_instances, args.workers,
            args.combine_trainval, args.flip_prob, args.padding, args.re_prob,
            args.HUAWEI_cloud, args.dataset_root)

    # Model settings
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ft_net(args=args, num_classes=num_classes, num_features=args.features)
    model = nn.DataParallel(model, device_ids=[0,1])
    model = model.to(device)

    # Evaluation components
    evaluator = Evaluator(model)
    metric = DistanceMetric(algorithm=args.dist_metric)

    start_epoch = 0
    if args.resume:
        #########################################
        checkpoint = load_checkpoint(args.resume)
        state_dict = checkpoint['model']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' not in k:
                k = 'module.' + k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        #########################################
        start_epoch = checkpoint['epoch']
        print("=> Start epoch {}".format(start_epoch))

    if args.evaluate:
        metric.train(model, train_loader)
        evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery)
        exit()

    # Losses
    ce_Loss = CrossEntropyLabelSmooth(num_classes= num_classes, epsilon=args.epsilon).cuda()
    associate_loss = ASS_loss().cuda()
    rank_Loss = Rank_loss(margin_1= args.margin_1, margin_2 =args.margin_2, alpha_1 =args.alpha_1, alpha_2= args.alpha_2).cuda()

    print(args)
    # optimizers and schedulers
    conv_optim = model.module.optims()

    conv_scheduler = WarmupMultiStepLR(conv_optim, args.mile_stone, args.gamma, args.warmup_factor,
                                          args.warmup_iters, args.warmup_methods)

    trainer = Trainer(args, model, ce_Loss, rank_Loss, associate_loss, trainvallabel)

    best_top1 = -1

    # Start training
    for epoch in range(start_epoch, args.epochs):
        conv_scheduler.step()

        triple_loss, tot_loss = trainer.train(epoch, train_loader, conv_optim)

        save_checkpoint({
            'model': model.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, False, epoch, args.logs_dir, fpath='checkpoint.pth.tar')

        if epoch < args.begin_test:
            continue
        if not epoch % args.evaluate_freq == 0:
            continue

        top1 = evaluator.evaluate(query_loader, gallery_loader, metric)

        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint({
            'model': model.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, epoch, args.logs_dir, fpath='checkpoint.pth.tar')

    print('Test with best model:')
    print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
              format(epoch, top1, best_top1, ' *' if is_best else ''))
    print(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cross_modality for Person Re-identification")

    # dataset settings
    parser.add_argument('-d', '--dataset', type=str, default='sysu', choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)

    # transformer
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default= 128)
    parser.add_argument('--flip_prob', type=float, default=0.5)
    parser.add_argument('--re_prob', type=float, default=0.0)
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--combine-trainval', default=True, action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--num-instances', type=int, default=8,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # model
    parser.add_argument('--features', type=int, default=2048)

    # rank loss settings
    parser.add_argument('--margin_1', type=float, default=0.9, help="margin_1 of the triplet loss, default: 0.9")
    parser.add_argument('--margin_2', type=float, default=1.0, help="margin_1 of the triplet loss, default: 1.5")
    parser.add_argument('--alpha_1', type=float, default=2.2,  help="alpha_1 of the triplet loss, default: 2.4")
    parser.add_argument('--alpha_2', type=float, default=2.0,  help="alpha_2 of the triplet loss, default: 2.2")

    # optimizer and scheduler
    parser.add_argument('--lr', type=float, default=2.6e-4, help="learning rate of all parameters")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--use_adam', action='store_true', help="use Adam as the optimizer, elsewise SGD ")
    parser.add_argument('--gamma', type=float, default = 0.1, help="gamma for learning rate decay")

    parser.add_argument('--mile_stone', type=list, default=[210])

    parser.add_argument('--warmup_iters', type=int, default=10)
    parser.add_argument('--warmup_methods', type=str, default = 'linear', choices=('linear', 'constant'))
    parser.add_argument('--warmup_factor', type=float, default = 0.01 )

    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='')
    parser.add_argument('--evaluate',action='store_true',
                        help="this option meaningless "
                             "since it is required to conduct evaluation on officially approved codes")

    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--start_save', type=int, default=100, help="start saving checkpoints after specific epoch")
    parser.add_argument('--begin_test', type=int, default=100)
    parser.add_argument('--evaluate_freq', type=int, default=5)

    parser.add_argument('--seed', type=int, default=1)

    # adopted metric
    parser.add_argument('--dist-metric', type=str, default='euclidean')

    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'Exp_before_OpenSource_7_1'))
    # hyper-parameters
    parser.add_argument('-CE_loss', type=int, default=1, help="weight of cross entropy loss")
    parser.add_argument('-epsilon', type=float, default=0.1, help="label smooth")
    parser.add_argument('-Triplet_loss', type=int, default=1, help="weight of triplet loss")
    parser.add_argument('-Associate_loss', type=float, default=0.0, help="weight of loss")

    parser.add_argument('-CML_loss', type=int, default=8, help="the weight of conventional mutual learning")
    parser.add_argument('-VCD_loss', type=int, default=2, help="the weight of VCD and VML")
    parser.add_argument('-VSD_loss', type=float, default=2, help="weight of VSD")
    parser.add_argument('-temperature', type=int, default=1, help="the temperature used in knowledge distillation")

    # Bottleneck
    parser.add_argument('-z_dim', type=int, default=256, help="dimension of latent z, better set to {128, 256, 512}")
    # device set
    parser.add_argument('--visible_device', default='2, 1', type=str, help='gpu_ids: e.g. 0, 0,1,2  0,2')

    # HUAWEI cloud
    parser.add_argument('--HUAWEI_cloud', type=bool, default=False)
    parser.add_argument('--dataset_root', type=str, metavar='PATH', default="/test-ddag/dataset")
    parser.add_argument('--data_url', type=str, default="")
    parser.add_argument('--init_method', type=str, default="")
    parser.add_argument('--train_url', type=str, default="")

    main(parser.parse_args())
