from __future__ import print_function
from torch import nn
import argparse
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from reid.utils.serialization import load_checkpoint
from reid.models import ft_net
from eval_metrics import eval_regdb
from utils import *
from PIL import Image
from reid.utils.data import transforms as T

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='regdb', help='dataset name: regdb or sysu]')

parser.add_argument('--features', type=int, default=2048, help='feature dimensions')
parser.add_argument('--z_dim', type=int, default=256, help='information bottleneck dimensions')

parser.add_argument('--height', type=int, default=256, help='image height, should be chosen in {256, 288, 312}')
parser.add_argument('--width', type=int, default=128, help='image width, should be chosen in {128, 144, 156}')

parser.add_argument('--model_path', default='/home/txd/Variational Distillation/Exp_RegDB/RegDB_2/', type=str)
args = parser.parse_args()

# overall settings
data_path = '../data/RegDB/'
n_class = 206
test_mode = [2, 1]

# model settings
overall_feats = 8192
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 1"
model = ft_net(args=args, num_classes=n_class, num_features=args.features)
model = nn.DataParallel(model, device_ids=[0, 1])
cudnn.benchmark = True

#checkpoint_path = args.model_path

print('==> Loading data..')

# Data loading code
normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

test_transformer = T.Compose([
    #T.Resize((312, 156)),
    T.Resize((args.height, args.width)),
    T.ToTensor(),
    normalizer,
])

if args.dataset == 'regdb':
    data_dir = '../data/RegDB/'

    for trial in range(1,9):
        test_trial = trial +1
        model_path = args.model_path + '{}/model_best.pth.tar'.format(trial)

        #########################################
        checkpoint = load_checkpoint(model_path)
        state_dict = checkpoint['model']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' not in k or 'cam_module' in k:
                k = 'module.' + k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        #########################################
        model = model.cuda()
        model.eval()

        # v to t
        query_list   = data_dir + 'idx/test_visible_{}'.format(test_trial)+ '.txt'
        gallery_list = data_dir + 'idx/test_thermal_{}'.format(test_trial)+ '.txt'
        # t to v
        #query_list   = data_dir + 'idx/test_thermal_{}'.format(test_trial)+ '.txt'
        #gallery_list = data_dir + 'idx/test_visible_{}'.format(test_trial)+ '.txt'

        query_image_list, query_label = load_data(query_list)
        gallery_image_list, gallery_label =load_data(gallery_list)
        temp_list = []
        for index, pth in enumerate(query_image_list):
            filename = data_dir + pth
            img = Image.open(filename)
            img = test_transformer(img)
            img = img.view([1, 3, args.height, args.width])
            img = img.cuda()

            i_observation, i_representation, i_ms_observation, i_ms_representation, \
            v_observation, v_representation, v_ms_observation, v_ms_representation = model(img)
            result_y = torch.cat(tensors=[i_observation[1], i_ms_observation[1],
                                          v_observation[1], v_ms_observation[1]], dim=1)

            result_y = torch.nn.functional.normalize(result_y, dim=1, p=2)
            result_y = result_y.view(-1, overall_feats)
            result_y = result_y.squeeze()
            result_npy = result_y.data.cpu().numpy()
            result_npy = result_npy.astype('double')
            temp_list.append(result_npy)

        query_feat, query_label = np.array(temp_list), np.array(query_label)
        print('Query feature extraction: done')

        temp_list = []
        for index, pth in enumerate(gallery_image_list):
            filename = data_dir + pth
            img=Image.open(filename)
            img=test_transformer(img)
            img=img.view([1, 3, args.height, args.width])
            img=img.cuda()

            i_observation, i_representation, i_ms_observation, i_ms_representation, \
            v_observation, v_representation, v_ms_observation, v_ms_representation = model(img)
            result_y = torch.cat(tensors=[i_observation[1], i_ms_observation[1],
                                          v_observation[1], v_ms_observation[1]], dim=1)

            result_y = torch.nn.functional.normalize(result_y, dim=1, p=2)
            result_y = result_y.view(-1, overall_feats)
            result_y = result_y.squeeze()
            result_npy=result_y.data.cpu().numpy()
            result_npy=result_npy.astype('double')
            temp_list.append(result_npy)

        gallery_feat, gallery_label = np.array(temp_list), np.array(gallery_label)

        print('Gallery feature extraction: done')
        distmat = np.matmul(query_feat, np.transpose(gallery_feat))
        cmc, mAP, mINP = eval_regdb(-distmat, query_label, gallery_label)

        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP

        print(
            'Results:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))


def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label