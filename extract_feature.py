from __future__ import print_function, absolute_import
import os

import argparse
import numpy as np
import torch
from torch import nn
import scipy.io as sio
from torch.backends import cudnn

from reid.models.newresnet import *
from reid.utils.serialization import load_checkpoint
import torchvision.transforms as transforms
from PIL import Image
from reid.utils.data import transforms as T


def getArgs():
	parser = argparse.ArgumentParser(description="Cross_modality for Person Re-identification")
	# data
	parser.add_argument('--height', type=int, default=256,
						help="input height, default: 256 for resnet*, "
							 "144 for inception")
	parser.add_argument('--width', type=int, default=128,
						help="input width, default: 128 for resnet*, "
							 "56 for inception")

	parser.add_argument('--features', type=int, default=2048)
	parser.add_argument('--dropout', type=float, default=0)

	# Bottleneck
	parser.add_argument('-z_dim', type=int, default=256,
						help="dimension of latent z, better belongs to {128, 256, 512}")
	# device set
	parser.add_argument('--visible_device', default='1, 0', type=str, help='gpu_ids: e.g. 0, 0,1,2  0,2')
	parser.add_argument('--weight-decay', type=float, default=5e-4)
	return parser.parse_args()

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

args = getArgs()

features = 8192 # 8192 for observation (four in total), 2048 for representation (four in total)
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 0"
model = ft_net(args=args, num_classes=395, num_features=args.features)
model_path = "/home/txd/Variational Distillation/Exp_before_OpenSource_7/"
checkpoint_I = load_checkpoint(model_path + 'model_best.pth.tar')
model = nn.DataParallel(model, device_ids=[0, 1])
cudnn.benchmark = True
#########################################
checkpoint = load_checkpoint(model_path + 'model_best.pth.tar')
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
normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

test_transformer = T.Compose([
	T.Resize((256, 128)),
	T.ToTensor(),
	normalizer,
])

print('model initialed \n')
print("The dimension of the feature is " + str(features))
print("Total params: " + str(count_param(model)))

cam_name = ''

path_list = ['./data/sysu/SYSU-MM01/cam1/','./data/sysu/SYSU-MM01/cam2/','./data/sysu/SYSU-MM01/cam3/', \
             './data/sysu/SYSU-MM01/cam4/','./data/sysu/SYSU-MM01/cam5/','./data/sysu/SYSU-MM01/cam6/']
pic_num = [333,333,533,533,533,333]
for index, path in enumerate(path_list):
	print(index)
	cams = torch.LongTensor([index])
	sub = ((cams == 2).long() + (cams == 5).long()).cuda()
	count = 1
	array_list = []
	person_id_list = []
	dict_person = {}
	tot_num = pic_num[index]
	array_list_to_array = [[] for _ in range(tot_num)]
	#print(path)
	for fpathe,dirs,fs in os.walk(path):
		person_id = fpathe.split('/')[-1]
		if(person_id == ''):
			continue
		cam_name = fpathe[-9:-5]
		fs.sort()
		person_id_list.append(person_id)
		dict_person[person_id] = fs
	person_id_list.sort()
	for person in person_id_list:
		temp_list = []
		for imagename in dict_person[person]:
			filename = path + str(person) + '/' + imagename
			img=Image.open(filename)
			img=test_transformer(img)
			img=img.view([1,3,256,128])
			img=img.cuda()
			i_observation, i_representation, i_ms_observation, i_ms_representation, \
			v_observation, v_representation, v_ms_observation, v_ms_representation = model(img)
			result_y = torch.cat(tensors=[i_observation[1], i_ms_observation[1],
										  v_observation[1], v_ms_observation[1]], dim=1)

			result_y = torch.nn.functional.normalize(result_y, dim=1, p=2)
			result_y = result_y.view(-1, features)
			result_y = result_y.squeeze()
			result_npy=result_y.data.cpu().numpy()
			result_npy=result_npy.astype('double')
			temp_list.append(result_npy)
		temp_array = np.array(temp_list)
		array_list_to_array[int(person)-1]=temp_array
	array_list_to_array = np.array(array_list_to_array)

	sio.savemat(model_path + "observation" + cam_name + '.mat', {'feature_test':array_list_to_array})

