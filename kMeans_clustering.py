import torch
import torch.nn as nn
import numpy as np
import os
import backbone
from tensorboardX import SummaryWriter
import configs
from io_utils import model_dict, parse_args, get_resume_file
from data.datamgr import NamedDataManager
from utils import load_pretrImagenet
import torchvision.models as models
import ipdb
import random
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda")
loss_fn = nn.CrossEntropyLoss()

def get_clsID(feat, Ck):
	feat = np.array([feat]).repeat(Ck.shape[0], axis=0)
	distances = np.sum((feat-Ck)**2, axis=1)
	return np.argmin(distances)

def update_clusterC(data):
	clsIDs = data['cluster_label']
	uniqueID = np.unique(clsIDs)
	Ck = []

	for uID in uniqueID:
		uind = np.argwhere(clsIDs==uID)
		ufeat = [ele for i, ele in enumerate(data['image_feat']) if i in uind]
		Ck.append(list(np.mean(ufeat, axis=0)))
		# print(uID, uind.shape)
	return np.array(Ck)

def initialize_clusters(data, n_cluster , type='random'):
	# type:
	# 'random': randomly initialize n_cluster cluster centers
	# 'k-shot': select n_cluster clusters according to the gt. This is a semi-supervised clustering
	num_ele = len(data['image_feat'])
	if type=='random':
		clusterID = random.sample(list(np.arange(0, num_ele)), n_cluster)
	elif type=='k-shot':
		uniqueL = np.unique(data['image_label'])
		clusterID = []
		for uL in uniqueL:
			indices = np.argwhere(data['image_label']==uL)
			clusterID.append(np.random.choice(indices.squeeze()))
	return np.array(data['image_feat'])[clusterID]

def cluster_set(data, n_cluster, params):
	Ck = initialize_clusters(data, n_cluster, type=params.supervision)
	cluster_label_prev = None
	iter=0
	while data['cluster_label'] != cluster_label_prev: # check if no change in cluster assignment
		iter+=1
		print('k-means iter: ',iter)
		cluster_label_prev = data['cluster_label'].copy()
		for i in range(len(data['image_feat'])):
			clsID = get_clsID(data['image_feat'][i], Ck)
			data['cluster_label'][i] = clsID
		Ck = update_clusterC(data)
	return data

if __name__=="__main__":
	np.random.seed(10)
	params = parse_args('train')
	if params.dataset == 'CUB':
		num_classes = 100
		image_size = 224
		base_file = configs.data_dir['CUB'] + 'base.json'
	elif params.dataset == 'emnist':
		num_classes = 31
		image_size = 224
		base_file = configs.data_dir['emnist'] + 'novel.json'
	elif params.dataset == 'miniImagenet':
		num_classes = 16
		image_size = 224
		base_file = configs.data_dir['miniImagenet'] + 'val.json'
	base_datamgr = NamedDataManager(image_size, batch_size=128)
	base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
	# load model
	model = model_dict[params.model]()
	model = load_pretrImagenet(model, params.load_modelpth)
	model = model.cuda()
	data = {'image_name': [], 'image_feat': [], 'image_label': [], 'cluster_label': []}
	for i, (name, x,y) in enumerate(base_loader):
		x = x.to(DEVICE)
		feat = model(x).squeeze()
		feat = list(feat.cpu().detach().numpy())
		y = list(y.numpy())
		data['image_name'].extend(name)
		data['image_feat'].extend(feat)
		data['image_label'].extend(y)

	# visualize gt clusters
	print('PCA...')
	pca = PCA(n_components=50).fit_transform(data['image_feat'])
	print('TSNE...')
	tsne = TSNE(n_components=2).fit_transform(pca)
	fig, ax = plt.subplots(1, 2)
	colors = np.random.rand(num_classes)
	ax[0].scatter(tsne[:, 0], tsne[:, 1], c=data['image_label'], alpha=0.5)
	ax[0].set_title('gt_labels')
	plt.savefig(params.dataset + '_clusters.jpg')

	# k-means clustering
	data['cluster_label'] = [-1]*len(data['image_label'])
	data = cluster_set(data, n_cluster=num_classes, params=params)

	# compute accuracy
	correct = 0
	total = 0
	unique_gt_labels = np.unique(data['image_label'])
	for uid in unique_gt_labels:
		uind = np.argwhere(uid==data['image_label'])
		clstr_labels = [ele for i, ele in enumerate(data['cluster_label']) if i in uind]
		mode_label = stats.mode(clstr_labels).mode[0]
		num_mode_label = len([l for l in clstr_labels if l==mode_label])
		num_label = len(clstr_labels)
		correct +=  num_mode_label
		total += num_label
		acc = (num_mode_label*100.0)/num_label
		print("label: {}, accuracy: {}".format(uid, acc))
	total_acc = (100.0 *correct)/total
	print("total accuracy: {}".format(total_acc))

	# visualize k-means clusters
	ax[1].scatter(tsne[:,0], tsne[:,1], c = data['cluster_label'], alpha=0.5)
	ax[1].set_title('kMeans_labels')
	plt.savefig(params.dataset + '_clusters.jpg')




