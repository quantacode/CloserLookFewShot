import torch
import torch.nn as nn
import numpy as np
import os
import backbone
from tensorboardX import SummaryWriter
import configs
from io_utils import model_dict, parse_args, get_resume_file
from data.datamgr import SimpleDataManager, SetDataManager
from utils import load_model
import ipdb

DEVICE = torch.device("cuda")
loss_fn = nn.CrossEntropyLoss()
#######################################################################
class Classifier(nn.Module):
    def __init__(self, n_way, params):
        super(Classifier, self).__init__()
        c
        # # cross
        # self.out = nn.Linear(512, n_way)

		# cross_char
        self.out = nn.Linear(64, n_way)

    def forward(self, x):
	    feature = self.feat_extr(x)
	    out = self.out(feature)
	    return out, feature

def forward_loss(x, y, model):
	scores, feat = model.forward(x)
	return loss_fn(scores, y)

def train_loop(epoch, train_loader, model, optimizer, writer):
	print_freq = 10
	avg_loss = 0

	for i, (x, y) in enumerate(train_loader):
		x, y = x.to(DEVICE), y.long().to(DEVICE)
		optimizer.zero_grad()
		loss = forward_loss(x, y, model)
		loss.backward()
		optimizer.step()

		avg_loss = avg_loss + loss.item()

		if i % print_freq == 0:
			# print(optimizer.state_dict()['param_groups'][0]['lr'])
			print(
				'Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss / float(i + 1)))
	# log
	writer.add_scalar('train/loss', avg_loss / float(i + 1), epoch)


def DBindex(cl_data_file):
	# For the definition Davis Bouldin index (DBindex), see https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index
	# DB index present the intra-class variation of the data
	# As baseline/baseline++ do not train few-shot classifier in training, this is an alternative metric to evaluate the validation set
	# Emperically, this only works for CUB dataset but not for miniImagenet dataset

	class_list = cl_data_file.keys()
	cl_num = len(class_list)
	cl_means = []
	stds = []
	DBs = []
	for cl in class_list:
		cl_means.append(np.mean(cl_data_file[cl], axis=0))
		stds.append(np.sqrt(np.mean(np.sum(np.square(cl_data_file[cl] - cl_means[-1]), axis=1))))

	mu_i = np.tile(np.expand_dims(np.array(cl_means), axis=0), (len(class_list), 1, 1))
	mu_j = np.transpose(mu_i, (1, 0, 2))
	mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis=2))

	for i in range(cl_num):
		DBs.append(np.max([(stds[i] + stds[j]) / mdists[i, j] for j in range(cl_num) if j != i]))
	return np.mean(DBs)



def analysis_loop(val_loader, model, writer, epoch, record=None):
	class_file = {}
	for i, (x, y) in enumerate(val_loader):
		x_var =  x.to(DEVICE)
		_, feats = model.forward(x_var)
		feats = feats.cpu().detach().numpy()
		# labels = y.cpu().numpy()
		labels = y.numpy()
		for f, l in zip(feats, labels):
			if l not in class_file.keys():
				class_file[l] = []
			class_file[l].append(f)

	for cl in class_file:
		class_file[cl] = np.array(class_file[cl])

	DB = DBindex(class_file)
	print('DB index = %4.2f' % (DB))
	return 1 / DB  # DB index: the lower the better

def test_loop(val_loader, model, writer, epoch):
	return analysis_loop(val_loader, model,  writer, epoch)

#######################################################################


if __name__=="__main__":
	np.random.seed(10)
	params = parse_args('train')
	if params.dataset == 'CUB':
		base_file = configs.data_dir['CUB'] + 'base_train.json'
		val_file = configs.data_dir['CUB'] + 'base_val.json'
		num_classes = 100
		image_size = 224
	elif params.dataset == 'emnist':
		base_file = configs.data_dir['emnist'] + 'novel_train.json'
		val_file = configs.data_dir['emnist'] + 'novel_val.json'
		num_classes = 31
		image_size = 28
	base_datamgr = SimpleDataManager(image_size, batch_size=16)
	base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
	val_datamgr = SimpleDataManager(image_size, batch_size=64)
	val_loader = val_datamgr.get_data_loader(val_file, aug=False)
	model = Classifier(n_way=num_classes, params=params)
	if params.load_modelpth:
		model = load_model(model, params.load_modelpth)
	model = model.cuda()
	optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
	params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
	if params.train_aug:
		params.checkpoint_dir += '_aug'
	params.checkpoint_dir += '/%s' % (params.exp_id)
	if not os.path.isdir(params.checkpoint_dir):
		os.makedirs(params.checkpoint_dir)
	logdir = params.checkpoint_dir.replace('checkpoints', 'logs')
	if not os.path.isdir(logdir):
		os.makedirs(logdir)
	writer = SummaryWriter(logdir)
	start_epoch = params.start_epoch
	stop_epoch = params.stop_epoch
	#training
	max_acc=0
	for epoch in range(start_epoch, stop_epoch):
		model.train()
		train_loop(epoch, base_loader, model, optimizer, writer)
		model.eval()
		acc = test_loop(val_loader, model, writer, epoch)
		writer.add_scalar('test/DB_index', acc, epoch)

		if acc > max_acc:  # for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
			print("best model! save...")
			max_acc = acc
			outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
			torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

		if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
			outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
			torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
