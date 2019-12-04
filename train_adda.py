import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
import ipdb

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from tensorboardX import SummaryWriter
from io_utils import model_dict, parse_args, get_resume_file
from utils import load_model, load_pretrImagenet

DEVICE1 = torch.device("cuda:0")
DEVICE2 = torch.device("cuda:1")

def train(base_loader, val_loader, featexS, modelT, optimization, start_epoch, stop_epoch, params, writer):

    if optimization == 'Adam':
        optimizerD = torch.optim.Adam(modelT.discriminator.parameters(), lr=params.lr)
        optimizerG = torch.optim.Adam(modelT.feature.parameters(), lr=params.lr)
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0

    for epoch in range(start_epoch,stop_epoch):
        modelT.train()
        modelT.train_loop_ADDA(epoch, base_loader, featexS, optimizerD, optimizerG, writer)

        modelT.eval()
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        acc = modelT.test_loop( val_loader, writer, epoch, params=params)
        if acc > max_acc : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':modelT.state_dict()}, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':modelT.state_dict()}, outfile)


if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')
    if params.adaptFinetune:
        assert (not params.adversarial)

    if params.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json'
        val_file = configs.data_dir['CUB'] + 'val.json'
        novel_file  = configs.data_dir['CUB'] + 'base' +'.json'
    elif params.dataset == 'cross_char':
        base_file = configs.data_dir['omniglot'] + 'noLatin.json'
        val_file   = configs.data_dir['emnist'] + 'val.json'
        novel_file  = configs.data_dir['emnist'] + 'novel' +'.json'
    else:
        base_file = configs.data_dir[params.dataset] + 'base.json'
        val_file   = configs.data_dir[params.dataset] + 'val.json'
        if params.adversarial:
            novel_file  = configs.data_dir[params.dataset] + 'novel.json'

    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char', 'flowers', 'flowers_CUB']:
            image_size = 28
        elif params.dataset in ['CUB_flowers']:
            image_size = 48
        else:
            image_size = 84
    else:
        image_size = 224

    optimization = 'Adam'

    n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot)
    base_datamgr            = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
    base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
    if params.n_shot_test == -1: # modify val loader support
        params.n_shot_test =  params.n_shot
    else: # modify target loader support
        train_few_shot_params['n_support'] = params.n_shot_test
    target_datamgr = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
    target_loader = target_datamgr.get_data_loader(novel_file , aug = False)
    base_loader = [base_loader, target_loader]


    test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot_test)
    val_datamgr             = SetDataManager(image_size, n_query = n_query, **test_few_shot_params)
    val_loader              = val_datamgr.get_data_loader( val_file, aug = False)

    if params.method == 'protonet':
        modelT = ProtoNet(model_dict[params.model], params.test_n_way, params.n_shot, discriminator = backbone.Disc_model(params.train_n_way))
        modelS = ProtoNet(model_dict[params.model], params.test_n_way, params.n_shot)
    # pre_train or warm start
    if params.load_modelpth:
        modelS = load_model(modelS, params.load_modelpth)
        modelT = load_model(modelT, params.load_modelpth)
        print('preloading: ', params.load_modelpth)

    featexS = modelS.feature.to(DEVICE1)
    modelT = modelT.to(DEVICE2)

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if not params.method  in ['baseline', 'baseline++']:
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    params.checkpoint_dir += '/%s' %(params.exp_id)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    logdir = params.checkpoint_dir.replace('checkpoints', 'logs')
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(logdir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.method == 'maml' or params.method == 'maml_approx' :
        stop_epoch = params.stop_epoch * model.n_task #maml use multiple tasks in one update 

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model_state_dict = model.state_dict()
            pretr_dict = {k: v for k, v in tmp.items() if k in model_state_dict}
            model_state_dict.update(pretr_dict)
            model.load_state_dict(model_state_dict)
            # model.load_state_dict(tmp['state'])
    elif params.warmup: #We also support warmup from pretrained baseline feature, but we never used in our paper
        baseline_checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, 'baseline')
        if params.train_aug:
            baseline_checkpoint_dir += '_aug'
        warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
        tmp = torch.load(warmup_resume_file)
        if tmp is not None:
            state = tmp['state']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.feature.load_state_dict(state)
        else:
            raise ValueError('No warm_up file')

    train(base_loader, val_loader,  featexS, modelT, optimization, start_epoch, stop_epoch, params, writer)
