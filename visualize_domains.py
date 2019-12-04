import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
import os
import glob
import random
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import configs
import backbone
import data.feature_loader as feat_loader
from data.datamgr import SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file, get_best_file , get_assigned_file

def feature_evaluation(cl_data_file, model, n_way = 5, n_support = 5, n_query = 15, all_perm_ids = None, select_class =
None):
    class_list = cl_data_file.keys()

    if select_class is None:
        select_class = random.sample(class_list,n_way)
    if all_perm_ids is None:
        perm_flag = 0
        all_perm_ids = []
    else:
        perm_flag=1
    z_all  = []
    for cli, cl in enumerate(select_class):
        img_feat = cl_data_file[cl]
        if not perm_flag:
            perm_ids = np.random.permutation(len(img_feat)).tolist()
            all_perm_ids.append(perm_ids)
        else:
            perm_ids = all_perm_ids[cli]
        z_all.append( [ np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support+n_query) ] )     # stack each batch

    z_all = torch.from_numpy(np.array(z_all) )
    z_proto = z_all[:, :n_support, :].mean(1)#.reshape(-1, z_all.shape[-1])
    z_set = z_proto.view(-1)
    z_set = z_set.cpu().detach().numpy()
    return z_set, select_class,  all_perm_ids

if __name__ == '__main__':
    params = parse_args('test')

    acc_all = []

    iter_num = 600

    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)

    # if params.dataset in ['omniglot', 'cross_char']:
    #     assert params.model == 'Conv4' and not params.train_aug ,'omniglot only support Conv4 without augmentation'
    #     params.model = 'Conv4S'

    if params.method == 'baseline':
        model           = BaselineFinetune( model_dict[params.model], **few_shot_params )
    elif params.method == 'baseline++':
        model           = BaselineFinetune( model_dict[params.model], loss_type = 'dist', **few_shot_params )
    elif params.method == 'protonet':
        model           = ProtoNet( model_dict[params.model], **few_shot_params )
    elif params.method == 'matchingnet':
        model           = MatchingNet( model_dict[params.model], **few_shot_params )
    elif params.method in ['relationnet', 'relationnet_softmax']:
        if params.model == 'Conv4':
            feature_model = backbone.Conv4NP
        elif params.model == 'Conv6':
            feature_model = backbone.Conv6NP
        elif params.model == 'Conv4S':
            feature_model = backbone.Conv4SNP
        else:
            feature_model = lambda: model_dict[params.model]( flatten = False )
        loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
        model           = RelationNet( feature_model, loss_type = loss_type , **few_shot_params )
    elif params.method in ['maml' , 'maml_approx']:
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model = MAML(  model_dict[params.model], approx = (params.method == 'maml_approx') , **few_shot_params )
        if params.dataset in ['omniglot', 'cross_char']: #maml use different parameter in omniglot
            model.n_task     = 32
            model.task_update_num = 1
            model.train_lr = 0.1
    else:
       raise ValueError('Unknown method')

    model = model.cuda()

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++'] :
        checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
    checkpoint_dir += '/%s' % (params.exp_id)

    #modelfile   = get_resume_file(checkpoint_dir)

    if not params.method in ['baseline', 'baseline++'] :
        if params.save_iter != -1:
            modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
        else:
            modelfile   = get_best_file(checkpoint_dir)
        if modelfile is not None:
            model_state_dict = model.state_dict()
            pretr_dict = torch.load(modelfile)
            pretr_dict = {k: v for k, v in pretr_dict.items() if k in model_state_dict}
            model_state_dict.update(pretr_dict)
            model.load_state_dict(model_state_dict)

    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" +str(params.save_iter)
    else:
        split_str = split
    if params.method in ['maml', 'maml_approx']: #maml do not support testing with feature
        if 'Conv' in params.model:
            if params.dataset in ['omniglot', 'cross_char']:
                image_size = 28
            else:
                image_size = 84
        else:
            image_size = 224

        datamgr         = SetDataManager(image_size, n_eposide = iter_num, n_query = 15 , **few_shot_params)

        if params.dataset == 'cross':
            if split == 'base':
                loadfile = configs.data_dir['miniImagenet'] + 'all.json'
            else:
                loadfile   = configs.data_dir['CUB'] + split +'.json'
        elif params.dataset == 'cross_char':
            if split == 'base':
                loadfile = configs.data_dir['omniglot'] + 'noLatin.json'
            else:
                loadfile  = configs.data_dir['emnist'] + split +'.json'
        else:
            loadfile    = configs.data_dir[params.dataset] + split + '.json'

        novel_loader     = datamgr.get_data_loader( loadfile, aug = False)
        if params.adaptation:
            model.task_update_num = 100 #We perform adaptation on MAML simply by updating more times.
        model.eval()
        acc_mean, acc_std = model.test_loop( novel_loader, return_std = True)

    else:
        rootdir = '/home/rajshekd/projects/FSG/PRODA/features'
        NovelF = ['omniglot/Conv4S_protonet_5way_5shot/vanila/noLatin.hdf5',
                  'omniglot/Conv4S_protonet_5way_5shot/vanila/novel.hdf5',
                  'cross_char/Conv4S_protonet_5way_5shot/vanila-Protonet/novel.hdf5']
        all_z = []
        for novel_file in NovelF:
            all_perm_ids_iter = []
            select_class_iter = []
            print(novel_file)
            # import ipdb
            # ipdb.set_trace()
            cl_data_file = feat_loader.init_loader(os.path.join(rootdir,novel_file))
            for i in range(iter_num):
                z_set, select_class, all_perm_ids = feature_evaluation(cl_data_file, model, n_query = 15)
                all_perm_ids_iter.append(all_perm_ids)
                select_class_iter.append(select_class)
                all_z.append(z_set)
        # final file
        cl_data_file = feat_loader.init_loader(
            os.path.join(
                rootdir, 'cross_char/Conv4S_protonet_5way_5shot/adversarial-concatZ_domainReg-0.1_lr-0.0001_endEpoch-4000_DiscM-2FC512/novel.hdf5'))
        for i in range(iter_num):
            z_set, select_class, all_perm_ids = feature_evaluation(cl_data_file, model, n_query=15, all_perm_ids =
            all_perm_ids_iter[i], select_class = select_class_iter[i])
            all_z.append(z_set)

        all_z = np.vstack(all_z)
        print('PCA...')
        pca = PCA(n_components=50).fit_transform(all_z)
        print('TSNE...')
        tsne = TSNE(n_components=2).fit_transform(pca)

        # visualize
        fig, ax = plt.subplots()
        colors = ['gray', 'black', 'blue', 'green']
        for i, color in enumerate(['base train', 'base test', 'novel', 'novel adapt']):
            scat = ax.scatter(tsne[iter_num*i:iter_num*(i+1), 0], tsne[iter_num*i:iter_num*(i+1), 1], c=colors[i],
                              label=color, alpha=0.5)
        ax.legend()

        # color = np.ones((3,600,1))
        # scat = ax.scatter(tsne[:, 0], tsne[:, 1], c=color.squeeze(), label=[0, 1, 2], alpha=0.5)
        # # scat = ax.scatter(tsne[:, 0], tsne[:, 1])
        # legend1 = ax.legend(*scat.legend_elements())
        # ax.add_artist(legend1)
        ax.set_title('CUB (base) -> VGF (novel)')
        plt.savefig(params.dataset+'.jpg')

