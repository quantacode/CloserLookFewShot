import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from PIL import Image
import os
import ipdb
import glob
import random
import time
import cv2
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

def feature_evaluation(cl_data_file, model, n_way = 5, n_support = 5, n_query = 15, supp=None,
                       query=None, select_class=None):
    z_all  = []
    if query==None:
        imgID_perCls_supp = []
        imgID_perCls_query = []
    else:
        imgID_perCls_supp = supp.copy()
        imgID_perCls_query = query.copy()
    for cli, cl in enumerate(select_class):
        img_feat = cl_data_file[cl]
        if query==None:
            perm_ids = np.random.permutation(len(img_feat)).tolist()
            imgID_perCls_supp.append(perm_ids[:n_support])
            imgID_perCls_query.append(perm_ids[n_support:n_support+n_query])
        else:
            perm_ids = []
            perm_ids.extend(imgID_perCls_supp[cli].copy())
            perm_ids.extend(imgID_perCls_query[cli].copy())

        z_all.append( [ np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support+n_query) ] )     # stack each batch

    z_all = torch.from_numpy(np.array(z_all) )
   
    model.n_query = n_query
    scores  = model.set_forward(z_all, is_feature = True)
    pred = scores.data.cpu().numpy().argmax(axis = 1)
    y = np.repeat(range( n_way ), n_query )
    acc = np.mean(pred == y)*100
    if query==None:
        return acc, imgID_perCls_supp,imgID_perCls_query , select_class
    else:
        return acc, supp, query , select_class


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
        #classes = [1, 20, 40, 60, 80]#1
        # classes = [1, 10, 30, 50, 70]#2
        # classes = [1, 20, 50, 80, 97]#3
        classes = [1, 23, 53, 83, 93]#4
        data = json.load(open('/home/rajshekd/projects/FSG/PRODA/filelists/vgg_flower/all.json'))
        imlist = []


        # adv
        novel_file = "/home/rajshekd/projects/FSG/PRODA/features/CUB_flowers/ResNet10_protonet_aug_5way_5shot/adversarial-ConcatZ_domainReg-0.1_lr-0.0001_DiscM-4096_Base2Base/all.hdf5"
        cl_data_file1 = feat_loader.init_loader(novel_file)
        acc1_avg = []
        for _ in range(600):
            acc1, supp, query, classes = feature_evaluation(cl_data_file1, model, select_class=classes)
            acc1_avg.append(acc1)
        acc1_avg = np.mean(acc1_avg)
        print ("accuracy: ",acc1_avg)
        # vanila
        novel_file = "/home/rajshekd/projects/FSG/PRODA/features/CUB_flowers/ResNet10_protonet_aug_5way_5shot/vanila/all.hdf5"
        cl_data_file2 = feat_loader.init_loader(novel_file)
        # ipdb.set_trace()
        acc2_avg = []
        for _ in range(600):
            acc2, _, _, classes = feature_evaluation(cl_data_file2, model, select_class=classes)
            acc2_avg.append(acc2)
        acc2_avg = np.mean(acc2_avg)
        print("accuracy: ", acc2_avg)

        # acc2, _, _, classes = feature_evaluation(cl_data_file2, model, supp=supp,
        #                                                query=query, select_class=classes)

        # visualize
        for clsid, suppset in enumerate(supp):
            cls = classes[clsid]
            indices = [ind for ind,p in enumerate(data['image_labels']) if p == cls]
            images = [cv2.imread(data['image_names'][indices[si]]) for si in suppset]
            images = [Image.fromarray(np.uint8(img)).resize((100,100)) for img in images]
            images = [transforms.ToTensor()(img).unsqueeze(0) for img in images]
            imlist.extend(images)
        save_image(torch.cat(imlist), '/home/rajshekd/projects/FSG/PRODA/record/images/support_' + str(acc1_avg)+'_'+str(
            acc2_avg)+'.jpg',nrow=5,normalize=True)


