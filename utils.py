import torch
import numpy as np

def one_hot(y, num_class):         
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)

def DBindex(cl_data_file):
    class_list = cl_data_file.keys()
    cl_num= len(class_list)
    cl_means = []
    stds = []
    DBs = []
    for cl in class_list:
        cl_means.append( np.mean(cl_data_file[cl], axis = 0) )
        stds.append( np.sqrt(np.mean( np.sum(np.square( cl_data_file[cl] - cl_means[-1]), axis = 1))))

    mu_i = np.tile( np.expand_dims( np.array(cl_means), axis = 0), (len(class_list),1,1) )
    mu_j = np.transpose(mu_i,(1,0,2))
    mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis = 2))
    
    for i in range(cl_num):
        DBs.append( np.max([ (stds[i]+ stds[j])/mdists[i,j]  for j in range(cl_num) if j != i ]) )
    return np.mean(DBs)

def sparsity(cl_data_file):
    class_list = cl_data_file.keys()
    cl_sparsity = []
    for cl in class_list:
        cl_sparsity.append(np.mean([np.sum(x!=0) for x in cl_data_file[cl] ])  ) 

    return np.mean(cl_sparsity) 

def load_model(model, loadpth):
    model_state_dict = model.state_dict()
    pretr_dict = torch.load(loadpth)['state']
    if 'module' in list(pretr_dict.keys())[0]:
        pretr_dict = {k: v for k, v in pretr_dict.items() if k in model_state_dict}
    else:
        pretr_dict = {k.replace('feature.','feature.module.'): v for k, v in pretr_dict.items() if k.replace('feature.','feature.module.') in model_state_dict}
    model_state_dict.update(pretr_dict)
    model.load_state_dict(model_state_dict)
    return model

def load_baselinePP(model, loadpth):
    model_state_dict = model.state_dict()
    pretr_dict = torch.load(loadpth)['state']
    if 'module' in list(model_state_dict.keys())[0]:
        pretr_dict = {k.replace('feature.', 'feature.module.'): v for k, v in pretr_dict.items() if
                      k.replace('feature.', 'feature.module.') in model_state_dict}
    else:
        pretr_dict = {k: v for k, v in pretr_dict.items() if k in model_state_dict}
    model_state_dict.update(pretr_dict)
    model.load_state_dict(model_state_dict)
    return model

def load_pretrImagenet(model, loadpth):
    model_state_dict = model.state_dict()
    pretr_state_dict = torch.load(loadpth)['state']
    keylist_model = list(model_state_dict.keys())
    keylist_pretr = list(pretr_state_dict.keys())
    pretr_dict = {keylist_model[i]: pretr_state_dict[keylist_pretr[i]] for i in range(len(keylist_model))}
    model_state_dict.update(pretr_dict)
    model.load_state_dict(model_state_dict)
    return model

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)