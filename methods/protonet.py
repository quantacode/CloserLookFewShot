# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
import ipdb

class ProtoNet(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, discriminator=None, cosine=False):
        super(ProtoNet, self).__init__(model_func,  n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.adv_loss_fn = nn.CrossEntropyLoss()
        self.discriminator = discriminator
        self.cosine_dist = cosine
        if self.cosine_dist: self.temperature = 10


    def set_forward(self,x,is_feature = False , is_adversarial=False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support = z_support.contiguous()
        z_proto = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        if self.cosine_dist:
            dists = cosine_dist(z_query, z_proto, self.temperature)
        else:
            dists = euclidean_dist(z_query, z_proto)
        scores = -dists

        if is_adversarial:
            # SeparateZ
            z_set = z_support.view(-1, z_support.shape[-1])
            # # ConcatZ
            # z_set = z_proto.view(-1).unsqueeze(0)
            # # AvgZ
            # z_set = z_proto.mean(dim=0).unsqueeze(0)
            return scores, z_set
        else:
            return scores

    def discriminator_score(self, zS, zT, adv_loss_fn):
        ls = adv_loss_fn(self.discriminator(zS), torch.ones(zS.shape[0], dtype=torch.long).cuda())
        lt = adv_loss_fn(self.discriminator(zT), torch.zeros(zT.shape[0], dtype=torch.long).cuda())
        loss = ls+lt
        return loss

    def set_forward_loss(self, xS, xT=None):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        # y_query = Variable(y_query.cuda())
        y_query = y_query.cuda()

        scores, z_source = self.set_forward(xS, is_adversarial=True)
        proto_loss = self.loss_fn(scores, y_query )

        if self.discriminator is not None:
            _, z_target = self.set_forward(xT, is_adversarial=True)
            adverasrial_loss = self.discriminator_score(z_source, z_target, self.adv_loss_fn)
            domain_reg = 0.1
            loss = proto_loss + domain_reg*adverasrial_loss
            return loss, proto_loss, adverasrial_loss
        else:
            return proto_loss


def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def cosine_dist( x, y, tau):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x/torch.norm(x, p=2, dim=1).unsqueeze(1).repeat(1,d)
    y = y/torch.norm(y, p=2, dim=1).unsqueeze(1).repeat(1,d)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return tau*torch.sum(x*y, dim=2)


