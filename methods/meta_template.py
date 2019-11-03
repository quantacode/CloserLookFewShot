import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import make_grid
import numpy as np
import torch.nn.functional as F
import utils
from abc import abstractmethod
import ipdb


class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, change_way = True, change_shots = True):
        super(MetaTemplate, self).__init__()
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = -1 #(change depends on input) 
        self.feature    = model_func()
        self.feat_dim   = self.feature.final_feat_dim
        self.change_way = change_way  #some methods allow different_way classification during training and test

    @abstractmethod
    def set_forward(self,x,is_feature, is_adversarial):
        pass

    @abstractmethod
    def set_forward_loss(self, x1, x2=None):
        pass

    @abstractmethod
    def discriminator_score(self, z_source, z_target, adv_loss_fn):
        pass

    def forward(self,x):
        out  = self.feature.forward(x)
        return out

    def parse_feature(self,x,is_feature):
        # x = Variable(x.cuda())
        x = x.cuda()
        if is_feature:
            z_all = x
        else:
            try:
                x = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:])
                # n_way, n_rest = x.size()[:2]
                # x = x.contiguous().view( -1, *x.size()[2:])
            except:
                import ipdb
                ipdb.set_trace()
            z_all = self.feature.forward(x)
            z_all = z_all.view( self.n_way, self.n_support + self.n_query, -1)
            # z_all = z_all.view(n_way, n_rest, -1)
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]

        return z_support, z_query




    def correct(self, x):       
        scores = self.set_forward(x)
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query)

    def train_loop_adversarial(self, epoch, train_loaders, optimizer, writer):
        print_freq = 10

        tl_source = iter(train_loaders[0])
        tl_target = iter(train_loaders[1])
        num_batches = np.min([len(tl_source), len(tl_target)])

        avg_loss=0
        avg_lossP=0
        avg_lossAdv=0
        for i in range(num_batches):
            x_source, _ = next(tl_source)
            x_target, _ = next(tl_target)

            # Note : Temp hack since couldnt find why subsampler works differently for source and target
            # command to check: tl_source.dataset.sub_dataloader[-1].batch_sampler.sampler.num_samples

            # if cross_char, need to adjust the shape of target
            x_target = x_target[:, :x_source.shape[1], :, :, :]

            self.n_query = x_source.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x_source.size(0)
            optimizer.zero_grad()

            loss, lossP, lossAdv = self.set_forward_loss( x_source, x_target )
            loss.backward()
            optimizer.step()
            # ipdb.set_trace()
            # print('main model: \n',[p.mean().item() for p in list(self.feature.parameters())])
            # print('disc model: \n',[p.mean().item() for p in list(self.discriminator.parameters())])
            # print('disc grad: \n',[p.grad.mean().item() for p in list(self.discriminator.parameters())])

            avg_loss = avg_loss+loss.data.item()
            avg_lossP = avg_lossP+lossP.data.item()
            avg_lossAdv = avg_lossAdv+lossAdv.data.item()

            if i % print_freq==0:
                mean_loss =  avg_loss/float(i+1)
                mean_lossP =  avg_lossP/float(i+1)
                mean_lossAdv =  avg_lossAdv/float(i+1)
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | total loss {:f} | LossP {:f} | Loss_adv {:f}'.
                      format(epoch, i, num_batches, mean_loss, mean_lossP, mean_lossAdv))
        writer.add_scalar('train/loss', mean_loss, epoch)
        writer.add_scalar('train/loss primary', mean_lossP, epoch)
        writer.add_scalar('train/loss adversarial', mean_lossAdv, epoch)
        log_images(x_source, 'train/source', epoch, writer)
        log_images(x_target, 'train/target', epoch, writer)

    def train_loop(self, epoch, train_loader, optimizer, writer, params = None):
        print_freq = 10
        avg_loss = 0

        for i, (x,_ ) in enumerate(train_loader):
            if params.n_shot_test != -1:
                # to nullify the modification made at test time
                self.n_support = params.n_shot
            if self.change_way:
                self.n_way  = x.size(0)

            self.n_query = x.size(1) - self.n_support
            optimizer.zero_grad()

            loss = self.set_forward_loss( x)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss+loss.data.item()

            if i % print_freq==0:
                mean_loss =  avg_loss/float(i+1)
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), mean_loss))
        # log
        writer.add_scalar('train/loss', mean_loss, epoch)
        log_images(x[:, :self.n_support,:,:,:], 'train/support_set', epoch, writer)
        log_images(x[:, self.n_support:,:,:,:], 'train/query_set', epoch, writer)


    def test_loop(self, test_loader, writer, epoch, params , record = None):
        correct =0
        count = 0
        acc_all = []

        iter_num = len(test_loader) 
        for i, (x,_) in enumerate(test_loader):
            if params.n_shot_test != -1:
                self.n_support = params.n_shot_test
            if self.change_way:
                self.n_way  = x.size(0)

            self.n_query = x.size(1) - self.n_support
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this/ count_this*100  )

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        # log
        writer.add_scalar('test/accuracy', acc_mean, epoch)
        log_images(x[:, :self.n_support,:,:,:], 'test/support_set', epoch, writer)
        log_images(x[:, self.n_support:,:,:,:], 'test/query_set', epoch, writer)
        return acc_mean

    def set_forward_adaptation(self, x, is_feature = True): #further adaptation, default is fixing feature and train a new softmax clasifier
        assert is_feature == True, 'Feature is fixed in further adaptation'
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous().view(self.n_way* self.n_support, -1 )
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        # y_support = Variable(y_support.cuda())
        y_support = y_support.cuda()

        linear_clf = nn.Linear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()
        
        batch_size = 4
        support_size = self.n_way* self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id] 
                scores = linear_clf(z_batch)
                loss = loss_function(scores,y_batch)
                loss.backward()
                set_optimizer.step()

        scores = linear_clf(z_query)
        return scores

def log_images(x, logid, epoch, writer):
    sliced_img = x[:, :, :, :, :]
    # sliced_img = x[:, :3, :, :, :]
    n_way, n_shot, chan, row, col = sliced_img.shape
    sliced_img = sliced_img.reshape(-1, chan, row, col)
    disp_img = make_grid(sliced_img, nrow=n_shot, normalize=True)
    writer.add_image(logid, disp_img, epoch)
