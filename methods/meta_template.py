import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import make_grid
import numpy as np
import torch.nn.functional as F
import utils
from abc import abstractmethod
import operator
import ipdb


ADDA_BatchSize = 32
DAN_BatchSize = 4
# if torch.cuda.device_count() == 2:
#     DEVICE1 = torch.device("cuda:0")
#     DEVICE2 = torch.device("cuda:1")

class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, change_way = True, change_shots = True):
        super(MetaTemplate, self).__init__()
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = -1 #(change depends on input) 
        self.feature    = model_func()
        self.feat_dim   = self.feature.final_feat_dim
        if (torch.cuda.device_count() > 1):
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.feature = nn.DataParallel(self.feature)
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
 
    def train_loop_ADDA(self, epoch, train_loaders, FES, optimizerD, optimizerG, writer):
        print_freq = 10
        tl_source = iter(train_loaders[0])
        tl_target = iter(train_loaders[1])
        num_batches = np.min([len(tl_source), len(tl_target)])

        avg_lossD = 0
        avg_lossG = 0
        i=-1

        while i<num_batches:
            z_source = torch.FloatTensor([]).to(DEVICE1)
            z_target = torch.FloatTensor([]).to(DEVICE2)
            for _ in range(ADDA_BatchSize):
                if i >= num_batches-1:
                    break
                i+=1
                x_source, _ = next(tl_source)
                x_target, _ = next(tl_target)
                x_source, x_target = x_source.to(DEVICE1), x_target.to(DEVICE2)
                # if cross_char, need to adjust the shape of target
                x_target = x_target[:, :x_source.shape[1], :, :, :]
                self.n_query = x_source.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x_source.size(0)

                x_source = x_source.to(DEVICE1)
                x_source = x_source.contiguous().view(self.n_way * (self.n_support + self.n_query), *x_source.size()[2:])
                featS = FES.forward(x_source)
                featS = featS.view(self.n_way, self.n_support + self.n_query, -1)

                ## LossD
                _, zS = self.set_forward(featS, is_feature=True, is_adversarial=True)
                _, zT = self.set_forward(x_target, is_adversarial=True)
                z_source = torch.cat([z_source.to(zS.device), zS])
                z_target = torch.cat([z_target.to(zT.device), zT])
            if  z_source.shape[0]==0: break
            z_all = torch.cat([z_source.to(DEVICE2), z_target])
            logit = self.discriminator(z_all)
            labels = torch.cat([torch.ones(z_source.shape[0], dtype=torch.long).to(DEVICE2),
                                torch.zeros(z_target.shape[0], dtype=torch.long).to(DEVICE2)])
            lossD = self.adv_loss_fn(logit, labels)
            accD = 100.0*((logit[:z_source.shape[0], 0] < logit[:z_source.shape[0], 1]).sum() + \
                   (logit[z_source.shape[0]:, 0] > logit[z_source.shape[0]:, 1]).sum()) / \
                   (1.0 * (logit.shape[0]))
            # logitS = self.discriminator(z_source.to(DEVICE2))
            # logitT = self.discriminator(z_target)
            # ls = self.adv_loss_fn(logitS, torch.ones(z_source.shape[0], dtype=torch.long).to(DEVICE2))
            # lt = self.adv_loss_fn(logitT, torch.zeros(z_target.shape[0], dtype=torch.long).to(DEVICE2))
            # lossD = ls + lt
            # accD = (logitS[:, 0] < logitS[:, 1]).sum() + (logitT[:, 0] > logitT[:, 1]).sum() / \
            #        (1.0 * (logitS.shape[0] + logitT.shape[0]))
            if accD<70:
                optimizerD.zero_grad()
                lossD.backward(retain_graph=True)
                optimizerD.step()

            ## LossG
            z_target = torch.FloatTensor([]).to(DEVICE2)
            for _ in range(ADDA_BatchSize):
                if i >= num_batches-1:
                    break
                i+=1
                x_source, _ = next(tl_source)
                x_target, _ = next(tl_target)
                # if cross_char, need to adjust the shape of target
                x_target = x_target[:, :x_source.shape[1], :, :, :]
                _, zT = self.set_forward(x_target, is_adversarial=True)
                z_target = torch.cat([z_target.to(zT.device), zT])
            if  z_target.shape[0]==0: break
            logitT = self.discriminator(z_target)
            lossG = self.adv_loss_fn(logitT, torch.ones(z_target.shape[0], dtype=torch.long).to(DEVICE2))
            optimizerD.zero_grad()
            optimizerG.zero_grad()
            lossG.backward(retain_graph=True)
            optimizerG.step()

        print('Epoch {:d} | Batch {:d}/{:d} | LossD {:f} | LossG {:f} | accD {:f}'.
                  format(epoch, i, num_batches, lossD.item(), lossG.item(), accD))
        # writer.add_scalar('train/loss discriminator', mean_lossD, epoch)
        # writer.add_scalar('train/loss generator', mean_lossG, epoch)
        # if epoch == 2:
        #     log_images(x_source, 'train/source', epoch, writer)
        #     log_images(x_target, 'train/target', epoch, writer)

    def train_loop_DAN(self, epoch, train_loaders, optimizer, writer, params=None):
        print_freq = 10
        tl_source = iter(train_loaders[0])
        tl_target = iter(train_loaders[1])
        num_batches = np.min([len(tl_source), len(tl_target)])

        bid=-1
        avg_loss = 0
        avg_lossP = 0
        avg_lossAdv = 0
        while bid<num_batches:
            z_source = torch.FloatTensor([]).cuda()
            z_target = torch.FloatTensor([]).cuda()
            lossP = torch.FloatTensor([]).cuda()
            for _ in range(DAN_BatchSize):
                if bid >= num_batches-1:
                    break
                bid+=1
                x_source, _ = next(tl_source)
                x_target, _ = next(tl_target)
                x_source, x_target = x_source.cuda(), x_target.cuda()
                # if cross_char, need to adjust the shape of target
                x_target = x_target[:, :x_source.shape[1], :, :, :]
                self.n_query = x_source.size(1) - self.n_support
                if self.change_way:
                    self.n_way = x_source.size(0)

                # source
                y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
                y_query = y_query.cuda()
                scores, zS = self.set_forward(x_source, is_adversarial=True)
                lossP = torch.cat([lossP, self.loss_fn(scores, y_query).unsqueeze(0)])

                # target
                _, zT = self.set_forward(x_target, is_adversarial=True)

                z_source = torch.cat([z_source, zS])
                z_target = torch.cat([z_target, zT])

            if z_source.shape[0] == 0: break
            optimizer.zero_grad()
            lossAdv = self.DAN_loss(z_source, z_target)
            lossP = lossP.mean()
            loss = lossP + params.gamma * lossAdv
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
            avg_lossP = avg_lossP + lossP.item()
            avg_lossAdv = avg_lossAdv + lossAdv.item()

        mean_loss = avg_loss / float(bid + 1)
        mean_lossP = avg_lossP / float(bid + 1)
        mean_lossAdv = avg_lossAdv / float(bid + 1)
        # print(optimizer.state_dict()['param_groups'][0]['lr'])
        print('Epoch {:d} | Batch {:d}/{:d} | total loss {:f} | LossP {:f} | Loss_adv {:f}'.
              format(epoch, bid, num_batches, mean_loss, mean_lossP, mean_lossAdv))
        writer.add_scalar('train/loss', mean_loss, epoch)
        writer.add_scalar('train/loss primary', mean_lossP, epoch)
        writer.add_scalar('train/loss adversarial', mean_lossAdv, epoch)
        if epoch == 2:
            log_images(x_source, 'train/source', epoch, writer)
            log_images(x_target, 'train/target', epoch, writer)

    def train_loop_PRODA(self, epoch, train_loaders, optimizer, writer, params=None):
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

            # # if cross_char, need to adjust the shape of target
            x_target = x_target[:, :x_source.shape[1], :, :, :]
            self.n_query = x_source.size(1) - self.n_support

            if self.change_way:
                self.n_way = x_source.size(0)

            optimizer.zero_grad()

            loss, lossP, lossAdv = self.set_forward_loss( x_source, x_target, params=params, epoch=epoch)
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
        if epoch==2:
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

            loss = self.set_forward_loss(x)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss+loss.data.item()

            if i % print_freq==0:
                mean_loss =  avg_loss/float(i+1)
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), mean_loss))
        # log
        writer.add_scalar('train/loss', mean_loss, epoch)
        if epoch==2:
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
        if np.mod(epoch,100)==0:
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
