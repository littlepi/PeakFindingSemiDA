# -*- coding: utf-8 -*-
"""

Paper and original tensorflow implementation: damodara
Original pytorch translation by marc seibel

@author: Guang Zhao
Semi-supervised implementation of DeepJDOT. Optimized for peak finding problem.

"""

import torch
import numpy as np
import ot
import tqdm
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy
from tqdm import *
from torch.utils.tensorboard import SummaryWriter

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

class DnnModel(torch.nn.Module):
    '''
    embedding output: (*, 100)
    classifier output: (*, 1)
    '''
    def __init__(self, input_len=15, embedding=True):
        super(DnnModel, self).__init__()
        self.embedding=embedding
        self.input_len=input_len
        self.fc1    = torch.nn.Linear(input_len, 500)
        self.fc2    = torch.nn.Linear(500, 100)

        if embedding is True:
            self.fc3    = torch.nn.Linear(100, 100)
        else:
            self.fc3    = torch.nn.Linear(input_len, 100)
        self.fc_out = torch.nn.Linear(100, 1)
        
    def forward(self, x):
        if self.embedding is True:
            out = F.relu(self.fc1(x))
            emb = F.relu(self.fc2(out))
        else:
            emb = x

        out = F.relu(self.fc3(emb))
        clf = F.sigmoid(self.fc_out(out))
        
        return clf, emb

class RnnModel(torch.nn.Module):
    def __init__(self, input_len=15, embedding=True):
        super(RnnModel, self).__init__()
        self.embedding=embedding
        self.fc1 = torch.nn.Linear(input_len, 500)
        self.fc2 = torch.nn.Linear(500, 100)

        self.lstm = torch.nn.LSTM(input_size=1, num_layers=1, hidden_size=32, batch_first=True)
        self.fc3 = torch.nn.Linear(32, 32)
        self.fc4 = torch.nn.Linear(32, 1)

    def forward(self, x):
        # (h0, c0) = (torch.randn(1, x.shape[0], 32), torch.randn(1, x.shape[0], 32))
        # out, (h, c) = self.lstm(x, (h0, c0))

        if self.embedding is True:
            out = F.relu(self.fc1(x))
            emb = F.relu(self.fc2(out))
            out = torch.unsqueeze(emb, -1)
        else:
            emb = torch.unsqueeze(x, -1)
            out = emb

        out, (h, _) = self.lstm(out)
        out = h[-1]
        out = F.relu(self.fc3(out))
        clf = F.sigmoid(self.fc4(out))

        return clf, emb

class ModelTrainer(object):
    def __init__(self, model):
        self.net = model

    def fit(self, train_dataloader, val_dataloader, epoch_size=10, lr=1e-3):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_func = torch.nn.BCELoss()
        metric_func = BinaryAccuracy().to(device)

        for i in range(epoch_size):
            self.net.train()
            nbat = 0
            avg_loss = 0
            avg_acc = 0
            for X, y in train_dataloader:
                X = X.to(device)
                y = y.to(device)
                #print('X, y = {}, {}'.format(X.detach().numpy(), y.detach().numpy()))

                pred, cmb = self.net(X)
                pred.to(device)
                #print('pred = {}'.format(pred.detach().numpy()))
                loss = loss_func(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.detach().numpy()
                avg_acc += metric_func(pred, y).cpu().detach().numpy()
                nbat += 1
            print('Epoch {}:  Train Loss: {:.3f}, Train Acc: {:.3f}'.format(i, avg_loss/nbat, avg_acc/nbat))

            avg_loss_val = 0
            avg_acc_val = 0
            nbat_val = 0
            self.net.eval()
            for X, y in val_dataloader:
                X = X.to(device)
                y = y.to(device)

                pred, _ = self.net(X)
                pred.to(device)
                loss = loss_func(pred, y)
                acc = metric_func(pred, y)
                avg_loss_val += loss.detach().numpy()
                avg_acc_val += acc.cpu().detach().numpy()
                nbat_val += 1
            print('          Val   Loss: {:.3f}, Val   Acc: {:.3f}'.format(avg_loss_val/nbat_val, avg_acc_val/nbat_val))
    
    def predict(self, data):
        data = torch.tensor(data.astype(np.float32))
        self.net.eval()
        with torch.no_grad():
            ypred, _ = self.net(data)
        return ypred

    def evaluate(self, dataloader):
        """
        label as digits (0,1,... , num_classes)
        """

        metric_func = BinaryAccuracy().to(device)
        self.net.eval()
        acc = []
        for X, y in dataloader:
            with torch.no_grad():
                ypred, _ = self.net(X)
            acc.append(metric_func(ypred, y).cpu().detach().numpy())
            
        print('Test acc. = {}'.format(np.mean(acc)))

    def save(self, filename):
        torch.save(self.net.state_dict(), filename)



class DeepJDOT(ModelTrainer):
    def __init__(self, model, lambda_s=1.0, lambda_t=1.0, class_weight=(1.0, 1.0), alpha=0.01, ot_enable=True, ot_class_weight=(1.0, 1.0), ot_method='emd', 
                 init_lr=0.01, lr_decay=True, verbose=1, **kwargs): 
        super(DeepJDOT, self).__init__(model)

        '''
        lambda_s/lambda_t should be 0 or 1
        alpha is a float number from 0.0 to 1.0
        '''
        
        self.net          = model   # target model
        self.lambda_s     = torch.tensor(lambda_s)
        self.lambda_t     = torch.tensor(lambda_t)
        self.alpha        = torch.tensor(alpha)
        self.class_weight = torch.tensor(class_weight)
        
        self.ot_enable = ot_enable
        self.ot_method = ot_method
        self.init_lr  = init_lr
        self.lr_decay= lr_decay
        self.verbose = verbose
        self.ot_class_weight = ot_class_weight

        for key, value in kwargs.items():
            if key == 'sinkhorn_reg':
                self.sinkhorn_reg = value
        
        
        def classification_loss(fs, ft, ys):
            '''
            # fs/ft has shape (B, 1)
            # ys has shape (B, 1)
            '''

            # 1/m * sum_i (CE(ys, fs))
            source_loss = F.binary_cross_entropy(fs, ys, weight=self.bce_weight(ys))
            if self.ot_enable == False: 
                return torch.tensor([0.]), self.lambda_s * source_loss
            
            # loss calculation based on double sum (sum_ij (ys^i, ypred_t^j))
            one_ys = torch.ones(ys.shape)
            one_ft = torch.ones(ft.shape)
            target_loss = -self.class_weight[1] * (torch.matmul(ys, torch.transpose(torch.log(ft), 1, 0)) + 
                           self.class_weight[0] * torch.matmul((one_ys - ys), torch.transpose(torch.log(one_ft - ft), 1, 0)))

            return self.lambda_t * torch.sum(self.gamma * target_loss), self.lambda_s * source_loss
            # return self.lambda_t * torch.sum(self.gamma * target_loss) + self.lambda_s * source_loss


        self.classification_loss = classification_loss
        
        # L2 distance
        # def L2_dist(x,y):
        #     '''
        #     compute the squared L2 distance between two matrics
        #     '''
        #     x2 = torch.reshape(torch.sum(torch.square(x),1), (-1,1)) # (1, 100) -> (100, 1)
        #     y2 = torch.reshape(torch.sum(torch.square(y),1), (1,-1)) # (1, 100) -> (1, 100)
        #     dist = x2 + y2
        #     dist -= 2.0*torch.matmul(x, torch.transpose(y,0,1))  # (B, 100) * (100, B) = (B, B)
        #     return dist
            
       # feature alignment loss
        def align_loss(g_source, g_target):
            '''
            g_source: (B, 100)
            g_target: (B, 100)
            '''
            if self.ot_enable == False: 
                return torch.zeros(1)

            gdist = torch.cdist(g_source, g_target, p=2)**2
            return self.alpha * torch.sum(self.gamma * (gdist))
        self.align_loss= align_loss
        
    def fit(self, source_dataloader, target_dataloader, test_dataloader=None, epoch_size=100, verbose=0):
        method  = self.ot_method # for optimal transport
        alpha   = self.alpha
        # optimizer = torch.optim.SGD(self.net.parameters(), self.init_lr)
        optimizer = torch.optim.Adam(self.net.parameters(), self.init_lr)
        metric_func = BinaryAccuracy().to(device)
        writer = SummaryWriter('results/log/train')
        self.gamma = torch.zeros(size=(source_dataloader.batch_size, target_dataloader.batch_size))
        self.verbose = verbose

        
        for i in tqdm(range(epoch_size)):
            if self.lr_decay and i > 0 and i%500 ==0:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr']*0.1
    
            ### Train
            avg_loss = 0
            avg_acc = 0
            num = 0

            for ((Xs, ys), (Xt, _)) in zip(source_dataloader, target_dataloader):
                Xs.to(device)
                ys.to(device)
                Xt.to(device)
            
                ## Step 1: Fix g and f, solve gamma
                if self.ot_enable == True:
                    self.net.eval() # sets BatchNorm and Dropout in Test mode. Concat of source and target samples and prediction
                    with torch.no_grad():
                        fs, gs = self.net(Xs) # cls, emb
                        ft, gt = self.net(Xt)
                
                        C0 = torch.cdist(gs, gt, p=2.0)**2
                        C1 = F.binary_cross_entropy(ft, ys, weight=self.bce_weight(ys))
                        C = alpha*C0 + C1
                
                        # JDOT optimal coupling (gamma)
                        if self.ot_class_weight == (1.0, 1.0):
                            source_marginal = torch.tensor(ot.unif(gs.shape[0]), dtype=torch.float32)
                        else:
                            source_marginal = []
                            frac = self.ot_class_weight[0]/self.ot_class_weight[1]
                            n0 = len(ys[ys == 0])
                            n1 = len(ys[ys == 1])
                            for (i, g) in enumerate(ys):
                                if ys[i].numpy() == 0:
                                    source_marginal.append(frac/(frac*n0 + n1))
                                else:
                                    source_marginal.append(1/(frac*n0 + n1))
                            source_marginal = torch.tensor(source_marginal, dtype=torch.float32)

                        if self.ot_method == 'emd':
                            self.gamma = ot.emd(source_marginal,
                                                torch.tensor(ot.unif(gt.shape[0]), dtype=torch.float32), C)
                        elif self.ot_method == 'sinkhorn':
                            self.gamma = ot.sinkhorn(source_marginal,
                                                     torch.tensor(ot.unif(gt.shape[0]), dtype=torch.float32), C, self.sinkhorn_reg)
                        else:
                            print('No such OT method. Exit.')
                            exit(0)
                        

                ## Step 2: Fix gamma, update for g and f                
                self.net.train() # Batchnorm and Dropout for train mode
                fs, gs = self.net(Xs) # cls, emb
                ft, gt = self.net(Xt)
                tgt_cat_loss, src_cat_loss = self.classification_loss(fs, ft, ys)
                al_loss = self.align_loss(gs, gt)
                loss = tgt_cat_loss + src_cat_loss + al_loss
                if self.verbose > 0:
                    print('batch {}: tgt_cat_loss = {}, src_cat_loss = {}, al_loss = {}'.format(
                        num, tgt_cat_loss.detach().numpy(), src_cat_loss.detach().numpy(), al_loss.detach().numpy()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = metric_func(fs, ys)
                avg_loss += loss.detach().numpy()
                avg_acc += acc.cpu().detach().numpy()
                num += 1

            print('  Epoch {}: loss = {}, acc = {}'.format(i, avg_loss/num, avg_acc/num))
            writer.add_scalar('Train/loss', avg_loss/num, i)
            writer.add_scalar('Train/acc', avg_acc/num, i)
            if self.verbose > 1:
                torch.set_printoptions(profile="full")
                print(self.gamma)

            ### Validation           
            avg_test_loss = 0
            avg_test_acc = 0
            num_test = 0
            if test_dataloader is not None:
                self.net.eval()
                for Xtest, ytest in test_dataloader:
                    with torch.no_grad():
                        ypred, _ = self.net(Xtest)
                    loss = F.binary_cross_entropy(ypred, ytest)
                    acc = metric_func(ypred, ytest)
                    avg_test_loss += loss.detach().numpy()
                    avg_test_acc += acc.cpu().detach().numpy()
                    num_test += 1
                    # print('epoch {}, loss = {}, acc = {}, num = {}'.format(i, loss, acc, num_test))

                writer.add_scalar('Val/loss', avg_test_loss/num_test, i)
                writer.add_scalar('Val/acc', avg_test_acc/num_test, i)
                print('          val. loss = {}, val. acc = {}'.format(avg_test_loss/num_test, avg_test_acc/num_test))

        writer.close()
                        
    # target classification cross ent loss and source cross entropy
    def bce_weight(self, y):
        weight = torch.zeros(y.shape)
        for i in range(y.shape[0]):
            if y.dim() == 1:
                weight[i] = self.class_weight[0] if y[i].item() == 0 else self.class_weight[1]
            if y.dim() == 2:
                weight[i, 0] = self.class_weight[0] if y[i].item() == 0 else self.class_weight[1]

        return weight

class DeepSemiJDOT(DeepJDOT):
    def __init__(self, model, lambda_s=1.0, lambda_tl=1.0, lambda_t=1.0, class_weight=(1.0, 1.0), alpha=0.01, ot_enable=True, ot_class_weight=(1.0, 1.0), ot_method='emd', 
                 init_lr=0.01, lr_decay=True, verbose=1, **kwargs): 
        super(DeepSemiJDOT, self).__init__(model)

        '''
        lambda_s/lambda_t should be 0 or 1
        alpha is a float number from 0.0 to 1.0
        '''
        
        self.net          = model   # target model
        self.lambda_s     = torch.tensor(lambda_s)
        self.lambda_tl    = torch.tensor(lambda_tl)
        self.lambda_t     = torch.tensor(lambda_t)
        self.alpha        = torch.tensor(alpha)
        self.class_weight = torch.tensor(class_weight)
        
        self.ot_enable = ot_enable
        self.ot_method = ot_method
        self.init_lr  = init_lr
        self.lr_decay= lr_decay
        self.verbose = verbose
        self.ot_class_weight = ot_class_weight
        
        for key, value in kwargs.items():
            if key == 'sinkhorn_reg':
                self.sinkhorn_reg = value
        
        def classification_loss(fs, ft, ys, ft_l, yt_l):
            '''
            # fs/ft has shape (B, 1)
            # ys has shape (B, 1)
            '''

            # 1/m * sum_i (CE(ys, fs))
            source_loss = F.binary_cross_entropy(fs, ys, weight=self.bce_weight(ys))
            target_label_loss = F.binary_cross_entropy(ft_l, yt_l, weight=self.bce_weight(yt_l))
            if self.ot_enable == False: 
                return torch.tensor([0.]), self.lambda_s * source_loss, self.lambda_tl * target_label_loss
            
            # loss calculation based on double sum (sum_ij (ys^i, ypred_t^j))
            one_ys = torch.ones(ys.shape)
            one_ft = torch.ones(ft.shape)
            target_loss = -self.class_weight[1] * (torch.matmul(ys, torch.transpose(torch.log(ft), 1, 0)) + 
                           self.class_weight[0] * torch.matmul((one_ys - ys), torch.transpose(torch.log(one_ft - ft), 1, 0)))

            return self.lambda_t * torch.sum(self.gamma * target_loss), self.lambda_s * source_loss, self.lambda_tl * target_label_loss
            # return self.lambda_t * torch.sum(self.gamma * target_loss) + self.lambda_s * source_loss


        self.classification_loss = classification_loss
        
        # L2 distance
        # def L2_dist(x,y):
        #     '''
        #     compute the squared L2 distance between two matrics
        #     '''
        #     x2 = torch.reshape(torch.sum(torch.square(x),1), (-1,1)) # (1, 100) -> (100, 1)
        #     y2 = torch.reshape(torch.sum(torch.square(y),1), (1,-1)) # (1, 100) -> (1, 100)
        #     dist = x2 + y2
        #     dist -= 2.0*torch.matmul(x, torch.transpose(y,0,1))  # (B, 100) * (100, B) = (B, B)
        #     return dist
            
       # feature alignment loss
        def align_loss(g_source, g_target):
            '''
            g_source: (B, 100)
            g_target: (B, 100)
            '''
            if self.ot_enable == False: 
                return torch.zeros(1)

            gdist = torch.cdist(g_source, g_target, p=2)**2
            return self.alpha * torch.sum(self.gamma * (gdist))
        self.align_loss= align_loss
        
    def fit(self, source_dataloader, target_dataloader, test_dataloader=None, epoch_size=100, verbose=0):
        method  = self.ot_method # for optimal transport
        alpha   = self.alpha
        # optimizer = torch.optim.SGD(self.net.parameters(), self.init_lr)
        optimizer = torch.optim.Adam(self.net.parameters(), self.init_lr)
        metric_func = BinaryAccuracy().to(device)
        writer = SummaryWriter('results/log/train')
        self.gamma = torch.zeros(size=(source_dataloader.batch_size, target_dataloader.batch_size))
        self.verbose = verbose

        
        for iepoch in tqdm(range(epoch_size)):
            if self.lr_decay and iepoch > 0 and iepoch%500 ==0:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr']*0.1
    
            ### Train
            avg_loss = 0
            avg_acc = 0
            avg_acc_l = 0
            num = 0

            for ((Xs, ys), (Xt, yt)) in zip(source_dataloader, target_dataloader):
                l_idx = [ i for (i, y) in enumerate(yt) if y.item() != -1 ]
                Xt_l = Xt[l_idx].to(device)
                yt_l = yt[l_idx].to(device)
                # u_idx = [ i for (i, y) in enumerate(yt) if y.item() == -1 ]
                # Xt_u = Xt[u_idx].to(device)
                # yt_u = yt[u_idx].to(device)
                Xs.to(device)
                ys.to(device)
                # Xt.to(device)
            
                ## Step 1: Fix g and f, solve gamma
                if self.ot_enable == True:
                    self.net.eval() # sets BatchNorm and Dropout in Test mode. Concat of source and target samples and prediction
                    with torch.no_grad():
                        fs, gs = self.net(Xs) # cls, emb
                        ft, gt = self.net(Xt)
                
                        C0 = torch.cdist(gs, gt, p=2.0)**2
                        C1 = F.binary_cross_entropy(ft, ys, weight=self.bce_weight(ys))
                        C = alpha*C0 + C1
                
                        # JDOT optimal coupling (gamma)
                        if self.ot_class_weight == (1.0, 1.0):
                            source_marginal = torch.tensor(ot.unif(gs.shape[0]), dtype=torch.float32)
                        else:
                            source_marginal = []
                            frac = self.ot_class_weight[0]/self.ot_class_weight[1]
                            n0 = len(ys[ys == 0])
                            n1 = len(ys[ys == 1])
                            for (i, g) in enumerate(ys):
                                if ys[i].numpy() == 0:
                                    source_marginal.append(frac/(frac*n0 + n1))
                                else:
                                    source_marginal.append(1/(frac*n0 + n1))
                            source_marginal = torch.tensor(source_marginal, dtype=torch.float32)

                        if self.ot_method == 'emd':
                            self.gamma = ot.emd(source_marginal,
                                                torch.tensor(ot.unif(gt.shape[0]), dtype=torch.float32), C)
                        elif self.ot_method == 'sinkhorn':
                            self.gamma = ot.sinkhorn(source_marginal,
                                                     torch.tensor(ot.unif(gt.shape[0]), dtype=torch.float32), C, self.sinkhorn_reg)
                        else:
                            print('No such OT method. Exit.')
                            exit(0)
                        

                ## Step 2: Fix gamma, update for g and f                
                self.net.train() # Batchnorm and Dropout for train mode
                fs, gs = self.net(Xs) # cls, emb
                ft, gt = self.net(Xt)
                ft_l, gt_l = self.net(Xt_l)
                tgt_cat_loss, src_cat_loss, tgt_lbl_cat_loss = self.classification_loss(fs, ft, ys, ft_l, yt_l)
                al_loss = self.align_loss(gs, gt)
                loss = tgt_cat_loss + src_cat_loss + al_loss + tgt_lbl_cat_loss
                if self.verbose > 0:
                    print('batch {}: tgt_cat_loss = {}, tgt_lbl_loss = {}, src_cat_loss = {}, al_loss = {}'.format(
                        num, tgt_cat_loss.detach().numpy(), tgt_lbl_cat_loss.detach().numpy(), src_cat_loss.detach().numpy(), al_loss.detach().numpy()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = metric_func(fs, ys)
                acc_l = metric_func(ft_l, yt_l)
                avg_loss += loss.detach().numpy()
                avg_acc += acc.cpu().detach().numpy()
                avg_acc_l += acc_l.cpu().detach().numpy()
                num += 1

            print('  Epoch {}: loss = {}, acc = {}, acc_l = {}'.format(iepoch, avg_loss/num, avg_acc/num, avg_acc_l/num))
            writer.add_scalar('Train/loss', avg_loss/num, iepoch)
            writer.add_scalar('Train/acc', avg_acc/num, iepoch)
            if self.verbose > 1:
                torch.set_printoptions(profile="full")
                print(self.gamma)

            ### Validation           
            avg_test_loss = 0
            avg_test_acc = 0
            num_test = 0
            if test_dataloader is not None:
                self.net.eval()
                for Xtest, ytest in test_dataloader:
                    with torch.no_grad():
                        ypred, _ = self.net(Xtest)
                    loss = F.binary_cross_entropy(ypred, ytest)
                    acc = metric_func(ypred, ytest)
                    avg_test_loss += loss.detach().numpy()
                    avg_test_acc += acc.cpu().detach().numpy()
                    num_test += 1
                    # print('epoch {}, loss = {}, acc = {}, num = {}'.format(i, loss, acc, num_test))

                writer.add_scalar('Val/loss', avg_test_loss/num_test, iepoch)
                writer.add_scalar('Val/acc', avg_test_acc/num_test, iepoch)
                print('          val. loss = {}, val. acc = {}'.format(avg_test_loss/num_test, avg_test_acc/num_test))

        writer.close()