import torch
from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import ROOT
from ROOT import TFile
from lib.utility import *
import ot
from torch.utils.data import ConcatDataset

class WaveformSliceDataset(Dataset):
    def __init__(self, file_name, tree_name='sim', start=0, length=-1, nleft=5, nright=9, sign=1, norm_method='std', 
                 with_tag=True, tag_method='default', aux_tag_method=None):
        self.wf_list = []
        self.time_list = []
        self.tag_list = []
        self.aux_tag_list = []
        self.evtno2idx_dict = defaultdict(list) # key is evtno, value is a list of index
        self.baseline_rms = -99.
        self.norm_method = norm_method
        self.tag_method = tag_method
        self.aux_tag_method = aux_tag_method
        # for (key, value) in kwargs.items():
        #     if key == 'tag_method':
        #         self.tag_method = value
        #     if key == 'aux_tag_method':
        #         self.aux_tag_list = value
        
        f = TFile(file_name)
        t = f.Get(tree_name)
        wf = ROOT.std.vector['double'](0)
        time = ROOT.std.vector['double'](0)
        tag = ROOT.std.vector['int'](0)
        t.SetBranchAddress('wf_i', wf)
        if with_tag:
            t.SetBranchAddress('time', time)
            t.SetBranchAddress('tag', tag) # 0 for primary, 1 for secondary, 2 for noise
        
        wf_len = nleft + nright + 1
        size = t.GetEntries()
        stop = start + length if length > 0 and start + length < size else size
        idx = 0
        for i in tqdm(range(start, stop), desc='Creating dataset'):
            t.GetEntry(i)

            _wf = np.array(wf)
            _time = np.array(time)
            _tag = np.array(tag)

            if sign < 0:
                for j in range(len(_wf)):
                    _wf[j] *= -1.

            if with_tag is False:
                _time = np.array(self.__find_bump(_wf))

            if tag_method == 'default' and with_tag:
                tag_list = [1 if _t > 0 else _t for _t in _tag]
            elif tag_method is not None:
                tag_list = self.tag_method(_wf, _time)

            if aux_tag_method == 'default' and with_tag:
                aux_tag_list = [1 if _t > 0 else _t for _t in _tag]
            elif aux_tag_method is not None:
                aux_tag_list = self.aux_tag_method(_wf, _time)


            # has_tag = False
            # if with_tag:
            #     tag_list = [1 if _t > 0 else _t for _t in _tag]
            #     has_tag = True
            # else:
            #     tag_list = []
            #     _time = np.array(self.__find_bump(_wf))
            #     if self.tag_method is not None:
            #         tag_list = self.tag_method(_wf, _time)
            #         has_tag = True

            # if semi_mode == True:
            #     semi_tag_list = []
            #     if self.semi_method is not None:
            #         semi_tag_list = self.semi_method(_wf, _time)
            #     else:
            #         for _t in _time:
            #             semi_tag_list.append(self.__add_semi_tag(_wf, _t))

            for j in range(len(_time)):
                peak_idx = int(_time[j])
                if peak_idx < nleft or peak_idx > len(_wf) - nright - 1:
                    continue

                # if has_tag and semi_mode:
                #     self.tag_list.append(semi_tag_list[j]) # semi tag
                #     self.aux_tag_list.append(tag_list[j]) # default tag
                # if has_tag and not semi_mode:
                #     self.tag_list.append(tag_list[j])
                # if not has_tag and semi_mode:
                #     self.tag_list.append(semi_tag_list[j])
                # if not has_tag and not semi_mode:
                #     self.tag_list.append(-1)

                if tag_method is not None: self.tag_list.append(tag_list[j])
                else: self.tag_list.append(-1)
                if aux_tag_method is not None: self.aux_tag_list.append(aux_tag_list[j])

                wf_slice = np.zeros((wf_len,)) # rnn: (wf_len, 1)
                for k in range(wf_len):
                    wf_slice[k] = _wf[peak_idx - nleft + k]
                wf_slice = self.__preprocessing(wf_slice)

                self.wf_list.append(wf_slice)
                self.time_list.append(peak_idx)
                self.evtno2idx_dict[i].append(idx)
                idx += 1


        print('[WaveformSliceDataset] : Total # of waveform slices = {}'.format(len(self.wf_list)))
        print('[WaveformSliceDataset] : Dataset has a size of {}'.format(self.wf_list[0].shape))

    def __len__(self):
        return len(self.wf_list)

    def __getitem__(self, idx):
        wf_tensor = torch.tensor(self.wf_list[idx], dtype=torch.float32)
        tag_tensor = torch.unsqueeze(torch.tensor(self.tag_list[idx], dtype=torch.float32), -1)
        return wf_tensor, tag_tensor

    def __find_bump(self, wf):
        time = []
        for i in range(1, len(wf)-1):
            if wf[i] - wf[i - 1] > wf[i + 1] - wf[i]:
                time.append(i)
        return time
    
    # def __add_semi_tag(self, wf, time):
    #     idx = int(time)
    #     if self.baseline_rms < 0:
    #         baseline = np.array(wf[0:50])
    #         self.baseline_rms = baseline.std()

    #     if idx-2 < 0 or idx+2 > wf.size() - 1:
    #         return -1

    #     if wf[idx] < 3*self.baseline_rms:
    #         tag = 0
    #     elif wf[idx] - wf[idx-1] > 0.2*self.baseline_rms and \
    #          wf[idx] - wf[idx+1] > 0.2*self.baseline_rms and \
    #          wf[idx] - wf[idx-2] > 0.4*self.baseline_rms and \
    #          wf[idx] - wf[idx+2] > 0.4*self.baseline_rms and \
    #          wf[idx] > 10*self.baseline_rms:
    #         tag = 1
    #     else:
    #         tag = -1

    #     return tag

    def __preprocessing(self, wf):
        wf_out = np.array(wf)

        if self.norm_method == 'std':
            mean = np.mean(wf)
            sigma = np.std(wf)
            for i in range(len(wf)):
                wf_out[i] = (wf[i] - mean)/sigma

        if self.norm_method == 'shift':
            mean = np.mean(wf)
            for i in range(len(wf)):
                wf_out[i] = wf[i] - mean

        if self.norm_method == 'pdf':
            wf_sum = sum(wf_out)
            for i in range(len(wf)):
                wf_out[i] /= wf_sum

        return wf_out
    
    def GetClassWeight(self):
        num_class_0 = 0
        num_class_1 = 1
        for _tag in self.tag_list:
            if _tag == 0:
                num_class_0 += 1
            if _tag == 1:
                num_class_1 += 1
        num_total = num_class_1 + num_class_0

        # num_total = len(self.tag_list)
        # num_class_1 = np.sum(self.tag_list)
        # num_class_0 = num_total - num_class_1
        
        if num_class_0 == 0 or num_class_1 == 0:
            return (1., 1.)
        else:
            return (num_class_1/num_total, num_class_0/num_total)

    def GetWaveformSliceTime(self, idx):
        return self.time_list[idx]

    def GetEventNoToIndexDict(self):
        return self.evtno2idx_dict
    
    def GetAuxTagList(self):
        return self.aux_tag_list


class ToyDataset(Dataset):
    def __init__(self, n=1000, w=(0.5, 0.5), 
                 mu1=[0, 0], cov1=[[1, 0], [0, 1]],
                 mu2=[0, 0], cov2=[[1, 0], [0, 1]]):
        
        n1 = int(n * w[0]/(w[0] + w[1]))
        n2 = n - n1
        x1 = ot.datasets.make_2D_samples_gauss(n1, mu1, cov1)
        x2 = ot.datasets.make_2D_samples_gauss(n2, mu2, cov2)
        y1 = np.zeros(n1)
        y2 = np.ones(n2)
        
        self.n = n
        self.X = np.concatenate((x1, x2)).astype('float32')
        self.y = np.concatenate((y1, y2)).astype('float32')
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        X = torch.tensor(self.X[index], dtype=torch.float32)
        y = torch.tensor(np.expand_dims(self.y[index], axis=-1), dtype=torch.float32)
        return (X, y)
    
    def data(self):
        return (self.X, self.y)
    

class ConcatWaveformSliceDataset(ConcatDataset):
    def __init__(self, datasets):
        super(ConcatWaveformSliceDataset, self).__init__(datasets)
        self.time_list = []
        self.tag_list = []
        self.aux_tag_list = []
        self.evtno2idx_dict = {}
        for (i, d) in enumerate(datasets):
            self.time_list.extend(d.time_list)
            self.tag_list.extend(d.tag_list)
            self.aux_tag_list.extend(d.aux_tag_list)

            if i == 0:
                self.evtno2idx_dict = d.evtno2idx_dict
                last_evtno = list(self.evtno2idx_dict)[-1]
            else:
                for (k, v) in d.evtno2idx_dict.items():
                    last_evtno += 1
                    self.evtno2idx_dict[last_evtno] = v
    
    def GetClassWeight(self):
        num_class_0 = 0
        num_class_1 = 1
        for _tag in self.tag_list:
            if _tag == 0:
                num_class_0 += 1
            if _tag == 1:
                num_class_1 += 1
        num_total = num_class_1 + num_class_0

        if num_class_0 == 0 or num_class_1 == 0:
            return (1., 1.)
        else:
            return (num_class_1/num_total, num_class_0/num_total)

    def GetWaveformSliceTime(self, idx):
        return self.time_list[idx]

    def GetEventNoToIndexDict(self):
        return self.evtno2idx_dict

    def GetAuxTagList(self):
        return self.aux_tag_list
    