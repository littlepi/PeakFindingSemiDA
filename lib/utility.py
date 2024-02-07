import matplotlib.pyplot as plt
import random
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from scipy import signal
import ROOT
from ROOT import TFile, vector
from sklearn.cluster import AgglomerativeClustering

def plot_waveform(wf, sign=1, xrange=None, time=None, time_cls=None, truth_time=None, truth_tag=None, filename=None, figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel('Index')
    ax.set_ylabel('Amplitude')

    wf_x = list(range(len(wf)))
    if sign < 0:
        wf = [-w for w in wf]
    ax.step(wf_x, wf,label='Waveform')
    if xrange is not None: ax.set_xlim(xrange)

    if time is not None:
        amp = [wf[int(t)] for t in time] 
        ax.plot(time, amp, 'o', label='Detected electrons')

    if time_cls is not None:
        amp = [wf[int(t)] for t in time_cls] 
        ax.plot(time_cls, amp, '^b', label='Detected clusters')

    if truth_time is not None:
        truth_time_pri = [t for i, t in enumerate(truth_time) if truth_tag[i] == 1]
        truth_amp_pri = [wf[int(t)] for i, t in enumerate(truth_time) if truth_tag[i] == 1]
        ax.vlines(truth_time_pri, 0, truth_amp_pri, color='orange', label='Primary electrons')

        truth_time_sec = [t for i, t in enumerate(truth_time) if truth_tag[i] == 2]
        truth_amp_sec = [wf[int(t)] for i, t in enumerate(truth_time) if truth_tag[i] == 2]
        ax.vlines(truth_time_sec, 0, truth_amp_sec, color='green', label='Secondary electrons')

    if filename is not None:
        plt.savefig(filename)

    ax.legend()

def analysis_waveform(filename, treename, irange=(0, 1), sign=1, with_truth=False,
                      dataset=None, model=None, thr=0.9, xrange=None, clustering_cut=-1., figsize=(12, 6)):
    f = TFile(filename)
    t = f.Get(treename)
    wf = vector['double'](0)
    time = vector['double'](0)
    tag = vector['int'](0)
    t.SetBranchAddress('wf_i', wf)
    if with_truth:
        t.SetBranchAddress('time', time)
        t.SetBranchAddress('tag', tag)

    for i in range(irange[0], irange[0]+irange[1]):
        t.GetEntry(i)

        detX = None
        detX_cls = None
        truthX = None
        truthTag = None
        if dataset is not None:
            wf_slice_idx = dataset.GetEventNoToIndexDict()[i]
            detX = []
            detX_cls = []
            truthX = []
            truthTag = []
            for j in wf_slice_idx:
                pred, _ = model(dataset[j][0])
                idx = int(dataset.GetWaveformSliceTime(j))
                if pred.detach().numpy() > thr:
                    detX.append(idx)
            if clustering_cut > 0.:
                cluster_id = AgglomerativeClustering(n_clusters=None, distance_threshold=clustering_cut).fit_predict(np.array(detX).reshape(-1, 1))
                cluster_map = {k:[] for k in cluster_id}
                for idx, k in enumerate(cluster_id):
                    cluster_map[k].append(detX[idx])

                for k in cluster_map:
                    detX_cls.append(np.mean(cluster_map[k]))


        if with_truth is not None:
            truthX = list(time)
            truthTag = list(tag)
        plot_waveform(wf, sign=sign, time=detX, time_cls=detX_cls, truth_time=truthX, truth_tag=truthTag, xrange=xrange, figsize=figsize)

def balanced_sampler(dataset, aux=False):
    sig_idx = []
    bkg_idx = []
    if aux is False:
        for i, (x, y) in enumerate(dataset):
            label = int(y.item())
            if label == 0:
                bkg_idx.append(i)
            if label == 1:
                sig_idx.append(i)
    else:
        for i, _t in enumerate(dataset.GetAuxTagList()):
            label = _t
            if label == 0:
                bkg_idx.append(i)
            if label == 1:
                sig_idx.append(i)

    # down sampling bkg
    bkg_idx = random.sample(bkg_idx, len(sig_idx))
    idx = sig_idx + bkg_idx
    
    print('number of balanced (sig, bkg) = ({}, {})'.format(len(sig_idx), len(bkg_idx)))

    sampler = SubsetRandomSampler(idx)
    return sampler

def label_sampler(dataset):
    idx = []
    for i, (X, y) in enumerate(dataset):
        if y.detach().numpy() >= 0:
            idx.append(i)

    sampler = SubsetRandomSampler(idx)
    return sampler
            

def peak_finding_derivative(wf, thr=1e-3, ma=2, step=1):
    def MA(wf, size=2):
        output = np.zeros(len(wf))
        K = np.ones(size)/size
        for i in range(size, len(wf)):
            for j in range(size):
                output[i] += K[j] * wf[i - j]
        return output

    def D1(wf, step=1):
        output = np.zeros(len(wf))
        for i in range(step, len(wf)):
            output[i] = wf[i] - wf[i-step]
            if output[i] < 0: output[i] = 0
        return output

    X = []
    if ma > 0: wf = MA(wf, size=ma)
    wf_d1 = D1(wf, step)
    wf_d2 = D1(wf_d1, step)
    wf_int = np.zeros(len(wf))
    for i in np.arange(1, len(wf)):
        if wf_d2[i] > 0:
            wf_int[i] = wf_int[i-1] + wf_d2[i]
        elif wf_int[i-1] > thr:
            X.append(i-1)

    return X

def semi_tag_method(wf, time, widths=np.arange(0.5, 10, 0.5), min_snr=3, amp_cut=-1., smear_sigma=0.):
    tag_list = []

    wf = np.array(wf)
    baseline_std = wf[0:50].std()
    peaks = signal.find_peaks_cwt(wf, widths=widths, min_snr=min_snr, 
                                  noise_perc=99, window_size=1000)
    peaks = [p for p in peaks if wf[int(p)] > amp_cut]

    for _t in time:
        idx = int(_t)
        if wf[idx] < 3*baseline_std:
            tag_list.append(0)
        else:
            matched = False
            for _p in peaks:
                if _t - _p >= -smear_sigma and _t - _p <= smear_sigma:
                    tag_list.append(1)
                    matched = True
                    break
            if not matched: tag_list.append(-1)

    return tag_list

def tag_method(wf, time, widths=np.arange(0.5, 10, 0.5), min_snr=3, amp_cut=-1., smear_sigma=1.):
    tag_list = []

    wf = np.array(wf)
    peaks = signal.find_peaks_cwt(wf, widths=widths, min_snr=min_snr, 
                                  noise_perc=99, window_size=1000)
    peaks = [p for p in peaks if wf[int(p)] > amp_cut]

    for _t in time:
        matched = False
        for _p in peaks:
            if _t - _p >= -smear_sigma and _t - _p <= smear_sigma:
                matched = True
                break
        if matched: 
            tag_list.append(1)
        else: 
            tag_list.append(0)

    return tag_list


