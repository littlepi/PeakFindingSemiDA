#!/usr/bin/env python

import torch
import argparse
import ast
import array
import numpy as np
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.cluster import AgglomerativeClustering

import ROOT
from ROOT import TFile
from ROOT import vector

from lib.dataset import WaveformSliceDataset
from lib.model import DnnModel, RnnModel
from lib.utility import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def process_waveform(infile, intree, outfile, mfile, start=0, length=-1, sign=-1, 
    threshold=0.5, with_tag=False, clustering_cut=-1.):
    f = TFile(infile)
    t = f.Get(intree)
    wf = ROOT.std.vector['double'](0)
    time = ROOT.std.vector['double'](0)
    tag = ROOT.std.vector['int'](0)
    t.SetBranchAddress('wf_i', wf)
    if with_tag:
        t.SetBranchAddress('time', time)
        t.SetBranchAddress('tag', tag)

    fout = TFile(outfile, 'recreate')
    tout = t.CloneTree(0)
    ncount = array.array('i', [-1])
    xcount = vector['double'](0)
    ncount_cls = array.array('i', [-1])
    xcount_cls = vector['double'](0)
    tout.Branch('ncount', ncount, 'ncount/I')
    tout.Branch('xcount', xcount)
    tout.Branch('ncount_cls', ncount_cls, 'ncount_cls/I')
    tout.Branch('xcount_cls', xcount_cls)

    wf_slice_dataset = WaveformSliceDataset(infile, intree, start, length, nleft=5, nright=9, with_tag=with_tag, tag_method=None, sign=sign)
    evtno2idx_dict = wf_slice_dataset.GetEventNoToIndexDict()

    model = DnnModel(embedding=True)
    model.load_state_dict(torch.load(mfile))
    model.eval()

    num_processed = 0
    labels = []
    predictions = []
    for evtno in tqdm(evtno2idx_dict, desc='Making predictions'):
        t.GetEntry(evtno)
        ncount[0] = 0
        xcount.clear()
        ncount_cls[0] = 0
        xcount_cls.clear()
        # det_time = []
        truth_time = None
        truth_tag = None
        if with_tag:
            truth_time = [t for i, t in enumerate(time) if tag[i] > 0]
            truth_tag = [t for i, t in enumerate(tag) if tag[i] > 0]

        index_list = evtno2idx_dict[evtno]
        for idx in index_list:
            with torch.no_grad():
                x, y = wf_slice_dataset[idx]
                x = x.to(device)
                pred, _ = model(x)
                if pred.item() > threshold:
                    ncount[0] += 1
                    xcount.push_back(wf_slice_dataset.GetWaveformSliceTime(idx))
                    # det_time.append(wf_slice_dataset.GetWaveformSliceTime(idx))
                if with_tag:
                    labels.append(y.item())
                    predictions.append(pred.item())
                # if idx == 0:
                #     writer.add_graph(model, x)
        
        if clustering_cut > 0:
            detX = np.array(xcount)
            cluster_id = AgglomerativeClustering(n_clusters=None, distance_threshold=clustering_cut).fit_predict(detX.reshape(-1, 1))
            cluster_map = {k:[] for k in cluster_id}
            for idx, k in enumerate(cluster_id):
                cluster_map[k].append(detX[idx])

            for k in cluster_map:
                xcount_cls.push_back(np.mean(cluster_map[k]))
            ncount_cls[0] = len(cluster_map)

        if num_processed < 10:
            if num_processed == 0:
                os.system('mkdir -p ./results/fig/')
            plot_waveform(wf, sign=sign, time=xcount, truth_time=truth_time, truth_tag=truth_tag, filename='./results/fig/wf_{}.png'.format(num_processed))
        tout.Fill()
        num_processed += 1

    if with_tag:
        fpr, tpr, thr = roc_curve(labels, predictions)
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title('ROC Curve')
        plt.savefig('./results/fig/roc.png')

        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(thr, fpr, color='b')
        ax.set_xlabel('THR')
        ax.set_ylabel('FPR')
        ax.set_xlim((0, 1))
        ax.semilogy()
        ax.grid(True)
        ax2 = ax.twinx()
        ax2.plot(thr, tpr, color='r')
        ax2.set_ylabel('TPR')
        plt.savefig('./results/fig/thr.png')

        def find_threshold(cut = 0.01):
            index = 0
            for i, f in enumerate(fpr):
                if fpr[i] < cut and fpr[i+1] > cut:
                    index = i
            return thr[index], tpr[index]
        print('FPR = 0.01: {}'.format(find_threshold(0.01)))
        print('FPR = 0.001: {}'.format(find_threshold(0.001)))

    fout.WriteTObject(tout)
    # writer.add_pr_curve('PRC', np.array(labels), np.array(predictions), 0)
    # writer.close()

def main(args):
    # writer = None

    process_waveform(
        args.input_file, args.input_tree, 
        args.output_file, args.model_file, 
        args.event_range[0], args.event_range[1], args.wf_sign, 
        args.prob_cut,
        with_tag=True if args.with_tag > 0 else False,
        clustering_cut=args.clustering_cut)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type=str, default='./dataset/161718_ch5_sig.root')
    parser.add_argument('--input_tree', type=str, default='signal')
    parser.add_argument('--output_file', type=str, default='./results/testbeam_pred.root')
    parser.add_argument('--wf_sign', type=int, default=-1)
    parser.add_argument('--model_file', type=str, default='./results/model_ot_usv.pth')
    parser.add_argument('--event_range', type=lambda x:ast.literal_eval(x), default='(5000, 10)')
    # parser.add_argument('--num_test', type=int, default=100, help='Number of events for test sample')
    parser.add_argument('--prob_cut', type=float, default=0.95, help='Probability cut value')
    # parser.add_argument('--with_tag', default=True, choices=('True', 'False'))
    parser.add_argument('--with_tag', type=int, default=1)
    parser.add_argument('--clustering_cut', type=float, default=-1.)

    args = parser.parse_args()
    main(args)
