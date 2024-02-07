import torch
import argparse
import ast
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# from torchinfo import summary
import random
from torchmetrics.classification import BinaryAccuracy
import matplotlib.pyplot as plt
import ROOT

from lib.dataset import WaveformSliceDataset
from lib.model import DnnModel, RnnModel, DeepJDOT, DeepSemiJDOT
from lib.utility import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    if args.input_type == 0:
        source_dataset = WaveformSliceDataset(args.source_file, 'sim', args.source_range[0], args.source_range[1], nleft=5, nright=9, with_tag=True)
        target_dataset = WaveformSliceDataset(args.target_file, 'signal', args.target_range[0], args.target_range[1], nleft=5, nright=9, sign=-1, 
                                              with_tag=False, semi_mode=True, semi_method=semi_tag_method, tag_method=tag_method)
        test_dataset = WaveformSliceDataset(args.test_file, 'sim', args.test_range[0], args.test_range[1], nleft=5, nright=9, with_tag=True)
    else:
        source_dataset = torch.load(args.source_dataset_file)
        target_dataset = torch.load(args.target_dataset_file)
        test_dataset = torch.load(args.test_dataset_file)
    source_dataloader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=balanced_sampler(source_dataset))
    target_dataloader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=balanced_sampler(target_dataset, aux=True))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=balanced_sampler(test_dataset))

    ### DeepJDOT model
    model = DnnModel(embedding=True)
    model = DeepSemiJDOT(model, init_lr=args.init_lr, lambda_s=args.lambda_s, lambda_t=args.lambda_t, lambda_tl=args.lambda_tl, alpha=args.alpha, 
                         ot_enable=False if args.ot_enable == 0 else True, ot_class_weight=args.ot_class_weight)
    model.fit(source_dataloader, target_dataloader, test_dataloader=test_dataloader, epoch_size=args.num_epoch, verbose=args.verbose)
    model.evaluate(test_dataloader)
    model.save(args.model_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_type', type=int, default=0)
    parser.add_argument('--source_file', type=str, default='./dataset/testbeam_sim.root')
    parser.add_argument('--target_file', type=str, default='./dataset/161718_ch5_sig.root')
    parser.add_argument('--test_file', type=str, default='./dataset/testbeam_sim.root')
    parser.add_argument('--source_dataset_file', type=str, default='./dataset/testbeam_source_dataset.pt')
    parser.add_argument('--target_dataset_file', type=str, default='./dataset/testbeam_target_dataset.pt')
    parser.add_argument('--test_dataset_file', type=str, default='./dataset/testbeam_source_test_dataset.pt')
    parser.add_argument('--source_range', type=lambda x:ast.literal_eval(x), default='(0, 1000)')
    parser.add_argument('--target_range', type=lambda x:ast.literal_eval(x), default='(0, 1000)')
    parser.add_argument('--test_range', type=lambda x:ast.literal_eval(x), default='(1000, 200)')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--init_lr', type=float, default=1e-2)
    parser.add_argument('--lambda_s', type=float, default=1.0)
    parser.add_argument('--lambda_t', type=float, default=1.0)
    parser.add_argument('--lambda_tl', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--ot_enable', type=int, default=1)
    parser.add_argument('--ot_class_weight', type=lambda x:ast.literal_eval(x), default='(0.5, 0.5)')
    parser.add_argument('--model_file', type=str, default='./results/model_ot_usv.pth')
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--verbose', type=int, default=0)

    args = parser.parse_args()
    main(args)