"""
Miscellanous Functions
"""

import sys
import re
import os
import shutil
import torch
from datetime import datetime
import logging
from subprocess import call
import shlex
from tensorboardX import SummaryWriter
import datasets
import numpy as np
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from config import cfg
import random


# Create unique output dir name based on non-default command line args
def make_exp_name(args, parser):
    exp_name = '{}-{}'.format(args.dataset[:4], args.arch[:])
    dict_args = vars(args)

    # sort so that we get a consistent directory name
    argnames = sorted(dict_args)
    ignorelist = ['date', 'exp', 'arch','prev_best_filepath', 'lr_schedule', 'max_cu_epoch', 'max_epoch',
                  'strict_bdr_cls', 'world_size', 'tb_path','best_record', 'test_mode', 'ckpt', 'coarse_boost_classes',
                  'crop_size', 'dist_url', 'syncbn', 'max_iter', 'color_aug', 'scale_max', 'scale_min', 'bs_mult',
                  'class_uniform_pct', 'class_uniform_tile']
    # build experiment name with non-default args
    for argname in argnames:
        if dict_args[argname] != parser.get_default(argname):
            if argname in ignorelist:
                continue
            if argname == 'snapshot':
                arg_str = 'PT'
                argname = ''
            elif argname == 'nosave':
                arg_str = ''
                argname=''
            elif argname == 'freeze_trunk':
                argname = ''
                arg_str = 'ft'
            elif argname == 'syncbn':
                argname = ''
                arg_str = 'sbn'
            elif argname == 'jointwtborder':
                argname = ''
                arg_str = 'rlx_loss'
            elif isinstance(dict_args[argname], bool):
                arg_str = 'T' if dict_args[argname] else 'F'
            else:
                arg_str = str(dict_args[argname])[:7]
            if argname is not '':
                exp_name += '_{}_{}'.format(str(argname), arg_str)
            else:
                exp_name += '_{}'.format(arg_str)
    # clean special chars out    exp_name = re.sub(r'[^A-Za-z0-9_\-]+', '', exp_name)
    return exp_name

def fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def save_log(prefix, output_dir, date_str, rank=0):
    fmt = '%(asctime)s.%(msecs)03d %(message)s'
    date_fmt = '%m-%d %H:%M:%S'
    filename = os.path.join(output_dir, prefix + '_' + date_str +'_rank_' + str(rank) +'.log')
    print("Logging :", filename)
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=date_fmt,
                        filename=filename, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    console.setFormatter(formatter)
    if rank == 0:
        logging.getLogger('').addHandler(console)
    else:
        fh = logging.FileHandler(filename)
        logging.getLogger('').addHandler(fh)



def prep_experiment(args, parser):
    """
    Make output directories, setup logging, Tensorboard, snapshot code.
    """
    ckpt_path = args.ckpt
    tb_path = args.tb_path
    exp_name = make_exp_name(args, parser)
    args.exp_path = os.path.join(ckpt_path, args.date, args.exp, str(datetime.now().strftime('%m_%d_%H')))
    args.tb_exp_path = os.path.join(tb_path, args.date, args.exp, str(datetime.now().strftime('%m_%d_%H')))
    args.ngpu = torch.cuda.device_count()
    args.date_str = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    args.best_record = {}
    # args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
    #                    'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
    args.last_record = {}
    if args.local_rank == 0:
        os.makedirs(args.exp_path, exist_ok=True)
        os.makedirs(args.tb_exp_path, exist_ok=True)
        save_log('log', args.exp_path, args.date_str, rank=args.local_rank)
        open(os.path.join(args.exp_path, args.date_str + '.txt'), 'w').write(
            str(args) + '\n\n')
        writer = SummaryWriter(log_dir=args.tb_exp_path, comment=args.tb_tag)
        return writer
    return None

def evaluate_eval_for_inference(hist, dataset=None):
    """
    Modified IOU mechanism for on-the-fly IOU calculations ( prevents memory overflow for
    large dataset) Only applies to eval/eval.py
    """
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

    print_evaluate_results(hist, iu, dataset=dataset)
    freq = hist.sum(axis=1) / hist.sum()
    mean_iu = np.nanmean(iu)
    logging.info('mean {}'.format(mean_iu))
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc



def evaluate_eval(args, net, optimizer, scheduler, val_loss, hist, dump_images, writer, epoch=0, dataset_name=None, dataset=None, curr_iter=0, optimizer_at=None, scheduler_at=None, save_pth=True):
    """
    Modified IOU mechanism for on-the-fly IOU calculations ( prevents memory overflow for
    large dataset) Only applies to eval/eval.py
    """
    if val_loss is not None and hist is not None:
        # axis 0: gt, axis 1: prediction
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

        print_evaluate_results(hist, iu, dataset_name=dataset_name, dataset=dataset)
        freq = hist.sum(axis=1) / hist.sum()
        mean_iu = np.nanmean(iu)
        logging.info('mean {}'.format(mean_iu))
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    else:
        mean_iu = 0

    if dataset_name not in args.last_record.keys():
        args.last_record[dataset_name] = {}

    if save_pth:
        # update latest snapshot
        if 'mean_iu' in args.last_record[dataset_name]:
            last_snapshot = 'last_{}_epoch_{}_mean-iu_{:.5f}.pth'.format(
                        dataset_name, args.last_record[dataset_name]['epoch'],
                        args.last_record[dataset_name]['mean_iu'])
            last_snapshot = os.path.join(args.exp_path, last_snapshot)
            try:
                os.remove(last_snapshot)
            except OSError:
                pass

        last_snapshot = 'last_{}_epoch_{}_mean-iu_{:.5f}.pth'.format(dataset_name, epoch, mean_iu)
        last_snapshot = os.path.join(args.exp_path, last_snapshot)
        args.last_record[dataset_name]['mean_iu'] = mean_iu
        args.last_record[dataset_name]['epoch'] = epoch

        torch.cuda.synchronize()

        if optimizer_at is not None:
            torch.save({
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'optimizer_at': optimizer_at.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scheduler_at': scheduler_at.state_dict(),
                'epoch': epoch,
                'mean_iu': mean_iu,
                'command': ' '.join(sys.argv[1:])
            }, last_snapshot)
        else:
            torch.save({
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'mean_iu': mean_iu,
                'command': ' '.join(sys.argv[1:])
            }, last_snapshot)

        if val_loss is not None and hist is not None:
            if dataset_name not in args.best_record.keys():
                args.best_record[dataset_name] = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                                                  'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
            # update best snapshot
            if mean_iu > args.best_record[dataset_name]['mean_iu'] :
                # remove old best snapshot
                if args.best_record[dataset_name]['epoch'] != -1:
                    best_snapshot = 'best_{}_epoch_{}_mean-iu_{:.5f}.pth'.format(
                        dataset_name, args.best_record[dataset_name]['epoch'],
                        args.best_record[dataset_name]['mean_iu'])

                    best_snapshot = os.path.join(args.exp_path, best_snapshot)
                    assert os.path.exists(best_snapshot), \
                        'cant find old snapshot {}'.format(best_snapshot)
                    os.remove(best_snapshot)

                # save new best
                args.best_record[dataset_name]['val_loss'] = val_loss.avg
                args.best_record[dataset_name]['epoch'] = epoch
                args.best_record[dataset_name]['acc'] = acc
                args.best_record[dataset_name]['acc_cls'] = acc_cls
                args.best_record[dataset_name]['mean_iu'] = mean_iu
                args.best_record[dataset_name]['fwavacc'] = fwavacc

                best_snapshot = 'best_{}_epoch_{}_mean-iu_{:.5f}.pth'.format(
                        dataset_name, args.best_record[dataset_name]['epoch'],
                        args.best_record[dataset_name]['mean_iu'])
                best_snapshot = os.path.join(args.exp_path, best_snapshot)
                shutil.copyfile(last_snapshot, best_snapshot)
        else:
            logging.info("Saved file to {}".format(last_snapshot))

    if val_loss is not None and hist is not None:
        logging.info('-' * 107)
        fmt_str = '[epoch %d], [dataset name %s], [val loss %.5f], [acc %.5f], [acc_cls %.5f], ' +\
                  '[mean_iu %.5f], [fwavacc %.5f]'
        logging.info(fmt_str % (epoch, dataset_name, val_loss.avg, acc, acc_cls, mean_iu, fwavacc))
        if save_pth:
            fmt_str = 'best record: [dataset name %s], [val loss %.5f], [acc %.5f], [acc_cls %.5f], ' +\
                      '[mean_iu %.5f], [fwavacc %.5f], [epoch %d], '
            logging.info(fmt_str % (dataset_name,
                                    args.best_record[dataset_name]['val_loss'], args.best_record[dataset_name]['acc'],
                                    args.best_record[dataset_name]['acc_cls'], args.best_record[dataset_name]['mean_iu'],
                                    args.best_record[dataset_name]['fwavacc'], args.best_record[dataset_name]['epoch']))
            logging.info('-' * 107)

        if writer:
            # tensorboard logging of validation phase metrics
            writer.add_scalar('{}/acc'.format(dataset_name), acc, curr_iter)
            writer.add_scalar('{}/acc_cls'.format(dataset_name), acc_cls, curr_iter)
            writer.add_scalar('{}/mean_iu'.format(dataset_name), mean_iu, curr_iter)
            writer.add_scalar('{}/val_loss'.format(dataset_name), val_loss.avg, curr_iter)





def print_evaluate_results(hist, iu, dataset_name=None, dataset=None):
    # fixme: Need to refactor this dict
    try:
        id2cat = dataset.id2cat
    except:
        id2cat = {i: i for i in range(datasets.num_classes)}
    iu_false_positive = hist.sum(axis=1) - np.diag(hist)
    iu_false_negative = hist.sum(axis=0) - np.diag(hist)
    iu_true_positive = np.diag(hist)

    logging.info('Dataset name: {}'.format(dataset_name))
    logging.info('IoU:')
    logging.info('label_id      label    iU    Precision Recall TP     FP    FN')
    for idx, i in enumerate(iu):
        # Format all of the strings:
        idx_string = "{:2d}".format(idx)
        class_name = "{:>13}".format(id2cat[idx]) if idx in id2cat else ''
        iu_string = '{:5.1f}'.format(i * 100)
        total_pixels = hist.sum()
        tp = '{:5.1f}'.format(100 * iu_true_positive[idx] / total_pixels)
        fp = '{:5.1f}'.format(
            iu_false_positive[idx] / iu_true_positive[idx])
        fn = '{:5.1f}'.format(iu_false_negative[idx] / iu_true_positive[idx])
        precision = '{:5.1f}'.format(
            iu_true_positive[idx] / (iu_true_positive[idx] + iu_false_positive[idx]))
        recall = '{:5.1f}'.format(
            iu_true_positive[idx] / (iu_true_positive[idx] + iu_false_negative[idx]))
        logging.info('{}    {}   {}  {}     {}  {}   {}   {}'.format(
            idx_string, class_name, iu_string, precision, recall, tp, fp, fn))




class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
