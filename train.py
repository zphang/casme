import argparse
import os
import random
import time

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms

import archs
from stats import AverageMeter, StatisticsContainer
from train_utils import accuracy, adjust_learning_rate, save_checkpoint, set_args

from utilities.containers import DataSplits
import utilities.logger as logging_module
import patch_classification.patch_model.torch.patch_data_torch as patch_data
import patch_classification.patch_model.torch.models_torch as models
from torch.utils.data import DataLoader
DATA_CONFIG = {
    "num_parts": {
        "training": {'malignant': 30, 'benign': 30, 'unknown': 100, 'outside': 100},
        "validation": {'malignant': 1, 'benign': 1, 'unknown': 1, 'outside': 1},
        "test": {'malignant': 1, 'benign': 1, 'unknown': 1, 'outside': 1},
    },
    "samples_per_epoch": {
        "training": {'malignant': 100, 'benign': 200, 'unknown': 51700, 'outside': 48000},
        # "training": {'malignant': 100 , 'benign': 200, 'unknown': 100, 'outside': 100},
        "validation": {'malignant': -1, 'benign': -1, 'unknown': -1, 'outside': -1},
        "test": {'malignant': -1, 'benign': -1, 'unknown': -1, 'outside': -1},
    },
    "parts_to_load": {
        "training": {'malignant': 1, 'benign': 1, 'unknown': 6, 'outside': 5},
        # "training": {'malignant': 1, 'benign': 1, 'unknown': 1, 'outside': 1},
        "validation": {'malignant': 1, 'benign': 1, 'unknown': 1, 'outside': 1},
        "test": {'malignant': 1, 'benign': 1, 'unknown': 1, 'outside': 1},
    },
    "paths": {
        "training": "/gpfs/data/geraslab/Nan/data/segmentation_patches/patches_by_class_0907/training/",
        "validation": "/gpfs/data/geraslab/Nan/data/segmentation_patches/patches_by_class_0907/validation/",
        "test": "/gpfs/data/geraslab/Nan/data/segmentation_patches/patches_by_class_0907/test/",
    },
    "label_list": ["malignant", "benign", "unknown", "outside"],
}


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--casms-path', default='',
                    help='path to models that generate masks')
parser.add_argument('--log-path', default='',
                    help='directory for logs')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('-n', '--name', default=randomhash+'random',
                    help='name used to build a path where the models and log are saved (default: random)')
parser.add_argument('--print-freq', default=100, type=int,
                    help='print frequency (default: 100)')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')

parser.add_argument('--epochs', default=60, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--pot', default=0.2, type=float,
                    help='percent of training set seen in each epoch')
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    help='initial learning rate for classifier')
parser.add_argument('--lr-casme', '--learning-rate-casme', default=0.00001, type=float,
                    help='initial learning rate for casme')
parser.add_argument('--lrde', default=20, type=int,
                    help='how often is the learning rate decayed')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum for classifier')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay for both classifier and casme (default: 1e-4)')

parser.add_argument('--upsample', default='nearest',
                    help='mode for final upsample layer in the decoder (default: nearest)')
parser.add_argument('--fixed-classifier', action='store_true',
                    help='train classifier')
parser.add_argument('--hp', default=0.5, type=float,
                    help='probability for evaluating historic model')
parser.add_argument('--smf', default=1000, type=int,
                    help='frequency of model saving to history (in batches)')
parser.add_argument('--f-size', default=30, type=int,
                    help='size of F set - maximal number of previous classifier iterations stored')
parser.add_argument('--lambda-r', default=10, type=float,
                    help='regularization weight controlling mask size')
parser.add_argument('--adversarial', action='store_true',
                    help='adversarial training uses classification loss instead of entropy')

parser.add_argument('--reproduce', default='',
                    help='reproducing paper results (F|L|FL|L100|L1000)')

parser.add_argument('--add-prob-layers', action='store_true')
parser.add_argument('--prob-sample-low', default=0.25)
parser.add_argument('--prob-sample-high', default=0.25)
parser.add_argument('--prob-loss-func', default="l1")

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

set_args(args)

F_k = {}


MODEL_PARAMETERS = {
    "number_of_classes": 3,
    "num_filters": 16,
    "input_channels": 1,
    "first_layer_kernel_size": 7,
    "first_layer_conv_stride": 2,
    "first_pool_size": 3,
    "first_pool_stride": 2,
    "blocks_per_layer_list": [3, 4, 6, 3],
    "block_strides_list": [1, 2, 2, 2],
    "block_fn": "bottleneck",
}


def main():
    global args

    # create models and optimizers
    print("=> creating models...")
    classifier = models.PatchResNetV2(
        parameters=MODEL_PARAMETERS,
    ).to(device)
    state_dict = torch.load(
        "/gpfs/data/geraslab/zphang/working/09.18.patch/FirstRun_1537315201.9966803/epoch_600/model.p")
    classifier.load_state_dict(state_dict["model"])
    decoder = archs.decoder(
        final_upsample_mode=args.upsample,
        add_prob_layers=args.add_prob_layers,
    ).to(device)

    optimizer = {
        "classifier": torch.optim.Adam(
            classifier.parameters(), args.lr,
            # weight_decay=args.weight_decay,
        ),
        "decoder": torch.optim.Adam(
            decoder.parameters(), args.lr_casme,
            weight_decay=args.weight_decay,
        ),
    }
    logger = logging_module.PrintLogger()

    cudnn.benchmark = True

    # data loading code
    datasets = DataSplits(
        train=patch_data.PickleDataset("training", data_config=DATA_CONFIG, logger=logger),
        val=patch_data.PickleDataset("validation", data_config=DATA_CONFIG, logger=logger),
        test=None,
    )
    dataloaders = DataSplits(
        train=DataLoader(datasets.train, args.batch_size),
        val=DataLoader(datasets.val, args.batch_size),
        test=None,
    )
    datasets.train.new_epoch()
    datasets.val.new_epoch()

    # training loop
    for epoch in range(args.epochs):

        epoch_start_time = time.time()

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        tr_s = train_or_eval(dataloaders.train, classifier, decoder,
                             train=True, optimizer=optimizer, epoch=epoch)

        # evaluate on validation set
        val_s = train_or_eval(dataloaders.val, classifier, decoder)

        # save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict_classifier': classifier.state_dict(),
            'state_dict_decoder': decoder.state_dict(),
            'optimizer_classifier': optimizer['classifier'].state_dict(),
            'optimizer_decoder': optimizer['decoder'].state_dict(),
            'args': args,
        }, args)
        datasets.train.new_epoch()

        # log
        with open(args.log_path, 'a') as f:
            f.write(str(epoch + 1) + ' ' + str(time.time() - epoch_start_time) + ' ' + 
                    tr_s['acc'] + ' ' + val_s['acc'] + ' ' + tr_s['acc_m'] + ' ' + val_s['acc_m'] + ' ' + 
                    tr_s['avg_mask'] + ' ' + val_s['avg_mask'] + ' ' + 
                    tr_s['std_mask'] + ' ' + val_s['std_mask'] + ' ' + 
                    tr_s['entropy'] + ' ' + val_s['entropy'] + ' ' + 
                    tr_s['tv'] + ' ' + val_s['tv'] + '\n')


def train_or_eval(data_loader, classifier, decoder, train=False, optimizer=None, epoch=None):
    # initialize all metric used
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    losses_m = AverageMeter()
    acc_m = AverageMeter()
    statistics = StatisticsContainer()

    classifier_criterion = nn.CrossEntropyLoss().to(device)

    # switch to train mode if needed
    if train:
        decoder.train()
        if args.fixed_classifier:
            classifier.eval()
        else:
            classifier.train()
    else:
        decoder.eval()  
        classifier.eval()

    # data loop
    end = time.time()
    for i, (input_, target) in enumerate(data_loader):
        if train and i > len(data_loader)*args.pot:
            break

        # measure data loading time
        data_time.update(time.time() - end)

        # move input and target on the device
        input_, target = input_.to(device), target.to(device)

        if args.add_prob_layers:
            use_p = torch.Tensor(input_.shape[0]).uniform_(0.25, 0.75).to(device)
        else:
            use_p = None

        # compute classifier prediction on the original images and get inner layers
        with torch.set_grad_enabled(train and (not args.fixed_classifier)):
            output, layers = classifier(input_, return_intermediate=True)

            # Pad hack
            layers[0] = F.pad(layers[0], (1, 1, 1, 1))
            layers[1] = F.pad(layers[1], (1, 1, 1, 1))
            layers[2] = F.pad(layers[2], (1, 0, 1, 0))
            # End Pad hack

            classifier_loss = classifier_criterion(output, target)

        # update metrics
        losses.update(classifier_loss.item(), input_.size(0))
        acc.update(accuracy(output.detach(), target, topk=(1,))[0].item(), input_.size(0))

        # update classifier - compute gradient and do SGD step for clean image, save classifier
        if train and (not args.fixed_classifier):
            optimizer['classifier'].zero_grad()
            classifier_loss.backward()
            optimizer['classifier'].step()

            # save classifier (needed only if previous iterations are used i.e. args.hp > 0)
            global F_k
            if args.hp > 0 and ((i % args.smf == -1 % args.smf) or len(F_k) < 1):
                print('Current iteration is saving, will be used in the future. ', end='', flush=True)
                if len(F_k) < args.f_size:
                    index = len(F_k) 
                else:
                    index = random.randint(0, len(F_k) - 1)
                state_dict = classifier.state_dict()
                F_k[index] = {}
                for p in state_dict:
                    F_k[index][p] = state_dict[p].cpu()
                print('There are {0} iterations stored.'.format(len(F_k)), flush=True)

        # detach inner layers to make them be features for decoder
        layers = [l.detach() for l in layers]

        with torch.set_grad_enabled(train):
            # compute mask and masked input
            mask = decoder(layers, use_p=use_p)
            input_m = input_*(1-mask)

            # update statistics
            statistics.update(mask)

            # randomly select classifier to be evaluated on masked image and compute output
            if (not train) or args.fixed_classifier or (random.random() > args.hp):
                output_m = classifier(input_m)
                update_classifier = not args.fixed_classifier
            else:
                try:
                    confuser
                except NameError:
                    import copy
                    confuser = copy.deepcopy(classifier)
                index = random.randint(0, len(F_k) - 1)
                confuser.load_state_dict(F_k[index])
                confuser.eval()

                output_m = confuser(input_m)
                update_classifier = False

            classifier_loss_m = classifier_criterion(output_m, target)

            # update metrics
            losses_m.update(classifier_loss_m.item(), input_.size(0))
            acc_m.update(accuracy(output_m.detach(), target, topk=(1,))[0].item(), input_.size(0))

        if train:
            # update classifier - compute gradient, do SGD step for masked image
            if update_classifier:
                optimizer['classifier'].zero_grad()
                classifier_loss_m.backward(retain_graph=True)
                optimizer['classifier'].step()

            # regularization for casme
            _, max_indexes = output.detach().max(1)
            _, max_indexes_m = output_m.detach().max(1)
            correct_on_clean = target.eq(max_indexes)
            mistaken_on_masked = target.ne(max_indexes_m)
            nontrivially_confused = (correct_on_clean + mistaken_on_masked).eq(2).float()

            mask_mean = F.avg_pool2d(mask, 256, stride=1).squeeze()
            if args.add_prob_layers:
                # adjust to minimize deviation from p
                mask_mean = (mask_mean - use_p)
                if args.prob_loss_func == "l1":
                    mask_mean = mask_mean.abs()
                elif args.prob_loss_func == "l2":
                    mask_mean = mask_mean.pow(2)
                else:
                    raise KeyError(args.prob_loss_func)

            # apply regularization loss only on non-trivially confused images
            regularization = -args.lambda_r * F.relu(nontrivially_confused - mask_mean).mean()

            # main loss for casme
            if args.adversarial:
                loss = -classifier_loss_m
            else:
                log_prob = F.log_softmax(output_m, 1)
                prob = log_prob.exp()
                negative_entropy = (log_prob * prob).sum(1)
                # apply main loss only when original images are correctly classified
                negative_entropy_correct = negative_entropy * correct_on_clean.float()
                loss = negative_entropy_correct.mean()

            casme_loss = loss + regularization

            # update casme - compute gradient, do SGD step
            optimizer['decoder'].zero_grad()
            casme_loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 10)
            optimizer['decoder'].step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print log
        if i % args.print_freq == 0:
            if train:
                print('Epoch: [{0}][{1}/{2}/{3}]\t'.format(
                    epoch, i, int(len(data_loader)*args.pot), len(data_loader)), end='')
            else:
                print('Test: [{0}/{1}]\t'.format(i, len(data_loader)), end='')
            print('Time {batch_time.avg:.3f} ({batch_time.val:.3f})\t'
                  'Data {data_time.avg:.3f} ({data_time.val:.3f})\n'
                  'Loss(C) {loss.avg:.4f} ({loss.val:.4f})\t'
                  'Prec@1(C) {acc.avg:.3f} ({acc.val:.3f})\n'
                  'Loss(M) {loss_m.avg:.4f} ({loss_m.val:.4f})\t'
                  'Prec@1(M) {acc_m.avg:.3f} ({acc_m.val:.3f})\t'.format(
                      batch_time=batch_time, data_time=data_time,
                      loss=losses, acc=acc, loss_m=losses_m, acc_m=acc_m), flush=True)
            statistics.printOut()
            print()

    if not train:
        print(' * Prec@1 {acc.avg:.3f} Prec@1(M) {acc_m.avg:.3f} '.format(acc=acc, acc_m=acc_m))
        statistics.printOut()

    return {
        'acc': str(acc.avg),
        'acc_m': str(acc_m.avg),
        **statistics.getDictionary()
    }


if __name__ == '__main__':
    main()
