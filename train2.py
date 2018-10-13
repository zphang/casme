import argparse
import os
import time

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import archs
import casme_core
from train_utils import adjust_learning_rate, save_checkpoint, set_args


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data',
                    help='path to dataset')
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
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate for classifier')
parser.add_argument('--lr-casme', '--learning-rate-casme', default=0.001, type=float,
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
parser.add_argument('--prob-sample-low', default=0.25, type=float)
parser.add_argument('--prob-sample-high', default=0.75, type=float)
parser.add_argument('--prob-loss-func', default="l1")

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

set_args(args)

F_k = {}


def main():
    global args

    # create models and optimizers
    print("=> creating models...")
    classifier = archs.resnet50shared(pretrained=True).to(device)
    decoder = archs.decoder(
        final_upsample_mode=args.upsample,
        add_prob_layers=args.add_prob_layers,
    ).to(device)

    optimizer = {
        "classifier": torch.optim.SGD(
            classifier.parameters(), args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay,
        ),
        "decoder": torch.optim.Adam(
            decoder.parameters(), args.lr_casme,
            weight_decay=args.weight_decay,
        ),
    }

    cudnn.benchmark = True

    # data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=False, sampler=None,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=False,
    )

    casme_runner = casme_core.CASMERunner(
        classifier=classifier,
        decoder=decoder,
        lr=args.lr,
        lr_casme=args.lr_casme,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        lambda_r=args.lambda_r,
        fixed_classifier=args.fixed_classifier,
        pot=args.pot,
        hp=args.hp,
        smf=args.smf,
        zoo_size=args.zoo_size,
        add_prob_layers=args.add_prob_layers,
        prob_sample_low=args.prob_sample_low,
        prob_sample_high=args.prob_sample_high,
        prob_loss_func=args.prob_loss_func,
        print_freq=args.print_freq,
        device=device,
        adversarial=args.adversarial,
    )

    # training loop
    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        tr_s = casme_runner.train_or_eval(
            data_loader=train_loader,
            is_train=True,
            epoch=epoch,
        )

        # evaluate on validation set
        val_s = casme_runner.train_or_eval(
            data_loader=val_loader,
            is_train=False,
            epoch=epoch
        )

        # save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict_classifier': classifier.state_dict(),
            'state_dict_decoder': decoder.state_dict(),
            'optimizer_classifier': optimizer['classifier'].state_dict(),
            'optimizer_decoder': optimizer['decoder'].state_dict(),
            'args': args,
        }, args)

        # log
        with open(args.log_path, 'a') as f:
            f.write(str(epoch + 1) + ' ' + str(time.time() - epoch_start_time) + ' ' +
                    tr_s['acc'] + ' ' + val_s['acc'] + ' ' + tr_s['acc_m'] + ' ' + val_s['acc_m'] + ' ' +
                    tr_s['avg_mask'] + ' ' + val_s['avg_mask'] + ' ' +
                    tr_s['std_mask'] + ' ' + val_s['std_mask'] + ' ' +
                    tr_s['entropy'] + ' ' + val_s['entropy'] + ' ' +
                    tr_s['tv'] + ' ' + val_s['tv'] + '\n')


if __name__ == '__main__':
    main()
