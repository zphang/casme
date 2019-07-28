import argparse
import os
import time

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from casme import core, archs, criterion
from casme.train_utils import single_adjust_learning_rate, save_checkpoint, set_args


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create models and optimizers
    print("=> creating models...")
    classifier = archs.resnet50shared(pretrained=True).to(device)
    masker = archs.Masker(
        in_channels=[64, 256, 512, 1024, 2048],
        out_channel=64,
        final_upsample_mode=args.upsample,
        add_prob_layers=args.add_prob_layers,
    ).to(device)
    classifier_optimizer = torch.optim.SGD(
        classifier.parameters(), args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay,
    )
    masker_optimizer = torch.optim.Adam(
        masker.parameters(), args.lr_casme,
        weight_decay=args.weight_decay,
    )
    infiller = archs.get_infiller(args.infiller_model).to(device)
    infiller_optimizer = torch.optim.Adam(
        infiller.parameters(), args.lr_infiller,
        weight_decay=args.weight_decay,
    )

    if args.casme_load_path:
        print("=> loading from {}".format(args.casme_load_path))
        loaded = torch.load(args.casme_load_path)
        classifier.load_state_dict(loaded["state_dict_classifier"])
        masker.load_state_dict(loaded["state_dict_masker"])

    cudnn.benchmark = True

    # data loading code
    print("=> setting up data loaders...")
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

    if args.masker_criterion == "crossentropy":
        masker_criterion = criterion.MaskerCriterion(
            lambda_r=args.lambda_r,
            add_prob_layers=args.add_prob_layers,
            prob_loss_func=args.prob_loss_func,
            adversarial=args.adversarial,
            device=device
        )
    elif args.masker_criterion == "kldivergence":
        masker_criterion = criterion.MaskerPriorCriterion(
            lambda_r=args.lambda_r,
            class_weights=[1 / 1000] * 1000,
            add_prob_layers=args.add_prob_layers,
            prob_loss_func=args.prob_loss_func,
            config=args.masker_criterion_config,
            device=device,
        )
    else:
        raise KeyError(args.masker_criterion)

    infiller_criterion = nn.MSELoss()

    casme_runner = core.InfillerCASMERunner(
        classifier=classifier,
        masker=masker,
        infiller=infiller,
        classifier_optimizer=classifier_optimizer,
        masker_optimizer=masker_optimizer,
        infiller_optimizer=infiller_optimizer,
        train_infiller=archs.should_train_infiller(args.infiller_model),
        shuffle_infiller_masks=args.shuffle_infiller_masks,
        classifier_criterion=nn.CrossEntropyLoss(),
        masker_criterion=masker_criterion,
        infiller_criterion=infiller_criterion,
        fixed_classifier=args.fixed_classifier,
        perc_of_training=args.perc_of_training,
        prob_historic=args.prob_historic,
        save_freq=args.save_freq,
        zoo_size=args.f_size,
        image_normalization_mode=None,
        add_prob_layers=args.add_prob_layers,
        prob_sample_low=args.prob_sample_low,
        prob_sample_high=args.prob_sample_high,
        print_freq=args.print_freq,
        device=device,
    )

    # training loop
    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        single_adjust_learning_rate(classifier_optimizer, epoch, args.lr, args.lrde)
        single_adjust_learning_rate(masker_optimizer, epoch, args.lr_casme, args.lrde)
        single_adjust_learning_rate(infiller_optimizer, epoch, args.lr_infiller, args.lrde)

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
            'state_dict_masker': masker.state_dict(),
            'state_dict_infiller': infiller.state_dict(),
            'optimizer_classifier': classifier_optimizer.state_dict(),
            'optimizer_masker': masker_optimizer.state_dict(),
            'optimizer_infiller': infiller_optimizer.state_dict(),
            'args': args,
            'infiller_model': args.infiller_model,
        }, args)

        # log
        with open(args.log_path, 'a') as f:
            f.write(str(epoch + 1) + ' ' + str(time.time() - epoch_start_time) + ' ' +
                    tr_s['acc'] + ' ' + val_s['acc'] + ' ' + tr_s['acc_m'] + ' ' + val_s['acc_m'] + ' ' +
                    tr_s['avg_mask'] + ' ' + val_s['avg_mask'] + ' ' +
                    tr_s['std_mask'] + ' ' + val_s['std_mask'] + ' ' +
                    tr_s['entropy'] + ' ' + val_s['entropy'] + ' ' +
                    tr_s['tv'] + ' ' + val_s['tv'] + '\n')


def get_args(*raw_args):
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data',
                        help='path to dataset')
    parser.add_argument('--casms-path', default='',
                        help='path to models that generate masks')
    parser.add_argument('--log-path', default='',
                        help='directory for logs')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('-n', '--name', default=randomhash + 'random',
                        help='name used to build a path where the models and log are saved (default: random)')
    parser.add_argument('--print-freq', default=100, type=int,
                        help='print frequency (default: 100)')
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--epochs', default=60, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('--perc-of-training', default=0.2, type=float,
                        help='percent of training set seen in each epoch')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        help='initial learning rate for classifier')
    parser.add_argument('--lr-casme', '--learning-rate-casme', default=0.001, type=float,
                        help='initial learning rate for casme')
    parser.add_argument('--lr-infiller', '--learning-rate-infiller', default=0.001, type=float,
                        help='initial learning rate for infiller')
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
    parser.add_argument('--prob-historic', default=0.5, type=float,
                        help='probability for evaluating historic model')
    parser.add_argument('--save-freq', default=1000, type=int,
                        help='frequency of model saving to history (in batches)')
    parser.add_argument('--f-size', default=30, type=int,
                        help='size of F set - maximal number of previous classifier iterations stored')
    parser.add_argument('--lambda-r', default=10, type=float,
                        help='regularization weight controlling mask size')
    parser.add_argument('--adversarial', action='store_true',
                        help='adversarial training uses classification loss instead of entropy')
    parser.add_argument('--masker-criterion', default="crossentropy", type=str,
                        help='crossentropy|kldivergence')
    parser.add_argument('--masker-criterion-config', default="", type=str,
                        help='etc')

    parser.add_argument('--reproduce', default='',
                        help='reproducing paper results (F|L|FL|L100|L1000)')

    parser.add_argument('--add-prob-layers', action='store_true')
    parser.add_argument('--prob-sample-low', default=0.25, type=float)
    parser.add_argument('--prob-sample-high', default=0.75, type=float)
    parser.add_argument('--prob-loss-func', default="l1")
    parser.add_argument('--casme-load-path', default=None, type=str)
    parser.add_argument('--infiller-model', default="cnn", type=str)
    parser.add_argument('--shuffle-infiller-masks', action="store_true")

    args = parser.parse_args(*raw_args)
    if not args.log_path:
        args.log_path = args.casms_path
    set_args(args)


    return args


if __name__ == '__main__':
    main(get_args())
