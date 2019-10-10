import time

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from casme import core, archs, criterion
from casme.train_utils import single_adjust_learning_rate, save_checkpoint, set_args
from casme.tasks.imagenet.utils import get_data_loaders

import zconf


@zconf.run_config
class RunConfiguration(zconf.RunConfig):

    # Experiment Setup
    train_json = zconf.attr(help='train_json path')
    val_json = zconf.attr(help='val_json path')
    output_path = zconf.attr(help='output_path')
    name = zconf.attr(default='random',
                      help='name used to build a path where the models and log are saved (default: random)')
    print_freq = zconf.attr(default=100, type=int,
                            help='print frequency (default: 100)')
    workers = zconf.attr(default=4, type=int,
                         help='number of data loading workers (default: 4)')

    epochs = zconf.attr(default=60, type=int,
                        help='number of total epochs to run')
    batch_size = zconf.attr(default=128, type=int,
                            help='mini-batch size (default: 128)')
    perc_of_training = zconf.attr(default=0.2, type=float,
                                  help='percent of training set seen in each epoch')
    lr = zconf.attr(default=0.001, type=float,
                    help='initial learning rate for classifier')
    lr_casme = zconf.attr(default=0.001, type=float,
                          help='initial learning rate for casme')
    lrde = zconf.attr(default=20, type=int,
                      help='how often is the learning rate decayed')
    momentum = zconf.attr(default=0.9, type=float,
                          help='momentum for classifier')
    weight_decay = zconf.attr(default=1e-4, type=float,
                              help='weight decay for both classifier and casme (default: 1e-4)')

    upsample = zconf.attr(default='nearest',
                          help='mode for final upsample layer in the decoder (default: nearest)')
    fixed_classifier = zconf.attr(action='store_true',
                                  help='train classifier')
    prob_historic = zconf.attr(default=0.5, type=float,
                               help='probability for evaluating historic model')
    save_freq = zconf.attr(default=1000, type=int,
                           help='frequency of model saving to history (in batches)')
    f_size = zconf.attr(default=30, type=int,
                        help='size of F set - maximal number of previous classifier iterations stored')
    lambda_r = zconf.attr(default=10, type=float,
                          help='regularization weight controlling mask size')
    lambda_tv = zconf.attr(default=None, type=float)

    mask_in_criterion = zconf.attr(default="none", type=str, help='crossentropy|kldivergence|none')
    mask_in_criterion_config = zconf.attr(default="", type=str, help='etc')
    mask_in_objective_direction = zconf.attr(default="maximize", help="maximize|minimize")
    mask_in_objective_type = zconf.attr(default="entropy", help="entropy|classification")
    mask_in_weight = zconf.attr(default=1.0, type=float)

    mask_out_criterion = zconf.attr(default="none", type=str, help='crossentropy|kldivergence|none')
    mask_out_criterion_config = zconf.attr(default="", type=str, help='etc')
    mask_out_objective_direction = zconf.attr(default="maximize", help="maximize|minimize")
    mask_out_objective_type = zconf.attr(default="entropy", help="entropy|classification")
    mask_out_weight = zconf.attr(default=1.0, type=float)

    reproduce = zconf.attr(default='',
                           help='reproducing paper results (F|L|FL|L100|L1000)')

    add_prob_layers = zconf.attr(action='store_true')
    prob_sample_low = zconf.attr(default=0.25, type=float)
    prob_sample_high = zconf.attr(default=0.75, type=float)
    prob_loss_func = zconf.attr(default="l1")

    # Placeholders
    casms_path = zconf.attr(default='')
    log_path = zconf.attr(default='')

    def _post_init(self):
        randomhash = ''.join(str(time.time()).split('.'))
        self.name = self.name + "___" + randomhash
        set_args(self)


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

    cudnn.benchmark = True

    train_loader, val_loader = get_data_loaders(
        train_json=args.train_json,
        val_json=args.val_json,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    mask_in_criterion = criterion.resolve_masker_criterion(
        masker_criterion_type=args.mask_in_criterion,
        masker_criterion_config=args.mask_in_criterion_config,
        lambda_r=args.lambda_r,
        lambda_tv=args.lambda_tv,
        add_prob_layers=args.add_prob_layers,
        prob_loss_func=args.prob_loss_func,
        objective_direction=args.mask_in_objective_direction,
        objective_type=args.mask_in_objective_type,
        device=device,
    )
    mask_out_criterion = criterion.resolve_masker_criterion(
        masker_criterion_type=args.mask_out_criterion,
        masker_criterion_config=args.mask_out_criterion_config,
        lambda_r=args.lambda_r,
        lambda_tv=args.lambda_tv,
        add_prob_layers=args.add_prob_layers,
        prob_loss_func=args.prob_loss_func,
        objective_direction=args.mask_out_objective_direction,
        objective_type=args.mask_out_objective_type,
        device=device,
    )

    casme_runner = core.CASMERunner(
        classifier=classifier,
        masker=masker,
        classifier_optimizer=classifier_optimizer,
        masker_optimizer=masker_optimizer,
        classifier_criterion=nn.CrossEntropyLoss(),
        mask_in_criterion=mask_in_criterion,
        mask_out_criterion=mask_out_criterion,
        fixed_classifier=args.fixed_classifier,
        perc_of_training=args.perc_of_training,
        prob_historic=args.prob_historic,
        save_freq=args.save_freq,
        zoo_size=args.f_size,
        image_normalization_mode=None,
        add_prob_layers=args.add_prob_layers,
        prob_sample_low=args.prob_sample_low,
        prob_sample_high=args.prob_sample_high,
        mask_in_weight=args.mask_in_weight,
        mask_out_weight=args.mask_out_weight,
        print_freq=args.print_freq,
        device=device,
    )

    # training loop
    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        single_adjust_learning_rate(
            optimizer=classifier_optimizer,
            epoch=epoch, lr=args.lr, lrde=args.lrde,
        )
        single_adjust_learning_rate(
            optimizer=masker_optimizer,
            epoch=epoch, lr=args.lr_casme, lrde=args.lrde,
        )

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
            'optimizer_classifier': classifier_optimizer.state_dict(),
            'optimizer_masker': masker_optimizer.state_dict(),
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
    main(args=RunConfiguration.run_cli_json_prepend())
