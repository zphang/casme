import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import stats
from train_utils import accuracy


class LogContainers:
    def __init__(self):
        self.batch_time = stats.TimeMeter()
        self.data_time = stats.TimeMeter()
        self.losses = stats.AverageMeter()
        self.acc = stats.AverageMeter()
        self.losses_m = stats.AverageMeter()
        self.acc_m = stats.AverageMeter()
        self.statistics = stats.StatisticsContainer()


class CASMERunner:
    def __init__(self, classifier, decoder,
                 lr, lr_casme, momentum, weight_decay, lambda_r,
                 fixed_classifier, pot, hp, smf, zoo_size,
                 add_prob_layers, prob_sample_low, prob_sample_high, prob_loss_func,
                 print_freq,
                 device, adversarial):
        self.classifier = classifier
        self.classifier_for_mask = None
        self.decoder = decoder
        self.lr = lr
        self.lr_casme = lr_casme
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lambda_r = lambda_r
        self.fixed_classifier = fixed_classifier
        self.pot = pot
        self.hp = hp
        self.smf = smf
        self.zoo_size = zoo_size
        self.add_prob_layers = add_prob_layers
        self.prob_sample_low = prob_sample_low
        self.prob_sample_high = prob_sample_high
        self.prob_loss_func = prob_loss_func
        self.print_freq = print_freq
        self.device = device
        self.adversarial = adversarial

        self.optimizer = {
            "classifier": torch.optim.SGD(
                classifier.parameters(), self.lr,
                momentum=self.momentum, weight_decay=self.weight_decay,
            ),
            "decoder": torch.optim.Adam(
                decoder.parameters(), self.lr_casme,
                weight_decay=self.weight_decay,
            ),
        }
        self.classifier_criterion = nn.CrossEntropyLoss().to(device)
        self.classifier_zoo = {}

    def train_or_eval(self, data_loader, is_train=False, epoch=None):
        log_containers = LogContainers()
        for i, (x, y) in enumerate(data_loader):
            x, y = x.to(self.device), y.to(self.device)
            log_containers.data_time.update()
            self.train_or_eval_batch(
                x, y, i,
                log_containers=log_containers, is_train=is_train,
            )
            # print log
            if i % self.print_freq == 0:
                if is_train:
                    print('Epoch: [{0}][{1}/{2}/{3}]\t'.format(
                        epoch, i, int(len(data_loader)*self.pot), len(data_loader)), end='')
                else:
                    print('Test: [{0}/{1}]\t'.format(i, len(data_loader)), end='')
                print('Time {lc.batch_time.avg:.3f} ({lc.batch_time.val:.3f})\t'
                      'Data {lc.data_time.avg:.3f} ({lc.data_time.val:.3f})\n'
                      'Loss(C) {lc.loss.avg:.4f} ({lc.loss.val:.4f})\t'
                      'Prec@1(C) {lc.acc.avg:.3f} ({lc.acc.val:.3f})\n'
                      'Loss(M) {lc.loss_m.avg:.4f} ({lc.loss_m.val:.4f})\t'
                      'Prec@1(M) {lc.acc_m.avg:.3f} ({lc.acc_m.val:.3f})\t'.format(lc=log_containers))
                log_containers.statistics.print_out()
                print()

        if not is_train:
            print(' * Prec@1 {lc.acc.avg:.3f} Prec@1(M) {lc.acc_m.avg:.3f} '.format(lc=log_containers))
            log_containers.statistics.print_out()

        return {
            'acc': str(log_containers.acc.avg),
            'acc_m': str(log_containers.acc_m.avg),
            **log_containers.statistics.get_dictionary()
        }

    def train_or_eval_batch(self, x, y, i, log_containers, is_train=False):
        self.models_mode(is_train)

        if self.add_prob_layers:
            use_p = torch.Tensor(x.shape[0])\
                .uniform_(self.prob_sample_low, self.prob_sample_high)\
                .to(self.device)
        else:
            use_p = None

        # compute classifier prediction on the original images and get inner layers
        with torch.set_grad_enabled(is_train and (not self.fixed_classifier)):
            y_hat, layers = self.classifier(x, return_intermediate=True)
            classifier_loss = self.classifier_criterion(y_hat, y)

        # update metrics
        # losses.update(classifier_loss.item(), x.size(0))
        # acc.update(accuracy(y_hat.detach(), y, topk=(1,))[0].item(), x.size(0))

        # update classifier - compute gradient and do SGD step for clean image, save classifier
        if is_train and (not self.fixed_classifier):
            self.optimizer['classifier'].zero_grad()
            classifier_loss.backward()
            self.optimizer['classifier'].step()

            # save classifier (needed only if previous iterations are used i.e. args.hp > 0)
            if self.hp > 0 and ((i % self.smf == -1 % self.smf) or len(self.classifier_zoo) < 1):
                print('Current iteration is saving, will be used in the future. ', end='', flush=True)
                if len(self.classifier_zoo) < self.zoo_size:
                    index = len(self.classifier_zoo)
                else:
                    index = random.randint(0, len(self.classifier_zoo) - 1)
                state_dict = self.classifier.state_dict()
                self.classifier_zoo[index] = {}
                for p in state_dict:
                    self.classifier_zoo[index][p] = state_dict[p].cpu()
                print('There are {0} iterations stored.'.format(len(self.classifier_zoo)), flush=True)

        # detach inner layers to make them be features for decoder
        layers = [l.detach() for l in layers]

        with torch.set_grad_enabled(is_train):
            # compute mask and masked input
            mask = self.decoder(layers, use_p=use_p)
            input_m = x*(1-mask)

            # update statistics
            log_containers.statistics.update(mask)

            # randomly select classifier to be evaluated on masked image and compute output
            if (not is_train) or self.fixed_classifier or (random.random() > self.hp):
                output_m = self.classifier(input_m)
                update_classifier = not self.fixed_classifier
            else:
                if self.classifier_for_mask is None:
                    self.classifier_for_mask = copy.deepcopy(self.classifier)
                index = random.randint(0, len(self.classifier_zoo) - 1)
                self.classifier_for_mask.load_state_dict(self.classifier_zoo[index])
                self.classifier_for_mask.eval()

                output_m = self.classifier_for_mask(input_m)
                update_classifier = False

            classifier_loss_m = self.classifier_criterion(output_m, y)

            # update metrics
            log_containers.losses_m.update(classifier_loss_m.item(), x.size(0))
            log_containers.acc_m.update(accuracy(output_m.detach(), y, topk=(1,))[0].item(), x.size(0))

        if is_train:
            # update classifier - compute gradient, do SGD step for masked image
            if update_classifier:
                self.optimizer['classifier'].zero_grad()
                classifier_loss_m.backward(retain_graph=True)
                self.optimizer['classifier'].step()

            # regularization for casme
            _, max_indexes = y_hat.detach().max(1)
            _, max_indexes_m = output_m.detach().max(1)
            correct_on_clean = y.eq(max_indexes)
            mistaken_on_masked = y.ne(max_indexes_m)
            nontrivially_confused = (correct_on_clean + mistaken_on_masked).eq(2).float()

            mask_mean = mask.mean(dim=3).mean(dim=2)
            if self.add_prob_layers:
                # adjust to minimize deviation from p
                mask_mean = (mask_mean - use_p)
                if self.prob_loss_func == "l1":
                    mask_mean = mask_mean.abs()
                elif self.prob_loss_func == "l2":
                    mask_mean = mask_mean.pow(2)
                else:
                    raise KeyError(self.prob_loss_func)

            # apply regularization loss only on non-trivially confused images
            regularization = -self.lambda_r * F.relu(nontrivially_confused - mask_mean).mean()

            # main loss for casme
            if self.adversarial:
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
            self.optimizer['decoder'].zero_grad()
            casme_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 10)
            self.optimizer['decoder'].step()

        # measure elapsed time
        log_containers.batch_time.update()

    def models_mode(self, train):
        if train:
            self.decoder.train()
            if self.fixed_classifier:
                self.classifier.eval()
            else:
                self.classifier.train()
        else:
            self.decoder.eval()
            self.classifier.eval()
