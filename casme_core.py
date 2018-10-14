import copy
import datetime as dt
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
        self.masker_total_loss = stats.AverageMeter()

        self.masker_loss = stats.AverageMeter()
        self.masker_reg = stats.AverageMeter()
        self.correct_on_clean = stats.AverageMeter()
        self.mistaken_on_masked = stats.AverageMeter()
        self.nontrivially_confused = stats.AverageMeter()


def default_apply_mask_func(x, mask):
    return x * (1 - mask)


class MaskerCriterion(nn.Module):
    def __init__(self, lambda_r, add_prob_layers, prob_loss_func, adversarial):
        super().__init__()
        self.lambda_r = lambda_r
        self.add_prob_layers = add_prob_layers
        self.prob_loss_func = prob_loss_func
        self.adversarial = adversarial

    def forward(self,
                mask, y_hat, y_hat_from_masked_x, y,
                classifier_loss_from_masked_x, use_p):
        _, max_indexes = y_hat.detach().max(1)
        _, max_indexes_on_masked_x = y_hat_from_masked_x.detach().max(1)
        correct_on_clean = y.eq(max_indexes)
        mistaken_on_masked = y.ne(max_indexes_on_masked_x)
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
        """
            nontrivially_confused = 
                    1 if relevant, else 0
            nontrivially_confused - mask_mean = 
                    if relevant: 1-mask_mean (small if large mask)
                    else: something less than 0
            F.relu(...) = 
                    if relevant: 1-mask_mean (small if large mask)
            - (...) = 
                    if relevant: mask_mean (+const)
        """

        # main loss for casme
        if self.adversarial:
            loss = -classifier_loss_from_masked_x
        else:
            log_prob = F.log_softmax(y_hat_from_masked_x, 1)
            prob = log_prob.exp()
            negative_entropy = (log_prob * prob).sum(1)
            # apply main loss only when original images are correctly classified
            negative_entropy_correct = negative_entropy * correct_on_clean.float()
            loss = negative_entropy_correct.mean()

        masker_loss = loss + regularization
        metadata = {
            "correct_on_clean": correct_on_clean.float().mean(),
            "mistaken_on_masked": mistaken_on_masked.float().mean(),
            "nontrivially_confused": nontrivially_confused.float().mean(),
            "loss": loss,
            "regularization": regularization,
        }
        return masker_loss, metadata


class MaskerPriorCriterion(nn.Module):
    def __init__(self, prior, lambda_r, add_prob_layers, prob_loss_func, config):
        super().__init__()
        self.prior = prior
        self.lambda_r = lambda_r
        self.add_prob_layers = add_prob_layers
        self.prob_loss_func = prob_loss_func

        import json
        self.config = json.loads(config)

    def forward(self,
                mask, y_hat, y_hat_from_masked_x, y,
                classifier_loss_from_masked_x, use_p):
        y_hat_prob = F.softmax(y_hat, dim=1)
        y_hat_from_masked_x_prob = F.softmax(y_hat_from_masked_x, dim=1)

        # Should this be / or - ?
        if self.config["prior"] == "subtract":
            y_hat_is_over_prior = y_hat_prob - self.prior
            y_hat_from_masked_x_prob_over_prior = y_hat_from_masked_x_prob - self.prior
        elif self.config["prior"] == "divide":
            y_hat_is_over_prior = y_hat_prob / self.prior
            y_hat_from_masked_x_prob_over_prior = y_hat_from_masked_x_prob / self.prior
        else:
            raise KeyError(self.config["prior"])

        _, max_indexes = y_hat_is_over_prior.detach().max(1)
        _, max_indexes_on_masked_x = y_hat_from_masked_x_prob_over_prior.detach().max(1)

        correct_on_clean = y.eq(max_indexes)
        mistaken_on_masked = y.ne(max_indexes_on_masked_x)
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
        """
            nontrivially_confused = 
                    1 if relevant, else 0
            nontrivially_confused - mask_mean = 
                    if relevant: 1-mask_mean (small if large mask)
                    else: something less than 0
            F.relu(...) = 
                    if relevant: 1-mask_mean (small if large mask)
            - (...) = 
                    if relevant: mask_mean (+const)
        """

        # main loss for casme
        if self.config["kl"] == "forward":
            log_prob = F.log_softmax(y_hat_from_masked_x, dim=1)
            kl = - (self.prior * log_prob).sum(dim=1)
        elif self.config["kl"] == "backward":
            log_prior = torch.log(self.prior)
            """
            negative_kl = (y_hat_from_masked_x_prob
                           * (log_prior - F.log_softmax(y_hat_from_masked_x))).sum(dim=1)
            """
            kl = - (y_hat_from_masked_x_prob
                    * (log_prior - F.log_softmax(y_hat_from_masked_x))).sum(dim=1)
        else:
            raise KeyError(self.config["kl"])

        # apply main loss only when original images are correctly classified
        kl_correct = kl * correct_on_clean.float()
        loss = kl_correct.mean()

        masker_loss = loss + regularization
        metadata = {
            "correct_on_clean": correct_on_clean.float().mean(),
            "mistaken_on_masked": mistaken_on_masked.float().mean(),
            "nontrivially_confused": nontrivially_confused.float().mean(),
            "loss": loss,
            "regularization": regularization,
        }
        return masker_loss, metadata


class CASMERunner:
    def __init__(self,
                 classifier, masker,
                 classifier_optimizer, masker_optimizer,
                 classifier_criterion, masker_criterion,
                 fixed_classifier, perc_of_training, prob_historic, save_freq, zoo_size,
                 add_prob_layers, prob_sample_low, prob_sample_high,
                 print_freq,
                 device,
                 apply_mask_func=default_apply_mask_func,
                 ):
        self.classifier = classifier
        self.classifier_for_mask = None
        self.masker = masker
        self.classifier_optimizer = classifier_optimizer
        self.masker_optimizer = masker_optimizer
        self.classifier_criterion = classifier_criterion.to(device)
        self.masker_criterion = masker_criterion.to(device)

        self.fixed_classifier = fixed_classifier
        self.perc_of_training = perc_of_training
        self.prob_historic = prob_historic
        self.save_freq = save_freq
        self.zoo_size = zoo_size
        self.add_prob_layers = add_prob_layers
        self.prob_sample_low = prob_sample_low
        self.prob_sample_high = prob_sample_high
        self.print_freq = print_freq
        self.device = device
        self.apply_mask_func = apply_mask_func

        self.classifier_zoo = {}

    def train_or_eval(self, data_loader, is_train=False, epoch=None):
        log_containers = LogContainers()
        self.models_mode(is_train)
        for i, (x, y) in enumerate(data_loader):
            if is_train and i > len(data_loader) * self.perc_of_training:
                break
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
                        epoch, i, int(len(data_loader)*self.perc_of_training), len(data_loader)), end='')
                else:
                    print('Test: [{0}/{1}]\t'.format(i, len(data_loader)), end='')
                print('Time {lc.batch_time.avg:.3f} ({lc.batch_time.val:.3f})\t'
                      'Data {lc.data_time.avg:.3f} ({lc.data_time.val:.3f})\n'
                      'Loss(C) {lc.losses.avg:.4f} ({lc.losses.val:.4f})\t'
                      'Prec@1(C) {lc.acc.avg:.3f} ({lc.acc.val:.3f})\n'
                      'Loss(M) {lc.losses_m.avg:.4f} ({lc.losses_m.val:.4f})\t'
                      'Prec@1(M) {lc.acc_m.avg:.3f} ({lc.acc_m.val:.3f})\n'
                      'MTLoss(M) {lc.masker_total_loss.avg:.4f} ({lc.masker_total_loss.val:.4f})\t'
                      'MLoss(M) {lc.masker_loss.avg:.4f} ({lc.masker_loss.val:.4f})\t'
                      'MReg(M) {lc.masker_reg.avg:.4f} ({lc.masker_reg.val:.4f})\n'
                      'CoC {lc.correct_on_clean.avg:.3f} ({lc.correct_on_clean.val:.3f})\t'
                      'MoM {lc.mistaken_on_masked.avg:.3f} ({lc.mistaken_on_masked.val:.3f})\t'
                      'NC {lc.nontrivially_confused.avg:.3f} ({lc.nontrivially_confused.val:.3f})\t'
                      ''.format(lc=log_containers))
                log_containers.statistics.print_out()
                print('{:%Y-%m-%d %H:%M:%S}'.format(dt.datetime.now()))
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
        log_containers.losses.update(classifier_loss.item(), x.size(0))
        log_containers.acc.update(accuracy(y_hat.detach(), y, topk=(1,))[0].item(), x.size(0))

        # update classifier - compute gradient and do SGD step for clean image, save classifier
        if is_train and (not self.fixed_classifier):
            self.classifier_optimizer.zero_grad()
            classifier_loss.backward()
            self.classifier_optimizer.step()

            # save classifier (needed only if previous iterations are used i.e. args.hp > 0)
            if self.prob_historic > 0 \
                    and ((i % self.save_freq == -1 % self.save_freq) or len(self.classifier_zoo) < 1):
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
            mask = self.masker(layers, use_p=use_p)
            masked_x = self.apply_mask_func(x=x, mask=mask)

            # update statistics
            log_containers.statistics.update(mask)

            # randomly select classifier to be evaluated on masked image and compute output
            if (not is_train) or self.fixed_classifier or (random.random() > self.prob_historic):
                y_hat_from_masked_x = self.classifier(masked_x)
                update_classifier = not self.fixed_classifier
            else:
                if self.classifier_for_mask is None:
                    self.classifier_for_mask = copy.deepcopy(self.classifier)
                index = random.randint(0, len(self.classifier_zoo) - 1)
                self.classifier_for_mask.load_state_dict(self.classifier_zoo[index])
                self.classifier_for_mask.eval()
                y_hat_from_masked_x = self.classifier_for_mask(masked_x)
                update_classifier = False

            classifier_loss_from_masked_x = self.classifier_criterion(y_hat_from_masked_x, y)

            # update metrics
            log_containers.losses_m.update(classifier_loss_from_masked_x.item(), x.size(0))
            log_containers.acc_m.update(accuracy(
                y_hat_from_masked_x.detach(), y, topk=(1,))[0].item(), x.size(0))

        if is_train:
            # update classifier - compute gradient, do SGD step for masked image
            if update_classifier:
                self.classifier_optimizer.zero_grad()
                classifier_loss_from_masked_x.backward(retain_graph=True)
                self.classifier_optimizer.step()

            masker_total_loss, masker_loss_metadata = self.masker_criterion(
                mask=mask, y_hat=y_hat, y_hat_from_masked_x=y_hat_from_masked_x, y=y,
                classifier_loss_from_masked_x=classifier_loss_from_masked_x, use_p=use_p,
            )

            # update casme - compute gradient, do SGD step
            self.masker_optimizer.zero_grad()
            masker_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.masker.parameters(), 10)
            self.masker_optimizer.step()
            log_containers.masker_total_loss.update(masker_total_loss.item(), x.size(0))
            log_containers.masker_loss.update(masker_loss_metadata["loss"].item(), x.size(0))
            log_containers.masker_reg.update(masker_loss_metadata["regularization"].item(), x.size(0))
            log_containers.correct_on_clean.update(masker_loss_metadata["correct_on_clean"].item(), x.size(0))
            log_containers.mistaken_on_masked.update(masker_loss_metadata["mistaken_on_masked"].item(), x.size(0))
            log_containers.nontrivially_confused.update(masker_loss_metadata["nontrivially_confused"].item(), x.size(0))

        # measure elapsed time
        log_containers.batch_time.update()

    def models_mode(self, train):
        if train:
            self.masker.train()
            if self.fixed_classifier:
                self.classifier.eval()
            else:
                self.classifier.train()
        else:
            self.masker.eval()
            self.classifier.eval()
