import copy
import datetime as dt
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import stats
from train_utils import accuracy
from PConv_Keras.libs.util import random_mask
from pytorch_inpainting_with_partial_conv.loss import gram_matrix, total_variation_loss
import numpy as np

import cv2
import scipy


class PrintLogger:
    def log(self, *args, **kwargs):
        print(*args, **kwargs)


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

        self.infiller_total_loss = stats.AverageMeter()
        self.infiller_loss = stats.AverageMeter()
        self.infiller_reg = stats.AverageMeter()
        self.infiller_hole = stats.AverageMeter()
        self.infiller_valid = stats.AverageMeter()
        self.infiller_perceptual = stats.AverageMeter()
        self.infiller_style_out = stats.AverageMeter()
        self.infiller_style_comp = stats.AverageMeter()
        self.infiller_tv = stats.AverageMeter()

        self.generator_total_loss = stats.AverageMeter()
        self.discriminator_total_loss = stats.AverageMeter()
        self.discriminator_real_loss = stats.AverageMeter()
        self.discriminator_fake_loss = stats.AverageMeter()


def default_apply_mask_func(x, mask):
    return x * (1 - mask)

def apply_inverted_mask_func(x, mask):
    return x * mask

def invert_mask_func(mask):
    return 1 - mask
    
def apply_uniform_random_value_mask_func(x, mask):
    # for mask, return image infilled with random value between 0 and 1?
    # zero_mean_std_1?
    # x[mask==1] = np.random.random((mask==1).shape)
    return None

def default_infill_func(x, mask, generated_image):
    # for mask, return image infilled with generated_image
    # mask = 1, non-mask = 0
    # x[mask==1] = generated_image[mask==1]
    return generated_image * (1 - mask) + x * mask # TODO: change the order?
    #return None

class DiscriminatorCriterion(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss()
        self.device = device

    def forward(self, real_images_logits, gen_images_logits):
        batch_size = real_images_logits.shape[0] # TODO


        valid = torch.tensor([1.]*batch_size, device=self.device, requires_grad=False, dtype=torch.float32).view(-1,1)
        fake = torch.tensor([0.]*batch_size, device=self.device, requires_grad=False, dtype=torch.float32).view(-1,1)
        generator_loss = self.adversarial_loss(gen_images_logits, valid)
        real_loss = self.adversarial_loss(real_images_logits, valid)
        fake_loss = self.adversarial_loss(gen_images_logits, fake)
        metadata = {
            'generator_loss': generator_loss,
            'real_loss': real_loss,
            'fake_loss': fake_loss,
        }
        discriminator_loss =  (real_loss + fake_loss) / 2
        return generator_loss, discriminator_loss, metadata

        

class InfillerCriterion(nn.Module):
    def __init__(self, model_type, style_loss_multiplier):
        super().__init__()
        self.l1 = nn.L1Loss()
        # TODO: take mask function as parameters
        # TODO: take model hyperparameters (how many layers to use fur perceptual loss, loss mutlipliers)
        # TODO: use different criterion depending on models
        self.model_type = model_type
        self.style_loss_multiplier = style_loss_multiplier

    def forward(self, x, mask, generated_image, infilled_image, layers, generated_layers, infilled_layers, dilated_boundaries):
        # x here is ground-truth

        # assuming 1 means background
        hole = self.l1(default_apply_mask_func(x, mask), default_apply_mask_func(generated_image, mask))
        valid = self.l1(apply_inverted_mask_func(x, mask), apply_inverted_mask_func(generated_image, mask))

        # layers = feature of gt image

        perceptual_loss = 0# self.l1(layers[i], infilled_layers[i])
        style_out_loss = 0# self.l1(gram_matrix(layers[i]), gram_matrix(infilled_layers[i]))
        style_comp_loss = 0# self.l1(gram_matrix(layers[i]), gram_matrix(infilled_layers[i]))
        for i in range(3):
            #TODO: normalize
            perceptual_loss += self.l1(layers[i], infilled_layers[i]) + self.l1(layers[i], generated_layers[i])
            style_out_loss += self.l1(gram_matrix(layers[i]), gram_matrix(generated_layers[i])) / (layers[i].shape[1] * layers[i].shape[2] * layers[i].shape[3])
            style_comp_loss += self.l1(gram_matrix(layers[i]), gram_matrix(infilled_layers[i])) / (layers[i].shape[1] * layers[i].shape[2] * layers[i].shape[3])
            #print(layers[i].shape)

        # TODO: Is this loss wrong? if it only backprops to generated bits... ... Try implementing my own.
        # TODO: try using total_variation_loss on dilated boundary region only... but still, incorrect boundary
        tv = total_variation_loss(apply_inverted_mask_func(infilled_image, dilated_boundaries))
        regularization = 0
        loss = 6 * hole + valid + 0.05 * perceptual_loss + self.style_loss_multiplier*(style_out_loss+style_comp_loss) + 0.1*tv
        infiller_loss = loss + regularization
        metadata = {
            "hole": hole,
            "valid": valid,
            "perceptual_loss": perceptual_loss,
            "style_out_loss": style_out_loss,
            "style_comp_loss": style_comp_loss,
            "tv": tv,
            "loss": loss,
            "regularization": regularization,
        }
        return infiller_loss, metadata

class MaskerCriterion(nn.Module):
    def __init__(self, lambda_r, add_prob_layers, prob_loss_func, adversarial, device):
        super().__init__()
        self.lambda_r = lambda_r
        self.add_prob_layers = add_prob_layers
        self.prob_loss_func = prob_loss_func
        self.adversarial = adversarial
        self.device = device

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
    def __init__(self, lambda_r, class_weights, add_prob_layers, prob_loss_func, config, device):
        super().__init__()
        self.lambda_r = lambda_r
        self.add_prob_layers = add_prob_layers
        self.prob_loss_func = prob_loss_func
        self.config = json.loads(config)
        self.device = device

        self.class_weights = torch.Tensor(class_weights).to(device)
        inverse_class_weights = 1 / self.class_weights
        self.prior = (inverse_class_weights / inverse_class_weights.sum())

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
        regularization = -self.lambda_r * F.relu(nontrivially_confused - mask_mean)

        # main loss for casme
        log_prob = F.log_softmax(y_hat_from_masked_x, dim=1)
        if self.config["kl"] == "forward":
            # - sum: p_i log(q_i)
            kl = - (self.prior * log_prob).sum(dim=1)
        elif self.config["kl"] == "backward":
            log_prior = torch.log(self.prior)
            # - sum: q_i log(p_i / q_i)
            kl = - (y_hat_from_masked_x_prob * (log_prior - log_prob)).sum(dim=1)
        else:
            raise KeyError(self.config["kl"])

        # apply main loss only when original images are correctly classified
        if self.config["loss_on_coc"]:
            kl = kl * correct_on_clean.float()
        else:
            kl = kl

        if "nothing_class" in self.config:
            keep_filter = (y != self.config["nothing_class"]).float()
            kl = kl * keep_filter

        if self.config["apply_class_weight"]:
            sample_weights = torch.index_select(self.class_weights, dim=0, index=y)
            regularization = regularization * sample_weights
            kl = kl * sample_weights

        if "nothing_class_reg_weight" in self.config:
            reg_weight = (
                (y == self.config["nothing_class"]).float()
                * (self.config["nothing_class_reg_weight"] - 1)
                + 1
            )
            regularization = regularization * reg_weight

        regularization = regularization.mean()
        loss = kl.mean()

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
                 logger=None,
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
        self.logger = logger if logger is not None else PrintLogger()
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
                    self.logger.log('Epoch: [{0}][{1}/{2}/{3}]\t'.format(
                        epoch, i, int(len(data_loader)*self.perc_of_training), len(data_loader)), end='')
                else:
                    self.logger.log('Test: [{0}][{1}/{2}]\t'.format(epoch, i, len(data_loader)), end='')
                self.logger.log('Time {lc.batch_time.avg:.3f} ({lc.batch_time.val:.3f})\t'
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
                self.logger.log('{:%Y-%m-%d %H:%M:%S}'.format(dt.datetime.now()))
                self.logger.log()

        if not is_train:
            self.logger.log(' * Prec@1 {lc.acc.avg:.3f} Prec@1(M) {lc.acc_m.avg:.3f} '.format(lc=log_containers))
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
                self.logger.log('Current iteration is saving, will be used in the future. ', end='', flush=True)
                if len(self.classifier_zoo) < self.zoo_size:
                    index = len(self.classifier_zoo)
                else:
                    index = random.randint(0, len(self.classifier_zoo) - 1)
                state_dict = self.classifier.state_dict()
                self.classifier_zoo[index] = {}
                for p in state_dict:
                    self.classifier_zoo[index][p] = state_dict[p].cpu()
                self.logger.log('There are {0} iterations stored.'.format(len(self.classifier_zoo)), flush=True)

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

class InfillerRunner:
    def __init__(self,
                 classifier, infiller,
                 infiller_optimizer,
                 infiller_criterion,
                 discriminator,
                 discriminator_optimizer,
                 discriminator_criterion,
                 save_freq,
                 print_freq,
                 perc_of_training,
                 device,
                 logger=None,
                 ):
        self.classifier = classifier
        self.infiller = infiller
        self.infiller_optimizer = infiller_optimizer
        self.infiller_criterion = infiller_criterion
        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_criterion = discriminator_criterion #TODO

        self.perc_of_training = perc_of_training
        self.save_freq = save_freq
        self.print_freq = print_freq
        self.device = device
        self.logger = logger if logger is not None else PrintLogger()
        if self.infiller.model_type == "ciGAN":
            # TODO: ciGAN uses 0 as non-mask, and mask will be converted so. 
            self.apply_mask_func = apply_uniform_random_value_mask_func
        else:
            self.apply_mask_func = apply_inverted_mask_func
        self.infill_func = default_infill_func

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
                is_printlogger = 'PrintLogger' in str(type(self.logger))
                if is_train:
                    if is_printlogger:
                        self.logger.log('Epoch: [{0}][{1}/{2}/{3}]\t'.format(
                            epoch, i, int(len(data_loader)*self.perc_of_training), len(data_loader)), end='')
                    else:
                        self.logger.log('Epoch: [{0}][{1}/{2}/{3}]\t'.format(
                            epoch, i, int(len(data_loader)*self.perc_of_training), len(data_loader)))
                else:
                    if is_printlogger:
                        self.logger.log('Test: [{0}][{1}/{2}]\t'.format(epoch, i, len(data_loader)), end='')
                    else:
                        self.logger.log('Test: [{0}][{1}/{2}]\t'.format(epoch, i, len(data_loader)))
                # TODO: print out other parts of losses
                self.logger.log('Time {lc.batch_time.avg:.3f} ({lc.batch_time.val:.3f})\t'
                                'Data {lc.data_time.avg:.3f} ({lc.data_time.val:.3f})\n'
                                'ITLoss(M) {lc.infiller_total_loss.avg:.4f} ({lc.infiller_total_loss.val:.4f})\t'
                                'ILoss(M) {lc.infiller_loss.avg:.4f} ({lc.infiller_loss.val:.4f})\n'
                                'IHole(M) {lc.infiller_hole.avg:.4f} ({lc.infiller_hole.val:.4f})\n'
                                'IValid(M) {lc.infiller_valid.avg:.4f} ({lc.infiller_valid.val:.4f})\n'
                                'IPerceptual(M) {lc.infiller_perceptual.avg:.4f} ({lc.infiller_perceptual.val:.4f})\n'
                                'IStyleOut(M) {lc.infiller_style_out.avg:.4f} ({lc.infiller_style_out.val:.4f})\n'
                                'IStyleComp(M) {lc.infiller_style_comp.avg:.4f} ({lc.infiller_style_comp.val:.4f})\n'
                                'ITV(M) {lc.infiller_tv.avg:.4f} ({lc.infiller_tv.val:.4f})\n'
                                #'IReg(M) {lc.infiller_reg.avg:.4f} ({lc.infiller_reg.val:.4f})\n'
                                ''.format(lc=log_containers))

                if self.discriminator != None:
                    self.logger.log('DReal(M) {lc.discriminator_real_loss.avg:.4f} ({lc.discriminator_real_loss.val:.4f})\n'
                                'DFake(M) {lc.discriminator_fake_loss.avg:.4f} ({lc.discriminator_fake_loss.val:.4f})\n'
                                'DTotal(M) {lc.discriminator_total_loss.avg:.4f} ({lc.discriminator_total_loss.val:.4f})\n'
                                'GTotal(M) {lc.generator_total_loss.avg:.4f} ({lc.generator_total_loss.val:.4f})\n'
                                #'IReg(M) {lc.infiller_reg.avg:.4f} ({lc.infiller_reg.val:.4f})\n'
                                ''.format(lc=log_containers))
                log_containers.statistics.print_out()
                self.logger.log('{:%Y-%m-%d %H:%M:%S}'.format(dt.datetime.now()))
                if is_printlogger:
                    self.logger.log()


    def train_or_eval_batch(self, x, y, i, log_containers, is_train=False):

        # TODO: use data loader to parallalize mask generation
        # TODO: mask will be inverted if generated by masker... handle it later
        mask = np.stack([random_mask(x.shape[2], x.shape[3], x.shape[1]) for _ in range(x.shape[0])], axis=0)
        # generated mask here uses 1 as non-mask
        if self.infiller.model_type == "ciGAN":
            mask = invert_mask_func(mask) # ciGAN uses 0 as non-mask
        #mask = mask.reshape(x.shape)
        # I should create FloatTensor and send to GPU and turn into cuda tensor by doing so
        dilated_boundaries = []
        for onemask in mask:
            # TODO: findCoutours does not work when num_channels is not 1. I can just fix it to 1 and 
            # create 3 identical copies based on one channel
            _, contours, _ = cv2.findContours(onemask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            boundary = np.zeros_like(onemask)
            cv2.drawContours(boundary, contours, -1, 1, 1)
            dilated_boundary = scipy.ndimage.morphology.binary_dilation(boundary, iterations=1)
            dilated_boundaries.append(dilated_boundary)
        dilated_boundaries = np.stack(dilated_boundaries, axis=0).astype(float)

        mask = torch.cuda.FloatTensor(mask).to(self.device)
        mask = mask.permute(0,3,1,2)

        dilated_boundaries = torch.cuda.FloatTensor(dilated_boundaries).to(self.device)
        dilated_boundaries = dilated_boundaries.permute(0,3,1,2)

        masked_x = self.apply_mask_func(x=x, mask=mask)

        with torch.set_grad_enabled(is_train):
            # compute mask and masked input
            if self.infiller.model_type == 'pconv_infogan':
                generated_image, generated_mask = self.infiller(masked_x, mask, y)
            else:
                generated_image, generated_mask = self.infiller(masked_x, mask)
            infilled_image = self.infill_func(x=x, mask=mask, generated_image=generated_image)

        # compute classifier prediction on the original images and get inner layers
        with torch.set_grad_enabled(is_train):#False):
            y_hat, layers = self.classifier(x, return_intermediate=True)
            generated_y_hat, generated_layers = self.classifier(generated_image, return_intermediate=True)
            infilled_y_hat, infilled_layers = self.classifier(infilled_image, return_intermediate=True)

        # TODO: decide which ones to detach
        #layers = [l.detach() for l in layers]
        #generated_layers = [l.detach() for l in generated_layers]
        #infilled_layers = [l.detach() for l in infilled_layers]


        if is_train:
            infiller_total_loss, infiller_loss_metadata = self.infiller_criterion(
                x=x, mask=mask, generated_image=generated_image, infilled_image=infilled_image,
                layers=layers, generated_layers=generated_layers, infilled_layers=infilled_layers,
                dilated_boundaries=dilated_boundaries
            )

            # update infiller using pconv loss
            #TODO: try using both pconv, gen, discriminator loss
            if self.discriminator == None:
                self.infiller_optimizer.zero_grad()
                infiller_total_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.infiller.parameters(), 10)
                self.infiller_optimizer.step()


            #if self.discriminator != None:
            else:
                with torch.set_grad_enabled(is_train):#False):
                    gen_images_logits = self.discriminator(infilled_layers[-1], self.classifier.resnet)
                    # TODO: Use infilled or generated images?
                    #gen_images_logits = self.discriminator(generated_layers[-1], self.classifier.resnet)
                    real_images_logits = self.discriminator(layers[-1], self.classifier.resnet)
                generator_total_loss, discriminator_total_loss, discriminator_loss_metadata = self.discriminator_criterion(
                    real_images_logits=real_images_logits, 
                    gen_images_logits=gen_images_logits
                )

                # update generator
                # TODO: don't do this update step for infiller if using pconv loss function
                self.infiller_optimizer.zero_grad() 
                # TODO: Try adding both loss and backproping once
                loss_on_infiller = generator_total_loss + infiller_total_loss
                loss_on_infiller.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.infiller.parameters(), 10) 
                self.infiller_optimizer.step() 
                # update discriminator
                self.discriminator_optimizer.zero_grad()
                discriminator_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 10)
                self.discriminator_optimizer.step()
                log_containers.discriminator_total_loss.update(discriminator_total_loss.item(), x.size(0))
                log_containers.generator_total_loss.update(generator_total_loss.item(), x.size(0))
                log_containers.discriminator_real_loss.update(discriminator_loss_metadata["real_loss"], x.size(0))
                log_containers.discriminator_fake_loss.update(discriminator_loss_metadata['fake_loss'], x.size(0))
    
            # TODO: print out other parts of losses
            log_containers.infiller_total_loss.update(infiller_total_loss.item(), x.size(0))
            log_containers.infiller_loss.update(infiller_loss_metadata["loss"].item(), x.size(0))
            log_containers.infiller_hole.update(infiller_loss_metadata["hole"].item(), x.size(0))
            log_containers.infiller_valid.update(infiller_loss_metadata["valid"].item(), x.size(0))
            log_containers.infiller_perceptual.update(infiller_loss_metadata["perceptual_loss"].item(), x.size(0))
            log_containers.infiller_style_out.update(infiller_loss_metadata["style_out_loss"].item(), x.size(0))
            log_containers.infiller_style_comp.update(infiller_loss_metadata["style_comp_loss"].item(), x.size(0))
            log_containers.infiller_tv.update(infiller_loss_metadata["tv"].item(), x.size(0))
            #log_containers.infiller_reg.update(infiller_loss_metadata["regularization"].item(), x.size(0))

        # measure elapsed time
        log_containers.batch_time.update()
        #print("batch done")

    def models_mode(self, train):
        self.classifier.eval()
        if train:
            self.infiller.train()
        else:
            self.infiller.eval()
