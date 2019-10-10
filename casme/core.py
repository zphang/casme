import copy
import datetime as dt
import random
import sys
import torch

from . import log_container
from . import criterion
from .casme_utils import per_image_normalization
from .train_utils import accuracy


class PrintLogger:
    @classmethod
    def log(cls, string_to_log="", log_log=True, print_log=True, no_enter=False):
        sep = "" if no_enter else "\n"
        print(string_to_log, sep=sep)

    @classmethod
    def flush(cls):
        sys.stdout.flush()

    def close(self):
        pass


class BaseRunner:
    def train_or_eval(self, *args, **kwargs):
        raise NotImplementedError()

    def train_or_eval_batch(self, *args, **kwargs):
        raise NotImplementedError()

    def models_mode(self, train):
        raise NotImplementedError()


class CASMERunner(BaseRunner):
    def __init__(self,
                 classifier, masker,
                 classifier_optimizer, masker_optimizer,
                 classifier_criterion, mask_in_criterion, mask_out_criterion,
                 fixed_classifier, perc_of_training, prob_historic, save_freq, zoo_size,
                 image_normalization_mode,
                 add_prob_layers, prob_sample_low, prob_sample_high,
                 mask_in_weight,
                 mask_out_weight,
                 print_freq,
                 device,
                 logger=None,
                 ):
        self.classifier = classifier
        self.classifier_for_mask = None
        self.masker = masker
        self.classifier_optimizer = classifier_optimizer
        self.masker_optimizer = masker_optimizer
        self.classifier_criterion = classifier_criterion.to(device)
        self.mask_in_criterion = mask_in_criterion
        self.mask_out_criterion = mask_out_criterion

        self.fixed_classifier = fixed_classifier
        self.perc_of_training = perc_of_training
        self.prob_historic = prob_historic
        self.save_freq = save_freq
        self.zoo_size = zoo_size
        self.image_normalization_mode = image_normalization_mode

        self.add_prob_layers = add_prob_layers
        self.prob_sample_low = prob_sample_low
        self.prob_sample_high = prob_sample_high
        self.mask_in_weight = mask_in_weight
        self.mask_out_weight = mask_out_weight

        self.print_freq = print_freq
        self.device = device
        self.logger = logger if logger is not None else PrintLogger()

        self.do_mask_in = self.mask_in_criterion is not None
        self.do_mask_out = self.mask_out_criterion is not None
        self.classifier_zoo = {}
        assert self.do_mask_in or self.do_mask_out

    def train_or_eval(self, data_loader, is_train=False, epoch=None):
        log_containers = log_container.LogContainers()
        self.models_mode(is_train)
        for i, (x, y) in enumerate(data_loader):
            if is_train and i > len(data_loader) * self.perc_of_training:
                break
            x, y = x.to(self.device), y.to(self.device)
            x = per_image_normalization(x, mode=self.image_normalization_mode)
            log_containers.data_time.update()
            self.train_or_eval_batch(
                x, y, i,
                log_containers=log_containers, is_train=is_train,
            )
            # print log
            if i % self.print_freq == 0:
                if is_train:
                    self.logger.log('Epoch: [{0}][{1}/{2}/{3}]\t'.format(
                        epoch, i, int(len(data_loader)*self.perc_of_training), len(data_loader)), no_enter=True)
                else:
                    self.logger.log('Test: [{0}][{1}/{2}]\t'.format(epoch, i, len(data_loader)), no_enter=True)
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

        """
        # update metrics
        log_containers.losses.update(classifier_loss.item(), x.size(0))
        log_containers.acc.update(accuracy(y_hat.detach(), y, topk=(1,))[0].item(), x.size(0))
        """

        # update classifier - compute gradient and do SGD step for clean image, save classifier
        if is_train and (not self.fixed_classifier):
            self.classifier_optimizer_step(classifier_loss=classifier_loss)
            self.maybe_save_classifier_to_history(i=i)

        with torch.set_grad_enabled(is_train):
            # compute mask and masked input
            mask = self.masker(
                layers=self.detach_layers(layers),
                use_p=use_p,
            )
            classifier_for_mask, update_classifier = self.choose_masked_classifier(is_train)
            """
            log_containers.statistics.update(mask)
            """
            if self.do_mask_in:
                masked_in_x = criterion.MaskFunc.mask_in(x=x, mask=mask)
                y_hat_from_masked_in_x = self.classifier(masked_in_x)
                classifier_loss_from_masked_in_x = self.classifier_criterion(y_hat_from_masked_in_x, y)
                mask_in_loss, mask_in_loss_metadata = self.mask_in_criterion(
                    mask=mask, y_hat=y_hat, y_hat_from_masked_x=y_hat_from_masked_in_x, y=y,
                    classifier_loss_from_masked_x=classifier_loss_from_masked_in_x, use_p=use_p,
                )
            else:
                classifier_loss_from_masked_in_x = mask_in_loss = 0
                mask_in_loss_metadata = None

            if self.do_mask_out:
                masked_out_x = criterion.MaskFunc.mask_out(x=x, mask=mask)
                y_hat_from_masked_out_x = self.classifier(masked_out_x)
                classifier_loss_from_masked_out_x = self.classifier_criterion(y_hat_from_masked_out_x, y)
                mask_out_loss, mask_out_loss_metadata = self.mask_out_criterion(
                    mask=mask, y_hat=y_hat, y_hat_from_masked_x=y_hat_from_masked_out_x, y=y,
                    classifier_loss_from_masked_x=classifier_loss_from_masked_out_x, use_p=use_p,
                )
            else:
                classifier_loss_from_masked_out_x = mask_out_loss = 0
                mask_out_loss_metadata = None

            classifier_loss_from_masked_x = (
                self.mask_in_weight * classifier_loss_from_masked_in_x
                + self.mask_out_weight * classifier_loss_from_masked_out_x
            )
            masker_total_loss = (
                self.mask_in_weight * mask_in_loss
                + self.mask_out_weight * mask_out_loss
            )

            """
            # update metrics
            log_containers.losses_m.update(classifier_loss_from_masked_x.item(), x.size(0))
            log_containers.acc_m.update(accuracy(
                y_hat_from_masked_x.detach(), y, topk=(1,))[0].item(), x.size(0))
            """

        if is_train:
            # update classifier - compute gradient, do SGD step for masked image
            if update_classifier:
                self.classifier_from_masked_optimizer_step(classifier_loss_from_masked_x)

            # update casme - compute gradient, do SGD step
            self.masker_optimizer_step(masker_total_loss)

            """
            log_containers.masker_total_loss.update(masker_total_loss.item(), x.size(0))
            log_containers.masker_loss.update(masker_loss_metadata["loss"].item(), x.size(0))
            log_containers.masker_reg.update(masker_loss_metadata["regularization"].item(), x.size(0))
            log_containers.correct_on_clean.update(masker_loss_metadata["correct_on_clean"].item(), x.size(0))
            log_containers.mistaken_on_masked.update(masker_loss_metadata["mistaken_on_masked"].item(), x.size(0))
            log_containers.nontrivially_confused.update(masker_loss_metadata["nontrivially_confused"].item(), x.size(0))
            """

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

    def maybe_save_classifier_to_history(self, i):
        # save classifier (needed only if previous iterations are used i.e. args.hp > 0)
        if self.prob_historic > 0 \
                and ((i % self.save_freq == -1 % self.save_freq) or len(self.classifier_zoo) < 1):
            self.logger.log('Current iteration is saving, will be used in the future. ', no_enter=True)
            if len(self.classifier_zoo) < self.zoo_size:
                index = len(self.classifier_zoo)
            else:
                index = random.randint(0, len(self.classifier_zoo) - 1)
            state_dict = self.classifier.state_dict()
            self.classifier_zoo[index] = {}
            for p in state_dict:
                self.classifier_zoo[index][p] = state_dict[p].cpu()
            self.logger.log('There are {0} iterations stored.'.format(len(self.classifier_zoo)))

    def setup_classifier_for_mask(self):
        if self.classifier_for_mask is None:
            self.classifier_for_mask = copy.deepcopy(self.classifier)
        index = random.randint(0, len(self.classifier_zoo) - 1)
        self.classifier_for_mask.load_state_dict(self.classifier_zoo[index])
        self.classifier_for_mask.eval()

    def classifier_optimizer_step(self, classifier_loss):
        self.classifier_optimizer.zero_grad()
        classifier_loss.backward()
        self.classifier_optimizer.step()

    def classifier_from_masked_optimizer_step(self, classifier_loss_from_masked_x):
        self.classifier_optimizer.zero_grad()
        classifier_loss_from_masked_x.backward(retain_graph=True)
        self.classifier_optimizer.step()

    def masker_optimizer_step(self, masker_total_loss):
        self.masker_optimizer.zero_grad()
        masker_total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.masker.parameters(), 10)
        self.masker_optimizer.step()

    def choose_masked_classifier(self, is_train):
        if (not is_train) or self.fixed_classifier or (random.random() > self.prob_historic):
            classifier_for_mask = self.classifier
            update_classifier = not self.fixed_classifier
        else:
            self.setup_classifier_for_mask()
            classifier_for_mask = self.classifier_for_mask
            update_classifier = False
        return classifier_for_mask, update_classifier

    @classmethod
    def detach_layers(cls, layers):
        return [l.detach() for l in layers]


class InfillerCASMERunner(BaseRunner):
    def __init__(self,
                 classifier, masker, infiller,
                 classifier_optimizer, masker_optimizer, infiller_optimizer,
                 classifier_criterion, masker_criterion, infiller_criterion,
                 train_infiller, shuffle_infiller_masks,
                 fixed_classifier, perc_of_training, prob_historic, save_freq, zoo_size,
                 image_normalization_mode,
                 mask_func,
                 add_prob_layers, prob_sample_low, prob_sample_high,
                 print_freq,
                 device,
                 logger=None,
                 ):
        self.classifier = classifier
        self.classifier_for_mask = None
        self.masker = masker
        self.infiller = infiller
        self.classifier_optimizer = classifier_optimizer
        self.masker_optimizer = masker_optimizer
        self.infiller_optimizer = infiller_optimizer
        self.classifier_criterion = classifier_criterion.to(device)
        self.masker_criterion = masker_criterion.to(device)
        self.infiller_criterion = infiller_criterion.to(device)
        self.train_infiller = train_infiller
        self.shuffle_infiller_masks = shuffle_infiller_masks

        self.fixed_classifier = fixed_classifier
        self.perc_of_training = perc_of_training
        self.prob_historic = prob_historic
        self.save_freq = save_freq
        self.zoo_size = zoo_size
        self.image_normalization_mode = image_normalization_mode

        self.add_prob_layers = add_prob_layers
        self.prob_sample_low = prob_sample_low
        self.prob_sample_high = prob_sample_high
        self.print_freq = print_freq
        self.device = device
        self.logger = logger if logger is not None else PrintLogger()
        self.mask_func = mask_func

        self.classifier_zoo = {}

    def train_or_eval(self, data_loader, is_train=False, epoch=None):
        log_containers = log_container.ICASMELogContainers()
        self.models_mode(is_train)
        for i, (x, y) in enumerate(data_loader):
            if is_train and i > len(data_loader) * self.perc_of_training:
                break
            x, y = x.to(self.device), y.to(self.device)
            x = per_image_normalization(x, mode=self.image_normalization_mode)
            log_containers.data_time.update()
            self.train_or_eval_batch(
                x, y, i,
                log_containers=log_containers, is_train=is_train,
            )
            # print log
            if i % self.print_freq == 0:
                if is_train:
                    self.logger.log('Epoch: [{0}][{1}/{2}/{3}]\t'.format(
                        epoch, i, int(len(data_loader)*self.perc_of_training), len(data_loader)), no_enter=True)
                else:
                    self.logger.log('Test: [{0}][{1}/{2}]\t'.format(epoch, i, len(data_loader)), no_enter=True)
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
                                'NC {lc.nontrivially_confused.avg:.3f} ({lc.nontrivially_confused.val:.3f})\n'
                                'Inf {lc.infiller_loss.avg:.3f} ({lc.infiller_loss.val:.3f})\n'
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
                self.logger.log('Current iteration is saving, will be used in the future. ', no_enter=True)
                if len(self.classifier_zoo) < self.zoo_size:
                    index = len(self.classifier_zoo)
                else:
                    index = random.randint(0, len(self.classifier_zoo) - 1)
                state_dict = self.classifier.state_dict()
                self.classifier_zoo[index] = {}
                for p in state_dict:
                    self.classifier_zoo[index][p] = state_dict[p].cpu()
                self.logger.log('There are {0} iterations stored.'.format(len(self.classifier_zoo)))

        # detach inner layers to make them be features for decoder
        layers = [l.detach() for l in layers]

        with torch.set_grad_enabled(is_train):
            # compute mask and masked input
            mask = self.masker(layers, use_p=use_p)
            masked_x = self.mask_func(x=x, mask=mask)

            # Compute infill (generate using detached mask, infill using real mask)
            with torch.set_grad_enabled(self.train_infiller and not self.shuffle_infiller_masks):
                generated = self.infiller(masked_x.detach(), mask.detach())

            infilled = criterion.default_infill_func(masked_x, mask, generated)

            # update statistics
            log_containers.statistics.update(mask)

            # randomly select classifier to be evaluated on masked image and compute output
            if (not is_train) or self.fixed_classifier or (random.random() > self.prob_historic):
                y_hat_from_masked_x = self.classifier(infilled)
                update_classifier = not self.fixed_classifier
            else:
                if self.classifier_for_mask is None:
                    self.classifier_for_mask = copy.deepcopy(self.classifier)
                index = random.randint(0, len(self.classifier_zoo) - 1)
                self.classifier_for_mask.load_state_dict(self.classifier_zoo[index])
                self.classifier_for_mask.eval()
                y_hat_from_masked_x = self.classifier_for_mask(infilled)
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

            self.masker_optimizer.zero_grad()
            masker_total_loss, masker_loss_metadata = self.masker_criterion(
                mask=mask, y_hat=y_hat, y_hat_from_masked_x=y_hat_from_masked_x, y=y,
                classifier_loss_from_masked_x=classifier_loss_from_masked_x, use_p=use_p,
            )
            masker_total_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.masker.parameters(), 10)
            self.masker_optimizer.step()

            self.infiller_optimizer.zero_grad()
            with torch.set_grad_enabled(self.train_infiller):
                if self.shuffle_infiller_masks:
                    shuf_mask = torch.flip(mask.detach(), dims=[0])
                    shuf_masked_x = self.mask_func(x=x, mask=shuf_mask)
                    generated = self.infiller(shuf_masked_x.detach(), shuf_mask.detach())
                    infilled = criterion.default_infill_func(shuf_masked_x, shuf_mask, generated)
                infiller_loss = self.infiller_criterion(infilled, x)

            if self.train_infiller:
                infiller_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.infiller_criterion.parameters(), 10)
                self.infiller_optimizer.step()

            # update casme - compute gradient, do SGD step
            log_containers.masker_total_loss.update(masker_total_loss.item(), x.size(0))
            log_containers.masker_loss.update(masker_loss_metadata["loss"].item(), x.size(0))
            log_containers.masker_reg.update(masker_loss_metadata["regularization"].item(), x.size(0))
            log_containers.correct_on_clean.update(masker_loss_metadata["correct_on_clean"].item(), x.size(0))
            log_containers.mistaken_on_masked.update(masker_loss_metadata["mistaken_on_masked"].item(), x.size(0))
            log_containers.nontrivially_confused.update(masker_loss_metadata["nontrivially_confused"].item(), x.size(0))
            log_containers.infiller_loss.update(infiller_loss.item(), x.size(0))

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
