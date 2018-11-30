import copy
import cv2
import datetime as dt
import numpy as np
import random
import scipy.ndimage
import sys
import torch

from . import log_container
from . import criterion
from .utils import per_image_normalization
from .train_utils import accuracy
from .ext.pconv_keras import random_mask


class PrintLogger:
	def log(self, string_to_log="", log_log=True, print_log=True, no_enter=False):
		sep = "" if no_enter else "\n"
		print(string_to_log, sep=sep)

	def flush(self):
		sys.stdout.flush()

	def close(self):
		pass


class _BaseRunner:
    def train_or_eval(self, *args, **kwargs):
        raise NotImplementedError()

    def train_or_eval_batch(self, *args, **kwargs):
        raise NotImplementedError()

    def models_mode(self, train):
        raise NotImplementedError()


class CASMERunner(_BaseRunner):
    def __init__(self,
                 classifier, masker,
                 classifier_optimizer, masker_optimizer,
                 classifier_criterion, masker_criterion,
                 fixed_classifier, perc_of_training, prob_historic, save_freq, zoo_size,
                 add_prob_layers, prob_sample_low, prob_sample_high,
                 print_freq,
                 device,
                 logger=None,
                 apply_mask_func=criterion.default_apply_mask_func,
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
        log_containers = log_container.LogContainers()
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


class InfillerRunner(_BaseRunner):
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
            self.apply_mask_func = criterion.apply_uniform_random_value_mask_func
        else:
            self.apply_mask_func = criterion.apply_inverted_mask_func
        self.infill_func = criterion.default_infill_func

    def train_or_eval(self, data_loader, is_train=False, epoch=None):
        log_containers = log_container.InfillingLogContainers()
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
                            epoch, i, int(len(data_loader)*self.perc_of_training), len(data_loader)), no_enter=True)
                    else:
                        self.logger.log('Epoch: [{0}][{1}/{2}/{3}]\t'.format(
                            epoch, i, int(len(data_loader)*self.perc_of_training), len(data_loader)))
                else:
                    if is_printlogger:
                        self.logger.log('Test: [{0}][{1}/{2}]\t'.format(epoch, i, len(data_loader)), no_enter=True)
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
        # TODO: use data loader to parallelize mask generation
        # TODO: mask will be inverted if generated by masker... handle it later
        mask = np.stack([random_mask(x.shape[2], x.shape[3], x.shape[1]) for _ in range(x.shape[0])], axis=0)
        # generated mask here uses 1 as non-mask
        if self.infiller.model_type == "ciGAN":
            mask = criterion.invert_mask_func(mask)  # ciGAN uses 0 as non-mask
        # mask = mask.reshape(x.shape)
        # I should create FloatTensor and send to GPU and turn into cuda tensor by doing so
        dilated_boundaries = []
        for onemask in mask:
            _, contours, _ = cv2.findContours(onemask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            boundary = np.zeros_like(onemask)
            cv2.drawContours(boundary, contours, -1, 1, 1)
            dilated_boundary = scipy.ndimage.morphology.binary_dilation(boundary, iterations=1)
            dilated_boundaries.append(dilated_boundary)
        dilated_boundaries = np.stack(dilated_boundaries, axis=0).astype(float)

        mask = torch.FloatTensor(mask).to(self.device)
        mask = mask.permute(0, 3, 1, 2)

        dilated_boundaries = torch.FloatTensor(dilated_boundaries).to(self.device)
        dilated_boundaries = dilated_boundaries.permute(0, 3, 1, 2)

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
            if self.discriminator is None:
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
                generator_total_loss, discriminator_total_loss, discriminator_loss_metadata = \
                    self.discriminator_criterion(
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
            # log_containers.infiller_reg.update(infiller_loss_metadata["regularization"].item(), x.size(0))

        # measure elapsed time
        log_containers.batch_time.update()
        # print("batch done")

    def models_mode(self, train):
        self.classifier.eval()
        if train:
            self.infiller.train()
        else:
            self.infiller.eval()


class InfillerCASMERunner(_BaseRunner):
    def __init__(self,
                 classifier, masker, infiller,
                 classifier_optimizer, masker_optimizer, infiller_optimizer,
                 classifier_criterion, masker_infiller_criterion,
                 fixed_classifier, perc_of_training, prob_historic, save_freq, zoo_size,
                 add_prob_layers, prob_sample_low, prob_sample_high,
                 print_freq,
                 device,
                 logger=None,
                 apply_mask_func=criterion.default_apply_mask_func,
                 ):
        self.classifier = classifier
        self.classifier_for_modified = None
        self.masker = masker
        self.infiller = infiller
        self.classifier_optimizer = classifier_optimizer
        self.masker_optimizer = masker_optimizer
        self.infiller_optimizer = infiller_optimizer
        self.classifier_criterion = classifier_criterion.to(device)
        self.masker_infiller_criterion = masker_infiller_criterion.to(device)

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
        self.infill_func = criterion.default_infill_func

        self.classifier_zoo = {}

    def train_or_eval(self, data_loader, is_train=False, epoch=None):
        log_containers = log_container.LogContainers()
        self.models_mode(is_train)
        for i, (x, y) in enumerate(data_loader):
            if is_train and i > len(data_loader) * self.perc_of_training:
                break
            x, y = x.to(self.device), y.to(self.device)
            x = per_image_normalization(x)
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
                                'MLoss(M) {lc.masker_loss.avg:.8f} ({lc.masker_loss.val:.8f})\t'
                                'MReg(M) {lc.masker_reg.avg:.4f} ({lc.masker_reg.val:.4f})\n'
                                'CoC {lc.correct_on_clean.avg:.3f} ({lc.correct_on_clean.val:.3f})\t'
                                'MoM {lc.mistaken_on_masked.avg:.3f} ({lc.mistaken_on_masked.val:.3f})\t'
                                'NC {lc.nontrivially_confused.avg:.3f} ({lc.nontrivially_confused.val:.3f})\t'
                                ''.format(lc=log_containers))
                self.logger.log(log_containers.statistics.str_out())
                self.logger.log('{:%Y-%m-%d %H:%M:%S}'.format(dt.datetime.now()))
                self.logger.log()

        if not is_train:
            self.logger.log(' * Prec@1 {lc.acc.avg:.3f} Prec@1(M) {lc.acc_m.avg:.3f} '.format(lc=log_containers))
            self.logger.log(log_containers.statistics.str_out())

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
            masked_x = self.apply_mask_func(x=x, mask=mask)
            generated_x, _ = self.infiller(masked_x, 1-mask)
            modified_x = self.infill_func(x=masked_x, mask=1-mask, generated_image=generated_x)

            # update statistics
            log_containers.statistics.update(mask)

            # randomly select classifier to be evaluated on masked image and compute output
            if (not is_train) or self.fixed_classifier or (random.random() > self.prob_historic):
                y_hat_from_modified_x = self.classifier(modified_x)
                update_classifier = not self.fixed_classifier
            else:
                if self.classifier_for_modified is None:
                    self.classifier_for_modified = copy.deepcopy(self.classifier)
                index = random.randint(0, len(self.classifier_zoo) - 1)
                self.classifier_for_modified.load_state_dict(self.classifier_zoo[index])
                self.classifier_for_modified.eval()
                y_hat_from_modified_x = self.classifier_for_modified(modified_x)
                update_classifier = False

            classifier_loss_from_modified_x = self.classifier_criterion(y_hat_from_modified_x, y)

            # update metrics
            log_containers.losses_m.update(classifier_loss_from_modified_x.item(), x.size(0))
            log_containers.acc_m.update(accuracy(
                y_hat_from_modified_x.detach(), y, topk=(1,))[0].item(), x.size(0))

        if is_train:
            # update classifier - compute gradient, do SGD step for masked image
            if update_classifier:
                self.classifier_optimizer.zero_grad()
                classifier_loss_from_modified_x.backward(retain_graph=True)
                self.classifier_optimizer.step()

            masker_total_loss, masker_loss_metadata = self.masker_infiller_criterion(
                mask=mask, modified_x=modified_x, y_hat=y_hat,
                y_hat_from_modified_x=y_hat_from_modified_x, y=y,
                classifier_loss_from_modified_x=classifier_loss_from_modified_x, use_p=use_p,
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


class GANRunner(_BaseRunner):
    def __init__(self,
                 discriminator, infiller,
                 d_optimizer, i_optimizer,
                 save_freq,
                 print_freq,
                 perc_of_training,
                 device,
                 logger=None,
                 ):
        self.discriminator = discriminator
        self.infiller = infiller
        self.d_optimizer = d_optimizer
        self.i_optimizer = i_optimizer

        self.perc_of_training = perc_of_training
        self.save_freq = save_freq
        self.print_freq = print_freq
        self.device = device
        self.logger = logger if logger is not None else PrintLogger()
        if self.infiller.model_type == "ciGAN":
            # TODO: ciGAN uses 0 as non-mask, and mask will be converted so.
            self.apply_mask_func = criterion.apply_uniform_random_value_mask_func
        else:
            self.apply_mask_func = criterion.apply_inverted_mask_func
        self.infill_func = criterion.default_infill_func

    def train_or_eval(self, data_loader, is_train=False, epoch=None):
        log_containers = log_container.GANLogContainers()
        self.models_mode(is_train)
        for i, (x, _) in enumerate(data_loader):
            if is_train and i > len(data_loader) * self.perc_of_training:
                break
            x = x.to(self.device)
            x = per_image_normalization(x)
            log_containers.data_time.update()
            self.train_or_eval_batch(x, i, log_containers=log_containers, is_train=is_train)
            # print log
            if i % self.print_freq == 0:
                is_printlogger = 'PrintLogger' in str(type(self.logger))
                if is_train:
                    if is_printlogger:
                        self.logger.log('Epoch: [{0}][{1}/{2}/{3}]\t'.format(
                            epoch, i, int(len(data_loader)*self.perc_of_training), len(data_loader)), no_enter=True)
                    else:
                        self.logger.log('Epoch: [{0}][{1}/{2}/{3}]\t'.format(
                            epoch, i, int(len(data_loader)*self.perc_of_training), len(data_loader)))
                else:
                    if is_printlogger:
                        self.logger.log('Test: [{0}][{1}/{2}]\t'.format(epoch, i, len(data_loader)), no_enter=True)
                    else:
                        self.logger.log('Test: [{0}][{1}/{2}]\t'.format(epoch, i, len(data_loader)))
                # TODO: print out other parts of losses
                self.logger.log('Time {lc.batch_time.avg:.3f} ({lc.batch_time.val:.3f})\t'
                                'Data {lc.data_time.avg:.3f} ({lc.data_time.val:.3f})\n'
                                'DLoss(M) {lc.d_loss.avg:.4f} ({lc.d_loss.val:.4f})\t'
                                'ILoss(M) {lc.i_loss.avg:.4f} ({lc.i_loss.val:.4f})\n'
                                'DLossR(M) {lc.d_loss_real.avg:.4f} ({lc.d_loss_real.val:.4f})\t'
                                'ILossF(M) {lc.d_loss_fake.avg:.4f} ({lc.d_loss_fake.val:.4f})\n'
                                ''.format(lc=log_containers))
                log_containers.statistics.print_out()
                self.logger.log('{:%Y-%m-%d %H:%M:%S}'.format(dt.datetime.now()))
                if is_printlogger:
                    self.logger.log()
        return log_containers

    def train_or_eval_batch(self, x, i, log_containers, is_train=False):
        # zp489: Hard-coding max_multiplier for now
        masked_x, mask = self.generate_masked_x(x, max_multiplier=2)
        self.d_step(x, masked_x, mask, i, log_containers, is_train, threshold=0.5)
        self.i_step(x, masked_x, mask, i, log_containers, is_train)

        # measure elapsed time
        log_containers.batch_time.update()

    def d_step(self, x, masked_x, mask, i, log_containers, is_train=False, threshold=None):
        with torch.set_grad_enabled(is_train):
            # compute mask and masked input
            generated_x, generated_mask = self.infiller(masked_x, mask)
            infilled_x = self.infill_func(x=x, mask=mask, generated_image=generated_x)
            real_pred = self.discriminator(x, return_intermediate=False)
            fake_pred = self.discriminator(infilled_x, return_intermediate=False)

            # loss_real = - torch.mean(real_pred)
            # loss_fake = fake_pred.mean()
            # Hinge from https://github.com/heykeetae/Self-Attention-GAN/blob/master/trainer.py

            noise_real = torch.rand(x.shape[0]).to(self.device) * 0.2 - 0.1
            noise_fake = torch.rand(x.shape[0]).to(self.device) * 0.2 - 0.1

            loss_real = torch.nn.ReLU()(1.0 - real_pred + noise_real).mean()
            loss_fake = torch.nn.ReLU()(1.0 + fake_pred + noise_fake).mean()
            loss = loss_real + loss_fake

        if is_train:
            if threshold is not None and loss.item() > threshold:
                self.d_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 10)
                self.d_optimizer.step()

        log_containers.d_loss_real.update(loss_real.item(), x.size(0))
        log_containers.d_loss_fake.update(loss_fake.item(), x.size(0))
        log_containers.d_loss.update(loss.item(), x.size(0) * 2)

    def i_step(self, x, masked_x, mask, i, log_containers, is_train=False):
        with torch.set_grad_enabled(is_train):
            # compute mask and masked input
            generated_x, generated_mask = self.infiller(masked_x, mask)
            infilled_x = self.infill_func(x=x, mask=mask, generated_image=generated_x)
            fake_pred = self.discriminator(infilled_x, return_intermediate=False)
            loss = - fake_pred.mean()

        if is_train:
            self.i_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.infiller.parameters(), 10)
            self.i_optimizer.step()

        log_containers.i_loss.update(loss.item(), x.size(0))

    def generate_masked_x(self, x, min_multiplier=1, max_multiplier=1, number=20):
        mask = np.stack([
            random_mask(
                x.shape[2], x.shape[3], x.shape[1],
                min_multiplier=min_multiplier,
                max_multiplier=max_multiplier,
                number=number,
            )
            for _ in range(x.shape[0])
        ], axis=0)
        # generated mask here uses 1 as non-mask
        if self.infiller.model_type == "ciGAN":
            mask = criterion.invert_mask_func(mask)  # ciGAN uses 0 as non-mask

        mask = torch.FloatTensor(mask).to(self.device)
        mask = mask.permute(0, 3, 1, 2)
        masked_x = self.apply_mask_func(x=x, mask=mask)
        return masked_x, mask

    def models_mode(self, train):
        if train:
            self.discriminator.train()
            self.infiller.train()
        else:
            self.discriminator.eval()
            self.infiller.eval()
