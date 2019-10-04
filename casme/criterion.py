import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ext.pytorch_inpainting_with_partial_conv import gram_matrix, total_variation_loss


class MaskFunc:

    MASK_IN = "MASK_IN"
    MASK_OUT = "MASK_OUT"
    MODES = [MASK_IN, MASK_OUT]

    def __init__(self, mask_mode):
        assert mask_mode in self.MODES
        self.mask_mode = mask_mode

    def apply_mask(self, x, mask):
        if self.mask_mode == self.MASK_IN:
            return self.mask_in(x, mask)
        elif self.mask_mode == self.MASK_OUT:
            return self.mask_out(x, mask)
        else:
            raise KeyError(self.mask_mode)

    @staticmethod
    def mask_out(x, mask):
        return x * (1 - mask)

    @staticmethod
    def mask_in(x, mask):
        return x * mask


def apply_uniform_random_value_mask_func(x, mask):
    # for mask, return image infilled with random value between 0 and 1?
    # zero_mean_std_1?
    # x[mask==1] = np.random.random((mask==1).shape)
    return None


def default_infill_func(x, mask, generated_image):
    return generated_image * mask + x * (1-mask)


class InfillerCriterion(nn.Module):
    def __init__(self, model_type, style_loss_multiplier):
        super().__init__()
        self.l1 = nn.L1Loss()
        # TODO: take mask function as parameters
        # TODO: take model hyperparameters (how many layers to use fur perceptual loss, loss mutlipliers)
        # TODO: use different criterion depending on models
        self.model_type = model_type

    def forward(self, x, mask, generated_image, infilled_image, layers, generated_layers,
                infilled_layers, dilated_boundaries):
        # x here is ground-truth

        # assuming 1 means background
        hole = self.l1(MaskFunc.mask_out(x, mask), MaskFunc.mask_out(generated_image, mask))
        valid = self.l1(MaskFunc.mask_in(x, mask), MaskFunc.mask_in(generated_image, mask))

        # layers = feature of gt image

        perceptual_loss = 0  # self.l1(layers[i], infilled_layers[i])
        style_out_loss = 0  # self.l1(gram_matrix(layers[i]), gram_matrix(infilled_layers[i]))
        style_comp_loss = 0  # self.l1(gram_matrix(layers[i]), gram_matrix(infilled_layers[i]))
        for i in range(3):
            perceptual_loss += self.l1(layers[i], infilled_layers[i])
            style_out_loss += self.l1(gram_matrix(layers[i]), gram_matrix(generated_layers[i]))
            style_comp_loss += self.l1(gram_matrix(layers[i]), gram_matrix(infilled_layers[i]))

        # TODO: Is this loss wrong? if it only backprops to generated bits... ... Try implementing my own.
        # TODO: try using total_variation_loss on dilated boundary region only... but still, incorrect boundary
        tv = total_variation_loss(MaskFunc.mask_in(infilled_image, dilated_boundaries))
        regularization = 0
        loss = (
            6 * hole
            + valid
            + 0.05 * perceptual_loss
            + self.style_loss_multiplier*(style_out_loss+style_comp_loss)
            + 0.1 * tv
        )
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
    def __init__(self, lambda_r, add_prob_layers, prob_loss_func,
                 objective_direction, objective_type,
                 device, y_hat_log_softmax=False):
        super().__init__()
        self.lambda_r = lambda_r
        self.add_prob_layers = add_prob_layers
        self.prob_loss_func = prob_loss_func
        self.objective_direction = objective_direction
        self.objective_type = objective_type
        self.device = device
        self.y_hat_log_softmax = y_hat_log_softmax

    def forward(self,
                mask, y_hat, y_hat_from_masked_x, y,
                classifier_loss_from_masked_x, use_p, reduce=True):
        _, max_indexes = y_hat.detach().max(1)
        _, max_indexes_on_masked_x = y_hat_from_masked_x.detach().max(1)
        correct_on_clean = y.eq(max_indexes)
        mistaken_on_masked = y.ne(max_indexes_on_masked_x)
        nontrivially_confused = (correct_on_clean + mistaken_on_masked).eq(2).float()

        mask_mean = F.avg_pool2d(mask, 224, stride=1).squeeze()
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

        loss, loss_metadata = self.compute_only_loss(
            y_hat_from_masked_x=y_hat_from_masked_x,
            y=y,
            classifier_loss_from_masked_x=classifier_loss_from_masked_x,
            correct_on_clean=correct_on_clean,
            reduce=reduce,
        )
        masker_loss = loss + regularization
        if reduce:
            regularization = regularization.mean()
            metadata = {
                "correct_on_clean": correct_on_clean.float().mean(),
                "mistaken_on_masked": mistaken_on_masked.float().mean(),
                "nontrivially_confused": nontrivially_confused.float().mean(),
                "loss": loss,
                "regularization": regularization,
            }
        else:
            metadata = {
                "correct_on_clean": correct_on_clean.float(),
                "mistaken_on_masked": mistaken_on_masked.float(),
                "nontrivially_confused": nontrivially_confused.float(),
                "loss": loss,
                "regularization": regularization,
            }
        if self.objective_type == "entropy":
            metadata["negative_entropy"] = loss_metadata["negative_entropy"]
        return masker_loss, metadata

    def compute_only_loss(self, y_hat_from_masked_x, y, classifier_loss_from_masked_x,
                          correct_on_clean, reduce):

        loss_metadata = {}
        # main loss for casme
        if self.objective_type == "classification":
            if reduce:
                objective = classifier_loss_from_masked_x
            else:
                # This is a hack.
                if self.y_hat_log_softmax:
                    classifier_criterion = F.nll_loss
                else:
                    classifier_criterion = F.cross_entropy
                objective = classifier_criterion(y_hat_from_masked_x, y, reduction='none')
        elif self.objective_type == "entropy":
            if self.y_hat_log_softmax:
                log_prob = y_hat_from_masked_x
            else:
                log_prob = F.log_softmax(y_hat_from_masked_x, dim=-1)
            prob = log_prob.exp()
            entropy = -(log_prob * prob).sum(1)
            # apply main loss only when original images are correctly classified
            entropy_correct = entropy * correct_on_clean.float()
            objective = entropy_correct
            loss_metadata["negative_entropy"] = -entropy
        else:
            raise KeyError(self.objective_type)

        if reduce:
            objective = objective.mean()

        loss = self.determine_loss_direction(objective)
        return loss, loss_metadata

    def determine_loss_direction(self, loss):
        if self.objective_direction == "maximize":
            # maximize objective = minimize negative loss
            return -loss
        elif self.objective_direction == "minimize":
            # minimize objective = minimize loss
            return loss
        else:
            raise KeyError(self.loss_type)


class MaskerPriorCriterion(nn.Module):
    def __init__(self, lambda_r, class_weights, add_prob_layers, prob_loss_func, config, device,
                 y_hat_log_softmax=False):
        super().__init__()
        self.lambda_r = lambda_r
        self.add_prob_layers = add_prob_layers
        self.prob_loss_func = prob_loss_func
        self.config = json.loads(config)
        self.device = device
        self.y_hat_log_softmax = y_hat_log_softmax

        self.class_weights = torch.Tensor(class_weights).to(device)
        inverse_class_weights = 1 / self.class_weights
        self.prior = (inverse_class_weights / inverse_class_weights.sum())

    def forward(self,
                mask, y_hat, y_hat_from_masked_x, y,
                classifier_loss_from_masked_x, use_p, reduce=True):
        if self.y_hat_log_softmax:
            y_hat_prob = torch.exp(y_hat)
            y_hat_from_masked_x_prob = torch.exp(y_hat_from_masked_x)
        else:
            y_hat_prob = F.softmax(y_hat, dim=-1)
            y_hat_from_masked_x_prob = F.softmax(y_hat_from_masked_x, dim=-1)

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

        mask_mean = F.avg_pool2d(mask, 224, stride=1).squeeze()
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
        if self.add_prob_layers or self.config.get("regularize_always"):
            regularization = self.lambda_r * mask_mean
        else:
            regularization = -self.lambda_r * F.relu(nontrivially_confused - mask_mean)

        # main loss for casme
        if self.y_hat_log_softmax:
            log_prob = y_hat_from_masked_x
        else:
            log_prob = F.log_softmax(y_hat_from_masked_x, dim=-1)
        if self.config["kl"] == "forward":
            # - sum: p_i log(q_i)
            kl = - (self.prior * log_prob).sum(dim=-1)
        elif self.config["kl"] == "backward":
            log_prior = torch.log(self.prior)
            # - sum: q_i log(p_i / q_i)
            kl = - (y_hat_from_masked_x_prob * (log_prior - log_prob)).sum(dim=-1)
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

        if reduce:
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
        else:
            loss = kl
            masker_loss = loss + regularization
            metadata = {
                "correct_on_clean": correct_on_clean.float(),
                "mistaken_on_masked": mistaken_on_masked.float(),
                "nontrivially_confused": nontrivially_confused.float(),
                "loss": loss,
                "regularization": regularization,
            }
        return masker_loss, metadata


class MaskerInfillerPriorCriterion(nn.Module):
    def __init__(self, lambda_r, class_weights, add_prob_layers, prob_loss_func, config, device,
                 y_hat_log_softmax=False):
        super().__init__()
        self.lambda_r = lambda_r
        self.add_prob_layers = add_prob_layers
        self.prob_loss_func = prob_loss_func
        self.config = json.loads(config)
        self.device = device
        self.y_hat_log_softmax = y_hat_log_softmax

        self.class_weights = torch.Tensor(class_weights).to(device)
        inverse_class_weights = 1 / self.class_weights
        self.prior = (inverse_class_weights / inverse_class_weights.sum())

    def forward(self,
                mask, modified_x,
                # generated_image, infilled_image,
                # layers, generated_layers, infilled_layers, dilated_boundaries,
                y_hat, y_hat_from_modified_x, y,
                classifier_loss_from_modified_x, use_p):
        y_hat_prob = F.softmax(y_hat, dim=1)
        y_hat_from_modified_x_prob = F.softmax(y_hat_from_modified_x, dim=1)

        # Should this be / or - ?
        if self.config["prior"] == "subtract":
            y_hat_is_over_prior = y_hat_prob - self.prior
            y_hat_from_modified_x_prob_over_prior = y_hat_from_modified_x - self.prior
        elif self.config["prior"] == "divide":
            y_hat_is_over_prior = y_hat_prob / self.prior
            y_hat_from_modified_x_prob_over_prior = y_hat_from_modified_x_prob / self.prior
        else:
            raise KeyError(self.config["prior"])

        _, max_indexes = y_hat_is_over_prior.detach().max(1)
        _, max_indexes_on_modified_x = y_hat_from_modified_x_prob_over_prior.detach().max(1)

        correct_on_clean = y.eq(max_indexes)
        mistaken_on_masked = y.ne(max_indexes_on_modified_x)
        nontrivially_confused = (correct_on_clean + mistaken_on_masked).eq(2).float()

        mask_mean = F.avg_pool2d(mask, 224, stride=1).squeeze()
        if self.add_prob_layers:
            # adjust to minimize deviation from p
            print("A", mask_mean.mean())
            mask_mean = (mask_mean - use_p)
            if self.prob_loss_func == "l1":
                mask_mean = mask_mean.abs()
            elif self.prob_loss_func == "l2":
                mask_mean = mask_mean.pow(2)
            else:
                raise KeyError(self.prob_loss_func)
            print("B", mask_mean.mean())

        # apply regularization loss only on non-trivially confused images
        if self.add_prob_layers:
            regularization = self.lambda_r * mask_mean
        else:
            regularization = -self.lambda_r * F.relu(nontrivially_confused - mask_mean)

        # main loss for casme
        log_prob = F.log_softmax(y_hat_from_modified_x, dim=-1)
        if self.config["kl"] == "forward":
            # - sum: p_i log(q_i)
            kl = - (self.prior * log_prob).sum(dim=-1)
        elif self.config["kl"] == "backward":
            log_prior = torch.log(self.prior)
            # - sum: q_i log(p_i / q_i)
            kl = - (y_hat_from_modified_x * (log_prior - log_prob)).sum(dim=-1)
        elif self.config["kl"] == "backward_ce":
            log_prior = torch.log(self.prior)
            # - sum: q_i log(p_i / q_i)
            kl = - (y_hat_from_modified_x * log_prior).sum(dim=-1)
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
        print("C", regularization.item())
        metadata = {
            "correct_on_clean": correct_on_clean.float().mean(),
            "mistaken_on_masked": mistaken_on_masked.float().mean(),
            "nontrivially_confused": nontrivially_confused.float().mean(),
            "loss": loss,
            "regularization": regularization,
        }
        return masker_loss, metadata


class DiscriminatorCriterion(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss()
        self.device = device

    def forward(self, real_images_logits, gen_images_logits):
        batch_size = real_images_logits.shape[0]  # TODO

        valid = torch.tensor([1.] * batch_size, device=self.device, requires_grad=False, dtype=torch.float32).view(-1,
                                                                                                                   1)
        fake = torch.tensor([0.] * batch_size, device=self.device, requires_grad=False, dtype=torch.float32).view(-1, 1)
        generator_loss = self.adversarial_loss(gen_images_logits, valid)
        real_loss = self.adversarial_loss(real_images_logits, valid)
        fake_loss = self.adversarial_loss(gen_images_logits, fake)
        metadata = {
            'generator_loss': generator_loss,
            'real_loss': real_loss,
            'fake_loss': fake_loss,
        }
        discriminator_loss = (real_loss + fake_loss) / 2
        return generator_loss, discriminator_loss, metadata


def determine_mask_func(objective_direction, objective_type):
    if objective_type == "classification" and objective_direction == "maximize":
        mask_func = MaskFunc(mask_mode=MaskFunc.MASK_IN)
    elif objective_type == "classification" and objective_direction == "minimize":
        mask_func = MaskFunc(mask_mode=MaskFunc.MASK_OUT)
    elif objective_type == "entropy" and objective_direction == "maximize":
        mask_func = MaskFunc(mask_mode=MaskFunc.MASK_OUT)
    elif objective_type == "entropy" and objective_direction == "minimize":
        mask_func = MaskFunc(mask_mode=MaskFunc.MASK_IN)
    else:
        raise KeyError((objective_type, objective_direction))
    return mask_func
