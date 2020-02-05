import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from casme.stats import AverageMeter, StatisticsContainer
from casme.model_basics import (
    casme_load_model, get_masks_and_check_predictions, BoxCoords, classification_accuracy,
    get_rectangular_mask,
)
from casme.utils.torch_utils import ImagePathDataset
import casme.tasks.imagenet.utils as imagenet_utils
from casme import archs

import zconf
import pyutils.io as io
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from casme.tasks.imagenet.score_bboxes import (
    masks_to_bboxes, get_path_stub, get_image_bboxes, get_loc_scores,
    compute_saliency_metric, compute_saliency_metric_ground_truth
)
from scipy.stats import rankdata

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
AXIS_RANGE = np.arange(224)


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    val_json = zconf.attr(help='train_json path')
    mode = zconf.attr(type=str)
    bboxes_path = zconf.attr(help='path to bboxes_json')
    casm_path = zconf.attr(help='model_checkpoint')
    classifier_load_mode = zconf.attr(default="pickled")
    output_path = zconf.attr(help='output_path')
    record_bboxes = zconf.attr(type=str, default=None)
    logits_use_p = zconf.attr(type=str, default=None)

    workers = zconf.attr(default=4, type=int, help='number of data loading workers (default: 4)')
    batch_size = zconf.attr(default=128, type=int, help='mini-batch size (default: 256)')
    print_freq = zconf.attr(default=10, type=int, help='print frequency (default: 10)')
    break_ratio = zconf.attr(action='store_true', help='break original aspect ratio when resizing')
    not_normalize = zconf.attr(action='store_true', help='prevents normalization')

    pot = zconf.attr(default=1, type=float, help='percent of validation set seen')


def main(args: RunConfiguration):
    # data loading code
    data_loader = torch.utils.data.DataLoader(
        ImagePathDataset.from_path(
            config_path=args.val_json,
            transform=transforms.Compose([
                transforms.Resize([224, 224] if args.break_ratio else 224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                imagenet_utils.NORMALIZATION,
            ]),
            return_paths=True,
        ),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False
    )
    original_classifier = archs.resnet50shared(pretrained=True).eval().to(device)

    # get score for special cases
    if args.mode == "max":
        model = {'special': 'max', 'classifier': original_classifier}
    elif args.mode == "min":
        model = {'special': 'min', 'classifier': original_classifier}
    elif args.mode == "center":
        model = {'special': 'center', 'classifier': original_classifier}
    elif args.mode == "ground_truth":
        model = {'special': 'ground_truth', 'classifier': original_classifier}
    elif args.mode == "casme":
        model = casme_load_model(args.casm_path, classifier_load_mode=args.classifier_load_mode)
    elif args.mode == "external":
        model = {'special': 'external', 'classifier': original_classifier, 'bboxes': io.read_json(args.casm_path)}
    else:
        raise KeyError(args.mode)

    gt_bboxes = io.read_json(args.bboxes_path)

    results, candidate_bbox_ls = score(
        args=args,
        model=model,
        data_loader=data_loader,
        bboxes=gt_bboxes,
        original_classifier=original_classifier,
        record_bboxes=args.record_bboxes,
    )

    io.write_json(results, args.output_path)
    if args.record_bboxes:
        assert candidate_bbox_ls
        io.write_json([bbox.to_dict() for bbox in candidate_bbox_ls], args.record_bboxes)


def score(args, model, data_loader, bboxes, original_classifier, record_bboxes=False):
    if 'special' in model.keys():
        print("=> Special mode evaluation: {}.".format(model['special']))

    # setup meters
    batch_time = 0
    data_time = 0
    f1_meter = AverageMeter()
    f1a_meter = AverageMeter()
    le_meter = AverageMeter()
    om_meter = AverageMeter()
    sm_meter = AverageMeter()
    sm1_meter = AverageMeter()
    sm2_meter = AverageMeter()
    sm_acc_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    binarized_meter = AverageMeter()
    statistics = StatisticsContainer()

    end = time.time()
    candidate_bbox_ls = []

    logits_use_p = torch.load(args.logits_use_p)

    # data loop
    for i, ((input_, target), paths) in enumerate(data_loader):
        if i > len(data_loader)*args.pot:
            break

        data_time += time.time() - end

        # compute continuous mask, rectangular mask and compare class predictions with targets
        if 'special' in model.keys():
            is_correct = target.ge(0).numpy()
            if model['special'] == 'max':
                continuous = binarized = np.ones((args.batch_size, 224, 224))
                rectangular = continuous
                bbox_coords = [BoxCoords(0, 224, 0, 224)] * len(target)
            elif model['special'] == 'min':
                continuous = binarized= np.zeros((args.batch_size, 224, 224))
                rectangular = continuous
                bbox_coords = [BoxCoords(0, 0, 0, 0)] * len(target)
            elif model['special'] == 'center':
                continuous = binarized = np.zeros((args.batch_size, 224, 224))
                continuous[:, 33:-33, 33:-33] = 1
                rectangular = continuous
                bbox_coords = [BoxCoords(33, 224-33, 33, 224-33)] * len(target)
            elif model['special'] == 'none':
                continuous = binarized = np.zeros((args.batch_size, 224, 224))
                rectangular = continuous
                bbox_coords = [BoxCoords(0, 0, 0, 0)] * len(target)
            elif model['special'] == 'ground_truth':
                # Special handling later
                rectangular = continuous = binarized = np.zeros((args.batch_size, 224, 224))
                bbox_coords = [None] * len(target)
            elif model['special'] == 'external':
                # Special handling later
                rectangular = continuous = binarized = np.zeros((args.batch_size, 224, 224))
                bbox_coords = [BoxCoords.from_dict(model["bboxes"][get_path_stub(path)]) for path in paths]
                for j, bbox_coord in enumerate(bbox_coords):
                    rectangular[j, bbox_coord.yslice, bbox_coord.xslice] = 1
            else:
                raise KeyError(model["special"])
        else:
            continuous, binarized, rectangular, is_correct, bbox_coords, classifier_outputs = \
                get_masks_and_check_predictions(
                    input_=input_, target=target, model=model,
                    use_p=None,
                    p_ls=logits_use_p[i * args.batch_size: (i+1) * args.batch_size],
                )

            acc1, acc5 = classification_accuracy(classifier_outputs, target.to(device), topk=(1, 5))
            top1_meter.update(acc1.item(), n=target.shape[0])
            top5_meter.update(acc5.item(), n=target.shape[0])
            if record_bboxes:
                candidate_bbox_ls += masks_to_bboxes(rectangular)

        # update statistics
        statistics.update(torch.tensor(continuous).unsqueeze(1))
        binarized_meter.update(binarized.reshape(target.shape[0], -1).mean(-1).mean(), n=target.shape[0])

        # image loop
        for idx, path in enumerate(paths):
            gt_boxes = get_image_bboxes(bboxes_dict=bboxes, path=path)

            # compute localization metrics
            f1s_for_image = []
            ious_for_image = []
            for gt_box in gt_boxes:
                if model.get('special') == 'ground_truth':
                    gt_mask = np.zeros([224, 224])
                    truncated_gt_box = gt_box.clamp(0, 224)
                    gt_mask[truncated_gt_box.yslice, truncated_gt_box.xslice] = 1
                    f1_for_box, iou_for_box = get_loc_scores(gt_box, gt_mask, gt_mask)
                else:
                    f1_for_box, iou_for_box = get_loc_scores(gt_box, continuous[idx], rectangular[idx])

                f1s_for_image.append(f1_for_box)
                ious_for_image.append(iou_for_box)

            f1_meter.update(np.array(f1s_for_image).max())
            f1a_meter.update(np.array(f1s_for_image).mean())
            le_meter.update(1 - np.array(ious_for_image).max())
            om_meter.update(1 - (np.array(ious_for_image).max() * is_correct[idx]))

        if model.get('special') == 'ground_truth':
            saliency_metric, sm1_ls, sm2_ls = compute_saliency_metric_ground_truth(
                input_=input_,
                target=target,
                bboxes=bboxes,
                paths=paths,
                classifier=original_classifier,
            )
            sm_acc = None
        else:
            saliency_metric, sm1_ls, sm2_ls, sm_acc = compute_saliency_metric(
                input_=input_,
                target=target,
                bbox_coords=bbox_coords,
                classifier=original_classifier,
            )

        for sm, sm1, sm2 in zip(saliency_metric, sm1_ls, sm2_ls):
            sm_meter.update(sm)
            sm1_meter.update(sm1)
            sm2_meter.update(sm2)
        if sm_acc is not None:
            sm_acc_meter.update(sm_acc, n=len(saliency_metric))

        # measure elapsed time
        batch_time += time.time() - end
        end = time.time()

        # print log
        if i % args.print_freq == 0 and i > 0:
            print('[{0}/{1}]\t'
                  'Time {batch_time:.3f}\t'
                  'Data {data_time:.3f}\n'
                  'F1 {F1.avg:.3f} ({F1.val:.3f})\t'
                  'F1a {F1a.avg:.3f} ({F1a.val:.3f})\t'
                  'OM {OM.avg:.3f} ({OM.val:.3f})\t'
                  'LE {LE.avg:.3f} ({LE.val:.3f})\n'
                  'SM {SM.avg:.3f} ({SM.val:.3f})\t'
                  'SM1 {SM1.avg:.3f} ({SM1.val:.3f})\t'
                  'SM2 {SM2.avg:.3f} ({SM2.val:.3f})\t'
                  ''.format(
                      i, len(data_loader), batch_time=batch_time, data_time=data_time,
                      F1=f1_meter, F1a=f1a_meter, OM=om_meter, LE=le_meter, SM=sm_meter,
                      SM1=sm1_meter, SM2=sm2_meter), flush=True)
            statistics.print_out()

    print('Final:\t'
          'Time {batch_time:.3f}\t'
          'Data {data_time:.3f}\n'
          'F1 {F1.avg:.3f} ({F1.val:.3f})\t'
          'F1a {F1a.avg:.3f} ({F1a.val:.3f})\t'
          'OM {OM.avg:.3f} ({OM.val:.3f})\t'
          'LE {LE.avg:.3f} ({LE.val:.3f})\n'
          'SM {SM.avg:.3f} ({SM.val:.3f})\t'
          'SM1 {SM1.avg:.3f} ({SM1.val:.3f})\t'
          'SM2 {SM2.avg:.3f} ({SM2.val:.3f})\t'
          ''.format(
                batch_time=batch_time, data_time=data_time, F1=f1_meter, F1a=f1a_meter, OM=om_meter,
                LE=le_meter, SM=sm_meter, SM1=sm1_meter, SM2=sm2_meter), flush=True)
    statistics.print_out()

    results = {
        'F1': f1_meter.avg,
        'F1a': f1a_meter.avg,
        'OM': om_meter.avg,
        'LE': le_meter.avg,
        'SM': sm_meter.avg,
        'SM1': sm1_meter.avg,
        'SM2': sm2_meter.avg,
        'top1': top1_meter.avg,
        'top5': top5_meter.avg,
        'sm_acc': sm_acc_meter.avg,
        'binarized': binarized_meter.avg,
        **statistics.get_dictionary(),
    }
    return results, candidate_bbox_ls


def get_mask_logits(classifier, masker, input_, use_p):
    classifier_output, layers = classifier(input_, return_intermediate=True)
    additional_channels = (
            masker.maybe_add_prob_layers(layers, use_p)
            + masker.maybe_add_class_layers(layers, None)
    )

    layers = masker.append_channels(layers, additional_channels)

    k = []
    if 0 in masker.use_layers:
        if masker.use_layers == (0,):
            k.append(masker.conv1x1_0(layers[0]))
        else:
            k.append(layers[0])
    if 1 in masker.use_layers:
        k.append(masker.conv1x1_1(layers[1]))
    if 2 in masker.use_layers:
        k.append(masker.conv1x1_2(layers[2]))
    if 3 in masker.use_layers:
        k.append(masker.conv1x1_3(layers[3]))
    if 4 in masker.use_layers:
        k.append(masker.conv1x1_4(layers[4]))
    final_inputs = torch.cat(k, 1)

    # safety checks cause this is a hack
    assert isinstance(masker.final[0], nn.Conv2d)
    assert isinstance(masker.final[1], nn.Sigmoid)
    assert masker.final[2].__class__.__name__ == "Upsample"
    assert len(masker.final) == 3

    logits = masker.final[0](final_inputs)
    big_logits = masker.final[2](logits)

    small_mask = masker.final[1](logits)
    big_mask = masker.final[2](small_mask)
    return big_mask, big_logits, classifier_output


def get_masks_and_check_predictions(input_, target, model, p_ls, erode_k=0, dilate_k=0, use_p=None):
    with torch.no_grad():
        input_, target = input_.clone(), target.clone()
        input_ = input_.to(device)
        mask, logits, classifier_output = get_mask_logits(
            classifier=model["classifier"],
            masker=model["masker"],
            input_=input_,
            use_p=use_p,
        )

        binarized_mask = binarize_logits(logits, p_ls)
        rectangular = torch.empty_like(binarized_mask)
        box_coord_ls = [BoxCoords(0, 0, 0, 0)] * len(input_)

        for idx in range(mask.size(0)):
            if binarized_mask[idx].sum() == 0:
                continue

            m = binarized_mask[idx].squeeze().cpu().numpy()
            if erode_k != 0:
                m = binary_erosion(m, iterations=erode_k, border_value=1)
            if dilate_k != 0:
                m = binary_dilation(m, iterations=dilate_k)
            rectangular[idx], box_coord_ls[idx] = get_rectangular_mask(m)

        target = target.to(device)
        _, max_indexes = classifier_output.data.max(1)
        is_correct = target.eq(max_indexes).long()

        return (
            mask.squeeze().cpu().numpy(),
            binarized_mask.cpu().numpy(),
            rectangular.squeeze().cpu().numpy(),
            is_correct.cpu().numpy(),
            box_coord_ls,
            classifier_output,
        )


def binarize_logits(logits, p_ls):
    n = logits.shape[0]
    size = 224 * 224
    flat_logits = logits.view(n, -1).detach().cpu().numpy()
    flat_binarized = []
    for j in range(n):
        flat_binarized.append(rankdata(flat_logits[j]) > (1 - p_ls[j]) * size)
    return torch.tensor(np.stack(flat_binarized).reshape(
        n, 1, 224, 224,
    )).float().cuda()


if __name__ == '__main__':
    main(args=RunConfiguration.run_cli_json_prepend())
