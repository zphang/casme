import numpy as np
import os
import json
import time

import torch
import torchvision.transforms as transforms

from casme.stats import AverageMeter, StatisticsContainer
from casme.model_basics import casme_load_model, get_masks_and_check_predictions
from casme.utils.torch_utils import ImagePathDataset
import casme.tasks.imagenet.utils as imagenet_utils

import zconf
import pyutils.io as io


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    val_json = zconf.attr(help='train_json path')
    mode = zconf.attr(type="str")
    bboxes_path = zconf.attr(help='path to bboxes_json')
    casm_path = zconf.attr(help='model_checkpoint')
    classifier_load_mode = zconf.attr(default="pickled")
    output_path = zconf.attr(help='output_path')

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

    # get score for special cases
    if args.mode == "max":
        model = {'special': 'max'}
    elif args.mode == "center":
        model = {'special': 'center'}
    elif args.mode == "casme":
        model = casme_load_model(args.casm_path, classifier_load_mode=args.classifier_load_mode)
    else:
        raise KeyError(args.mode)

    bboxes = io.read_json(args.bboxes_path)

    results = score(
        args=args,
        model=model,
        data_loader=data_loader,
        bboxes=bboxes,
    )

    with open(args.output_path, "w") as f:
        f.write(json.dumps(results, indent=2))


def score(args, model, data_loader, bboxes):
    if 'special' in model.keys():
        print("=> Special mode evaluation: {}.".format(model['special']))

    # setup meters
    batch_time = 0
    data_time = 0
    f1_meter = AverageMeter()
    f1a_meter = AverageMeter()
    le_meter = AverageMeter()
    om_meter = AverageMeter()
    statistics = StatisticsContainer()

    end = time.time()

    # data loop
    for i, ((input_, target), paths) in enumerate(data_loader):
        if i > len(data_loader)*args.pot:
            break

        data_time += time.time() - end

        # compute continuous mask, rectangular mask and compare class predictions with targets
        if 'special' in model.keys():
            is_correct = target.ge(0).numpy()
            if model['special'] == 'max':
                continuous = np.ones((args.batch_size, 224, 224))
                rectangular = continuous
            elif model['special'] == 'center':
                continuous = np.zeros((args.batch_size, 224, 224))
                continuous[:, :, 33:-33, 33:-33] = 1
                rectangular = continuous
            else:
                raise KeyError(model["special"])
        else:
            continuous, rectangular, is_correct, bboxes = \
                get_masks_and_check_predictions(input_, target, model)

        # update statistics
        statistics.update(torch.tensor(continuous).unsqueeze(1))

        # image loop
        for idx, path in enumerate(paths):
            gt_boxes = bboxes[os.path.basename(path).split('.')[0]]

            # compute localization metrics
            f1s_for_image = []
            ious_for_image = []
            for gt_box in gt_boxes:
                f1_for_box, iou_for_box = get_loc_scores(gt_box, continuous[idx], rectangular[idx])

                f1s_for_image.append(f1_for_box)
                ious_for_image.append(iou_for_box)

            f1_meter.update(np.array(f1s_for_image).max())
            f1a_meter.update(np.array(f1s_for_image).mean())
            le_meter.update(1 - np.array(ious_for_image).max())
            om_meter.update(1 - (np.array(ious_for_image).max() * is_correct[idx]))

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
                  'LE {LE.avg:.3f} ({LE.val:.3f})'.format(
                      i, len(data_loader), batch_time=batch_time, data_time=data_time,
                      F1=f1_meter, F1a=f1a_meter, OM=om_meter, LE=le_meter), flush=True)
            statistics.print_out()

    print('Final:\t'
          'Time {batch_time:.3f}\t'
          'Data {data_time:.3f}\n'
          'F1 {F1.avg:.3f} ({F1.val:.3f})\t'
          'F1a {F1a.avg:.3f} ({F1a.val:.3f})\t'
          'OM {OM.avg:.3f} ({OM.val:.3f})\t'
          'LE {LE.avg:.3f} ({LE.val:.3f})'.format(
                batch_time=batch_time, data_time=data_time, F1=f1_meter, F1a=f1a_meter, OM=om_meter,
                LE=le_meter), flush=True)
    statistics.print_out()

    results = {
        'F1': f1_meter.avg,
        'F1a': f1a_meter.avg,
        'OM': om_meter.avg,
        'LE': le_meter.avg, **statistics.get_dictionary()
    }
    return results


def compute_agg_loc_scores(gt_boxes, continuous_mask, rectangular_mask, is_correct, include_partial=False):
    f1s_for_image = []
    ious_for_image = []
    for box in gt_boxes:
        f1_for_box, iou_for_box = get_loc_scores(
            cor_pos=box,
            continuous_mask=continuous_mask,
            rectangular_mask=rectangular_mask,
        )
        f1s_for_image.append(f1_for_box)
        ious_for_image.append(iou_for_box)

    result = {
        "f1": np.array(f1s_for_image).max(),
        "le": 1 - np.array(ious_for_image).max(),
        "om": 1 - (np.array(ious_for_image).max() * is_correct),
    }
    if include_partial:
        result["f1s_for_image"] = f1s_for_image
        result["ious_for_image"] = ious_for_image

    return result


def get_loc_scores(cor_pos, continuous_mask, rectangular_mask):
    xmin, ymin, xmax, ymax = cor_pos["xmin"], cor_pos["ymin"], cor_pos["xmax"], cor_pos["ymax"]
    gt_box_size = (xmax - xmin)*(ymax - ymin)

    xmin_c, ymin_c, xmax_c, ymax_c = [clip(z, 0, 224) for z in [xmin, ymin, xmax, ymax]]

    if xmin_c == xmax_c or ymin_c == ymax_c:
        return 0, 0

    gt_box = np.zeros((224, 224))
    gt_box[ymin_c:ymax_c, xmin_c:xmax_c] = 1

    f1 = compute_f1(continuous_mask, gt_box, gt_box_size)
    iou = compute_iou(rectangular_mask, gt_box, gt_box_size)

    return f1, 1*(iou > 0.5)


def clip(x, a, b):
    if x < a:
        return a
    if x > b:
        return b

    return x


def compute_f1(m, gt_box, gt_box_size):
    with torch.no_grad():
        inside = (m*gt_box).sum()
        precision = inside / (m.sum() + 1e-6)
        recall = inside / gt_box_size

        return (2 * precision * recall)/(precision + recall + 1e-6)


def compute_iou(m, gt_box, gt_box_size):
    with torch.no_grad():
        intersection = (m*gt_box).sum()
        return intersection / (m.sum() + gt_box_size - intersection)


if __name__ == '__main__':
    main(args=RunConfiguration.run_cli_json_prepend())
