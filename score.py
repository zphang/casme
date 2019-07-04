import argparse
from bs4 import BeautifulSoup
import numpy as np
import os
import json
import time

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from casme.stats import AverageMeter, StatisticsContainer
from casme.model_basics import casme_load_model, icasme_load_model, get_masks_and_check_predictions


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        return super(ImageFolderWithPaths, self).__getitem__(index), self.imgs[index][0]


def main(args):
    # data loading code
    normalize = transforms.Normalize(
        mean=[0, 0, 0] if args.not_normalize else [0.485, 0.456, 0.406],
        std=[1, 1, 1] if args.not_normalize else [0.229, 0.224, 0.225],
    )

    data_loader = torch.utils.data.DataLoader(
        ImageFolderWithPaths(
            os.path.join(args.data, 'val'), transforms.Compose([
                transforms.Resize([224, 224] if args.break_ratio else 224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]),
        ),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False
    )

    # get score for special cases
    if args.mode == "max":
        model = {'special': 'max'}
    elif args.mode == "center":
        model = {'special': 'center'}
    elif args.mode == "casme":
        model = casme_load_model(args.casm_path)
    elif args.mode == "icasme":
        model = icasme_load_model(args.casm_path)
    else:
        raise KeyError(args.mode)

    results = score(
        args=args,
        model=model,
        data_loader=data_loader,
        ann_paths=args.annotation_path,
    )

    with open(args.output_path, "w") as f:
        f.write(json.dumps(results, indent=2))


def score(args, model, data_loader, ann_paths):
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

        input_, target = input_.numpy(), target.numpy()

        # compute continuous mask, rectangular mask and compare class predictions with targets
        if 'special' in model.keys():
            is_correct = target.ge(0).numpy()
            if model['special'] == 'max':
                continuous = np.ones((args.batch_size, 224, 224))
                rectangular = continuous
            if model['special'] == 'center':
                continuous = np.zeros((args.batch_size, 224, 224))
                continuous[:, :, 33:-33, 33:-33] = 1
                rectangular = continuous
        else:
            continuous, rectangular, is_correct = get_masks_and_check_predictions(input_, target, model)

        # update statistics
        statistics.update(torch.tensor(continuous).unsqueeze(1))

        # image loop
        for idx, path in enumerate(paths):
            # get basic image properties
            ann_path = os.path.join(ann_paths, os.path.basename(path)).split('.')[0] + '.xml'

            if not os.path.isfile(ann_path):
                print("Annotations aren't found. Aborting!")
                return

            with open(ann_path) as f:
                xml = f.readlines()
            anno = BeautifulSoup(''.join([line.strip('\t') for line in xml]), "html5lib")

            size = anno.findChildren('size')[0]
            width = int(size.findChildren('width')[0].contents[0])
            height = int(size.findChildren('height')[0].contents[0])

            category = path.split('/')[-2]

            # get ground truth boxes positions in the original resolution
            gt_boxes = get_ground_truth_boxes(anno, category)
            # get ground truth boxes positions in the resized resolution
            gt_boxes = get_resized_pos(gt_boxes, width, height, args.break_ratio)

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


def get_ground_truth_boxes(anno, category):
    boxes = []
    objs = anno.findAll('object')
    for obj in objs:
        obj_names = obj.findChildren('name')
        for name_tag in obj_names:
            if str(name_tag.contents[0]) == category:
                # fname = anno.findChild('filename').contents[0]
                bbox = obj.findChildren('bndbox')[0]
                xmin = int(bbox.findChildren('xmin')[0].contents[0])
                ymin = int(bbox.findChildren('ymin')[0].contents[0])
                xmax = int(bbox.findChildren('xmax')[0].contents[0])
                ymax = int(bbox.findChildren('ymax')[0].contents[0])

                boxes.append([xmin, ymin, xmax, ymax])
            else:
                print("Aborting!")
                return

    return boxes


def get_resized_pos(gt_boxes, width, height, break_ratio):
    resized_boxes = []
    for box in gt_boxes:
        resized_boxes.append(resize_pos(box, width, height, break_ratio))

    return resized_boxes


def resize_pos(raw_pos, width, height, break_ratio):
    if break_ratio:
        ratio_x = 224/width
        ratio_y = 224/height
        xcut = 0
        ycut = 0
    else:
        if width > height:
            ratio_x = 224/height
            ratio_y = 224/height
            xcut = (width*ratio_x - 224) / 2
            ycut = 0
        else:
            ratio_x = 224/width
            ratio_y = 224/width
            xcut = 0
            ycut = (height*ratio_y - 224) / 2

    semi_cor_pos = [(ratio_x*raw_pos[0] - xcut),
                    (ratio_y*raw_pos[1] - ycut),
                    (ratio_x*raw_pos[2] - xcut),
                    (ratio_y*raw_pos[3] - ycut)]

    return [int(x) for x in semi_cor_pos]


def get_loc_scores(cor_pos, continuous_mask, rectangular_mask):
    xmin, ymin, xmax, ymax = cor_pos
    gt_box_size = (xmax - xmin)*(ymax - ymin)

    xmin_c, ymin_c, xmax_c, ymax_c = [clip(x, 0, 224) for x in cor_pos]

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


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--mode')
    parser.add_argument('--annotation-path', default='',
                        help='path to annotations')
    parser.add_argument('--casm-path', default='',
                        help='model_checkpoint')
    parser.add_argument('--output-path')

    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--break-ratio', action='store_true',
                        help='break original aspect ratio when resizing')
    parser.add_argument('--not-normalize', action='store_true',
                        help='prevents normalization')

    parser.add_argument('--pot', default=1, type=float,
                        help='percent of validation set seen')

    args = parser.parse_args()
    return args


def get_args_dict(args):
    return {
        name: getattr(args, name)
        for name in sorted(vars(args))
    }


if __name__ == '__main__':
    main(get_args())
