import argparse
import os

from zphang import utils
from casme.tasks.imagenet.score_bboxes import main as score_main


def main(args):
    args.data = '/gpfs/data/geraslab/Krzysztof/unpacked_imagenet'
    args.bboxes_path = '/gpfs/data/geraslab/zphang/working/190624_new_casme/imagenet_annotation.json'
    if args.mode not in ("center", "max"):
        args.casm_path = utils.find_best_model(args.base_path)
    print("Scoring {}".format(args.casm_path))
    if not args.output_path:
        args.output_path = os.path.join(args.base_path, "score.json")
    score_main(args)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('base_path')
    parser.add_argument('-o', '--output_path', default=None)

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


if __name__ == '__main__':
    main(get_args())
