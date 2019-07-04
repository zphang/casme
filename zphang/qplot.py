import argparse
import glob
import os

from zphang import utils
import plot_icasme
import plot_casme


def main(args):
    args.data = '/gpfs/data/geraslab/Krzysztof/unpacked_imagenet'
    print("{}/save/*.chk".format(args.base_path))
    args.casm_path = utils.find_best_model(args.base_path)
    args.plots_path = os.path.join(args.base_path, "plots")
    print("Loading ", args.casm_path)
    print("Writing to ", args.plots_path)
    if args.mode == "casm":
        plot_casme.main(args)
    elif args.mode == "icasm":
        plot_icasme.main(args)
    else:
        raise KeyError(args.mode)


def get_args(*raw_args):
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('mode')
    parser.add_argument('base_path')

    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--resize', default=256, type=int,
                        help='resize parameter (default: 256)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        help='mini-batch size (default: 128)')

    parser.add_argument('--columns', default=7, type=int,
                        help='number of consecutive images plotted together,'
                             ' one per column (default: 7, recommended 4 to 7)')
    parser.add_argument('--plots', default=16, type=int,
                        help='number of different plots generated (default: 16, -1 to generate all of them)')
    parser.add_argument('--seed', default=931001, type=int,
                        help='random seed that is used to select images')
    parser.add_argument('--plots-path', default='',
                        help='directory for plots')

    args = parser.parse_args(*raw_args)

    if args.columns > args.batch_size:
        args.columns = args.batch_size
    return args


if __name__ == '__main__':
    main(get_args())
