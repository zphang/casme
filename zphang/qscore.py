import argparse
import os

from zphang import utils
import casme.tasks.imagenet.score_bboxes as score_bboxes

import zconf


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    mode = zconf.attr()
    base_path = zconf.attr()
    output_path = zconf.attr(default=None)

    workers = zconf.attr(default=4, type=int, help='number of data loading workers (default: 4)')
    batch_size = zconf.attr(default=128, type=int, help='mini-batch size (default: 256)')
    print_freq = zconf.attr(default=10, type=int, help='print frequency (default: 10)')
    break_ratio = zconf.attr(action='store_true', help='break original aspect ratio when resizing')
    not_normalize = zconf.attr(action='store_true', help='prevents normalization')
    pot = zconf.attr(default=1, type=float, help='percent of validation set seen')


def main(args: RunConfiguration):
    if args.mode in ("center", "max"):
        casm_path = None
    else:
        casm_path = utils.find_best_model(args.base_path)
    print("Scoring {}".format(casm_path))

    if not args.output_path:
        output_path = os.path.join(args.base_path, "score.json")
    else:
        output_path = args.output_path

    score_bboxes_args = score_bboxes.RunConfiguration(
        val_json="/gpfs/data/geraslab/zphang/working/1910/03_salmap_prep/imagenet_paths/val.json",
        mode=args.mode,
        bboxes_path='/gpfs/data/geraslab/zphang/working/190624_new_casme/imagenet_annotation.json',
        casm_path=casm_path,
        output_path=output_path,
        workers=args.workers,
        batch_size=args.batch_size,
        print_freq=args.print_freq,
        break_ratio=args.break_ratio,
        not_normalize=args.not_normalize,
        pot=args.pot,
    )
    score_bboxes.main(score_bboxes_args)


if __name__ == '__main__':
    main(args=RunConfiguration.run_cli_json_prepend())
