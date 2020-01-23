import os

from zphang import utils
import casme.tasks.imagenet.score_bboxes as score_bboxes

import zconf


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    mode = zconf.attr()
    base_path = zconf.attr(default="best")
    eval_mode = zconf.attr(default="train_val")
    output_path = zconf.attr(default=None)

    classifier_load_mode = zconf.attr(default="pickled")
    workers = zconf.attr(default=4, type=int, help='number of data loading workers (default: 4)')
    batch_size = zconf.attr(default=128, type=int, help='mini-batch size (default: 256)')
    print_freq = zconf.attr(default=10, type=int, help='print frequency (default: 10)')
    break_ratio = zconf.attr(action='store_true', help='break original aspect ratio when resizing')
    not_normalize = zconf.attr(action='store_true', help='prevents normalization')
    pot = zconf.attr(default=1, type=float, help='percent of validation set seen')


def resolve_data_paths(eval_mode):
    if eval_mode == "train_val":
        val_json = "/gpfs/data/geraslab/zphang/working/1912/29_new_metadata/train_val.json"
        bboxes_path = "/gpfs/data/geraslab/zphang/working/1912/29_new_metadata/train_val_bboxes.json"
    elif eval_mode == "val":
        val_json = "/gpfs/data/geraslab/zphang/working/1912/29_new_metadata/val.json"
        bboxes_path = "/gpfs/data/geraslab/zphang/working/190624_new_casme/imagenet_annotation.json"
    else:
        raise KeyError(eval_mode)
    return val_json, bboxes_path


def run_scoring(args: RunConfiguration,
                mode, casm_path, output_path, bboxes_path, val_json):
    score_bboxes_args = score_bboxes.RunConfiguration(
        val_json=val_json,
        bboxes_path=bboxes_path,
        mode=mode,
        casm_path=casm_path,
        classifier_load_mode=args.classifier_load_mode,
        output_path=output_path,
        workers=args.workers,
        batch_size=args.batch_size,
        print_freq=args.print_freq,
        break_ratio=args.break_ratio,
        not_normalize=args.not_normalize,
        pot=args.pot,
    )
    score_bboxes.main(score_bboxes_args)


def run_single(args: RunConfiguration,
               mode, casm_path, bboxes_path, val_json):
    if not args.output_path:
        if args.classifier_load_mode != "pickled":
            raise AssertionError("need explicit output path")
        output_path = os.path.join(args.base_path, "score.json")
    else:
        output_path = args.output_path

    run_scoring(
        args=args,
        mode=mode,
        casm_path=casm_path,
        output_path=output_path,
        bboxes_path=bboxes_path,
        val_json=val_json,
    )


def run_all(args: RunConfiguration,
            mode, casm_path_ls, bboxes_path, val_json):
    if not args.output_path:
        output_base_path = os.path.join(args.base_path, "all_scores")
    else:
        output_base_path = args.output_path
    os.makedirs(output_base_path, exist_ok=True)

    output_path_ls = []
    print("Scoring:")
    for casm_path in casm_path_ls:
        file_name = os.path.split(casm_path)[-1].split(".")[0]
        output_path = os.path.join(output_base_path, f"{file_name}_score.json")
        print(f"    From: {casm_path}")
        print(f"    To:   {output_path}")
        output_path_ls.append(output_path)

    for casm_path, output_path in zip(casm_path_ls, output_path_ls):
        run_scoring(
            args=args,
            mode=mode,
            casm_path=casm_path,
            output_path=output_path,
            bboxes_path=bboxes_path,
            val_json=val_json,
        )


def main(args: RunConfiguration):
    val_json, bboxes_path = resolve_data_paths(args.eval_mode)

    if args.mode in ("center", "max", 'ground_truth', 'none'):
        os.makedirs(os.path.split(args.output_path)[0], exist_ok=True)
        run_single(
            args=args,
            mode=args.mode,
            casm_path=None,
            bboxes_path=bboxes_path,
            val_json=val_json,
        )
    elif args.mode == "best":
        casm_path = utils.find_best_model(args.base_path)
        print(f"Scoring: {casm_path}")
        run_single(
            args=args,
            mode="casme",
            casm_path=casm_path,
            bboxes_path=bboxes_path,
            val_json=val_json,
        )
    elif args.mode == "all":
        casm_path_ls = utils.get_all_models(args.base_path)
        run_all(
            args=args,
            mode="casme",
            casm_path_ls=casm_path_ls,
            bboxes_path=bboxes_path,
            val_json=val_json,
        )
    else:
        raise KeyError(args.mode)


if __name__ == '__main__':
    main(args=RunConfiguration.run_cli_json_prepend())
