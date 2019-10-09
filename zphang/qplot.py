import os

from zphang import utils
from casme.tasks.imagenet import plot_casme, plot_icasme

import zconf


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    mode = zconf.attr(required=True)
    base_path = zconf.attr(required=True)


def main(args):
    val_json = "/gpfs/data/geraslab/zphang/working/1910/03_salmap_prep/imagenet_paths/train_val.json"
    best_casm_path = utils.find_best_model(args.base_path)
    plots_path = os.path.join(args.base_path, "plots")
    print("Loading ", best_casm_path)
    print("Writing to ", plots_path)

    if args.mode == "casme":
        plot_casme_args = plot_casme.RunConfiguration(
            val_json=val_json,
            casm_path=best_casm_path,
            plots_path=plots_path,
        )
        plot_casme.main(plot_casme_args)
    elif args.mode == "icasme":
        plot_icasme_args = plot_icasme.RunConfiguration(
            val_json=val_json,
            bboxes_path="/gpfs/data/geraslab/zphang/working/190624_new_casme/imagenet_annotation.json",
            casm_path=best_casm_path,
            plots_path=plots_path,
        )
        plot_icasme.main(plot_icasme_args)
    else:
        raise KeyError(args.mode)


if __name__ == '__main__':
    main(RunConfiguration.run_cli())
