import os

from zphang import utils
from casme.tasks.imagenet import plot_casme

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
            val_json="/gpfs/data/geraslab/zphang/working/1912/29_new_metadata/train_val.json",
            bbox_json="/gpfs/data/geraslab/zphang/working/1912/29_new_metadata/train_val_bboxes.json",
            casm_path=best_casm_path,
            plots_path=plots_path,
        )
        plot_casme.main(plot_casme_args)
    else:
        raise KeyError(args.mode)


if __name__ == '__main__':
    main(RunConfiguration.run_cli())
