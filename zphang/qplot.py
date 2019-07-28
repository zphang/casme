import os

from zphang import utils
from casme.tasks.imagenet import plot_casme, plot_icasme

import zconf


def main(args):
    data_path = '/gpfs/data/geraslab/Krzysztof/unpacked_imagenet'
    best_casm_path = utils.find_best_model(args.base_path)
    plots_path = os.path.join(args.base_path, "plots")
    print("Loading ", best_casm_path)
    print("Writing to ", plots_path)
    if args.mode == "casme":
        args.data = data_path
        args.casm_path = best_casm_path
        args.plots_path = plots_path
        plot_casme.main(args)
    elif args.mode == "icasme":
        run_config = plot_icasme.RunConfiguration(
            data=data_path,
            bboxes_path="/gpfs/data/geraslab/zphang/working/190624_new_casme/imagenet_annotation.json",
            casm_path=best_casm_path,
            plots_path=plots_path,
        )
        plot_icasme.main(run_config)
    else:
        raise KeyError(args.mode)


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    mode = zconf.attr(required=True)
    base_path = zconf.attr(required=True)


if __name__ == '__main__':
    main(RunConfiguration.run_cli())
