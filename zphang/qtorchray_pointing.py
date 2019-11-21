import sys
import zconf


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    method = zconf.attr(required=True)
    dataset = zconf.attr(required=True)


def main(args):
    sys.path = [
        '/gpfs/data/geraslab/zphang/code/TorchRay',
        '/gpfs/data/geraslab/zphang/code/BreastScreening/code',
        '/gpfs/data/geraslab/zphang/code/casme',
        '/gpfs/data/geraslab/zphang/code/othergits',
        '/gpfs/data/geraslab/zphang/code/zutils',
        '/gpfs/share/skynet/apps/anaconda3/envs/wmlce_env_1.6.2/lib/python37.zip',
        '/gpfs/share/skynet/apps/anaconda3/envs/wmlce_env_1.6.2/lib/python3.7',
        '/gpfs/share/skynet/apps/anaconda3/envs/wmlce_env_1.6.2/lib/python3.7/lib-dynload',
        '',
        '/gpfs/share/skynet/apps/anaconda3/envs/wmlce_env_1.6.2/lib/python3.7/site-packages',
        '/gpfs/share/skynet/apps/anaconda3/envs/wmlce_env_1.6.2/lib/python3.7/site-packages/tflms-2.0.2-py3.7.egg',
        '/gpfs/share/skynet/apps/anaconda3/envs/wmlce_env_1.6.2/lib/python3.7/site-packages/IPython/extensions',
        '/gpfs/home/zp489/.ipython',
    ]
    import examples.attribution_benchmark as attribution_benchmark
    experiment = attribution_benchmark.Experiment(
        series='attribution_benchmarks',
        method=args.method,
        arch='resnet50',
        dataset=args.dataset,
        chunk=None,
        root="/gpfs/data/geraslab/zphang/working_v2/1911/18_torchray_test/series",
    )
    attribution_benchmark.ExperimentExecutor(experiment, log=0).run()


if __name__ == '__main__':
    main(RunConfiguration.run_cli())
