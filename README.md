# Investigating and Simplifying Masking-based Saliency Methods for Model Interpretability

This repository contains code for running and replicating the experiments from [Investigating and Simplifying Masking-based Saliency Methods for Model Interpretability](PLACEHOLDER_URL). It is a modified fork of [Classifier-Agnostic Saliency Map Extraction](https://github.com/kondiz/casme), and contains the code originally forked from the [ImageNet training in PyTorch](https://github.com/pytorch/examples/tree/master/imagenet).


## Software requirements

- This repository requires Python 3.7 or later
- If want to use the PxAP metric from [Evaluating Weakly Supervised Object Localization Methods Right](https://arxiv.org/abs/2007.04178):
    - `git clone https://github.com/clovaai/wsolevaluation` and add it to your `PYTHONPATH`
    - `pip install munch` (as well as any other requirements listed [here](https://github.com/clovaai/wsolevaluation#3-code-dependencies))
- If you want to run the Grad-CAM and Guided-backprop saliency methods:
    - `pip install torchray`, or `git clone https://github.com/facebookresearch/TorchRay` and add it to your `PYTHONPATH`
- If you want to use the CA-GAN infiller from [Generative Image Inpainting with Contextual Attention](https://arxiv.org/abs/1801.07892)
    - `git clone https://github.com/daa233/generative-inpainting-pytorch` and add it to your `PYTHONPATH`
    - Download the linked [pretrained model](https://github.com/daa233/generative-inpainting-pytorch#test-with-the-trained-model) for PyTorch
- If you want to use the DFNet infiller from [https://arxiv.org/abs/1904.08060](https://arxiv.org/abs/1904.08060)
    - `git clone git@github.com:hughplay/DFNet.git` and add it to your `PYTHONPATH`
    - Download the linked [pretrained model](https://github.com/hughplay/DFNet#testing) for PyTorch


## Data requirements

- ImageNet dataset should be stored in `IMAGENET_PATH` path and set up in the usual way (separate `train` and `val` folders with 1000 subfolders each). See [this repo](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset) for detailed instructions how to download and set up the dataset.
- ImageNet annotations should be in `IMAGENET_ANN` directory that contains 50000 files named `ILSVRC2012_val_<id>.xml` where `<id>` is the validation image id (for example `ILSVRC2012_val_00050000.xml`). It may be simply obtained by unzipping [the official validation bounding box annotations archive](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_bbox_val_v3.tgz) to `IMAGENET-ANN` directory.
- Bounding box annotations for parts of the training set can downloaded from [here](http://image-net.org/Annotation/Annotation.tar.gz). This is used for our Train-Validation set. 
- If want to use the MaxBoxAcc or PxAP metrics from [Evaluating Weakly Supervised Object Localization Methods Right](https://arxiv.org/abs/2007.04178):
    - Download the relevant datasets in described [here](https://github.com/clovaai/wsolevaluation#2-dataset-downloading-and-license)

## Running the code

We will assume that experiments will be run in the following folder:

```bash
export EXP_DIR=/path/to/experiments
```

### Data Preparation
To facilitate effective subsetting and label shuffling for the ImageNet training set, we write a JSON files containing the paths to the example images, and their corresponding labels. These will be consumed by a modified ImageNet PyTorch Dataset.

Run the following command:

```bash
python3 casme/tasks/imagenet/preproc.py \
    --train_path ${IMAGENET_PATH}/train \
    --val_path ${IMAGENET_PATH}/val \
    --val_annotation_path ${IMAGENET_ANN} \
    --output_base_path ${EXP_DIR}/metadata
```

To use bounding boxes for the Train-Validation set, unzip the downloaded data from [here](http://image-net.org/Annotation/Annotation.tar.gz), and provided an additional argument `--extended_annot_base_path`.

We also condense the bounding box annotations for the validation set into a single JSON. Run the following command:

```bash
python zphang/imagenet_anno_script.py \
    --data_path ${IMAGENET_PATH} \
    --annotation_path ${IMAGENET_ANN} \
    --output_path /gpfs/data/geraslab/zphang/working/190624_new_casme/imagenet_annotation.json
```

### Training

To train a ASME or CASME model, you can run:

```bash
python train_casme.py \
    --train_json ${EXP_DIR}/metadata/train.json \
    --val_json ${EXP_DIR}/metadata/val.json \
    --ZZsrc ./assets/asme.json \
    --masker_use_layers 3,4 \
    --output_path ${EXP_DIR}/runs/ \
    --epochs 60 --lrde 20 \
    --name asme

python train_casme.py \
    --train_json ${EXP_DIR}/metadata/train.json \
    --val_json ${EXP_DIR}/metadata/val.json \
    --ZZsrc ./assets/casme.json \
    --masker_use_layers 3,4 \
    --output_path ${EXP_DIR}/runs/ \
    --epochs 60 --lrde 20 \
    --name casme
```

- The `--ZZsrc` arguments provide JSON files with additional options for the command-line interface. `./assets/asme.json` and `./assets/casme.json` contain options and final hyper-parameters chosen for the ASME and CASME models in the paper. 
- We also only use the 4th and 5th layers from the classifier in the masker model.
- `--train_json` and `--val_json` point to the JSON files containing the paths to the example images, and their corresponding labels, described above.

### Evaluation

To evaluate the model on WSOL metrics and Saliency Metric, run:

```bash
python casme/tasks/imagenet/score_bboxes.py \
    --val_json ${EXP_DIR}/metadata/val.json \
    --mode casme \
    --bboxes_path ${EXP_DIR}/metadata/val_bboxes.json \
    --casm_path ${EXP_DIR}/runs/casme/epoch_XXX.chk \
    --output_path ${EXP_DIR}/runs/casme/epoch_XXX_score1.json \
``` 

where `epoch_XXX.chk` corresponds to the model checkpoint you want to evaluate. Add argument `--eval_mode val` to run on the actual validation set. Note that the mode should be `casme` regardless of whether you are using CASME or ASME models.

To evaluate the model on PxAP, run:

```bash
python zphang/q_wsoleval.py \
    --cam_loader casme \
    --casm_base_path ${EXP_DIR}/runs/casme/epoch_XXX.chk \
    --casme_load_mode specific \
    --dataset OpenImages \
    --dataset_split test
```

## Reference

If you found this code useful, please cite [the following paper](PLACEHOLDER_URL):

Jason Phang, Jungkyu Park, Krzsyztof J. Geras **"Investigating and Simplifying Masking-based Saliency Methods for Model Interpretability."** *arXiv preprint arXiv:XXXX.XXXXX (2020).*
```
@article{phang2020investigating,
  title={Investigating and Simplifying Masking-based Saliency Methods for Model Interpretability},
  author={Phang, Jason and Park, Jungkyu and Geras, Krzysztof J},
  journal={arXiv preprint arXiv:XXXX.XXXXX,
  year={2020}
}
```