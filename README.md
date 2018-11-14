# D-SAM

This is the implementation of the D-SAM architecture for the DLIC2018 (PoliMi) course project paper "Domain Generalization with Domain-Specific Aggregation Modules"

The code structure is inspired by tensorflow's slim models repository:
https://github.com/tensorflow/models/tree/master/research/slim

### Prerequisites

* python 3.x
* Pytorch 4.0 (older versions will break the code)
* Tensorflow 1.x (for tensorboard)

### Installation

Clone this repository to your local workspace folder

## Prepararing the Datasets

* PACS: download from http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017
* OfficeHome: download from http://hemanthdv.org/OfficeHome-Dataset/

Unzip the files to your local data folder

### Usage

To run experiments use the main script

## main.py args

```
usage: main.py [-h] [--logdir LOGDIR] [--dataset_name DATASET_NAME]
               [--num_domains NUM_DOMAINS] [--dataset_dir DATASET_DIR]
               [--dataloader DATALOADER] [--model_name MODEL_NAME]
               [--pretrained PRETRAINED] [--transforms_name TRANSFORMS_NAME]
               [--training_fn TRAINING_FN] [--im_size IM_SIZE]
               [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE]
               [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY]
               [--num_epochs NUM_EPOCHS] [--step_size STEP_SIZE]
               [--gamma GAMMA] [--num_workers NUM_WORKERS]
```

* --logir: where to log tensorboard outputs and saved models
* --dataset_name: $DATASET_NAME:$TEST_DOMAIN. The name of the dataset, followed by the held out testing domain (ex. if training on PACS, with the held out domain being sketch, this field will be "PACS:sketch")
* --dataset_dir: The root directory of the dataset. The script creates symbolic links for training / validation sets
* --num_domains: number of training sources (defaults to 3 for PACS and OfficeHome)
* --dataloader: The dataloader function to use, defaults to DSAM_dataloader
* --model_name: Name of the architecture to use, shoul be one of [deepall_resnet18, DSAM_resnet18, deepall_alexnet, DSAM_alexnet]
* --pretrained: wheter to initialize the weights from an ImageNet pretrained checkpoint
* --transforms_name: name of the preprocessing function to use (defaults to DSAM_transforms)
* --training_fn: name of the training function to use (defaults to DSAM_training)
* --im_size: size of the input crop
* --batch_size: batch size per training domain (the total batch size for a single training step is batch_size * num_domains)
*  training arguments: main.py uses SGD with momentum, with a stepdown learning rate schedule



## Authors

* **Antonio D'Innocente** 

# D-SAM_public
