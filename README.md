#D-SAM public repository for the paper "Domain Generalization with Domain-Specific Aggregation Modules

## Prerequisites

The code was last tested with:

- Python 3.6.6
- PyTorch 1.2
- torchvision 0.4
- Tensorflow 1.15 (for tensorboard logging)

## Reproducing results on PACS - AlexNet

To reproduce results on PACS using AlexNet, on art_painting split, run:

> python3 main.py job_name --dataset_dir /path/to/PACS/kfold/root --dataset_fn_values art_painting

Change the last argument to run on a different split, mind that the code is using raw kfold image files 
