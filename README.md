# COMPSCI685 Natural Language Processing - Project

This codebase is a project implementation for the [COMPSCI685 Course](https://people.cs.umass.edu/~hschang/cs685/). 

We implement Deep Learning models for Symbolic Mathematics datset, and show that `TreeReg` auxiliary loss along with the standard language modelling loss is better for this type of structured datasets.

This project repository will have: Dataset genertaion, Model implementation, trainging and evaluation code files.

## Contributors
- [Sri Ram Bandi](https://github.com/srirambandi)
- [Huy Ngo](https://github.com/hdngo)
- [Eic Englehart](https://github.com/bob80333)

## Repository Overview
### Datasets
The `gen_dataset` directory has the required scripts to develop `Dataset_1` and hosts the filtered `Dataset_2` as well.

### Training and Evaluation
The `train_and_eval_paper_dataset*.ipynb` notebooks have the training and evaluation codes for the `baseline` model for the `Dataset_1` and `Dataset_2`. Accordingly, he `train_and_eval_treereg_paper_dataset*.ipynb` notebooks have the training and evaluation codes for the `TreeReg` model for the `Dataset_1` and `Dataset_2`. We have a total of 4 notebook, 2 for `Dataset_1` and 2 for `Dataset_2`, where each dataset has the `bassline` and `TreeReeg` implementations and trainings.

### Checkpoints
The `checkpoints` directors holds the `basleine` and `TreeReg` models' 50th `epoch` checkpoints, which can be loaded for inference.

### Results
The `figs` directors holds the training and evaluation loss graphs we observed during the experiments.

### Reports
This repository also holds our proposal and final reports for this project with our findings.
