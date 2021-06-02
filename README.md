# Group DRO Implementation

This repository implements the "GEORGE" algorithm from the following paper [No Subclass Left Behind: Fine-Grained Robustness in Coarse-Grained Classification Problems](https://arxiv.org/abs/2011.12945) but for the purpose of the CS329D project we specifically use it to run the group DRO optimizattion problem on our dataset. 

## Setup instructions

Prerequisites: Make sure you have Python>=3.6 and PyTorch>=1.5 installed. Then, install dependencies with:
```bash
pip install -r requirements.txt
```

Next, either add the base directory of the repository to your PYTHONPATH, or run:
```bash
pip install -e .
```

## Running Group DRO Experiments

To train and evaluate a group DRO we can use `./sripts/run_gas.sh` with the following script which performances a hyperparameter search over various values of l2 regularization for the model.

mode=true_subclass_gdro
for wd in 10 1 .1 .01 .001 .0001
    do
    for seed in 0 1 2 3 4 5 6 7 8 9
    do
        python stratification/run_gas.py configs/pmx_config.json mode=$mode seed=$seed classification_config.dataset_config.seed=$seed        exp_dir="/mnt/gaze_robustness_results/gdro/cxr_p/wd_$wd"
    done
done

Then we can use  `./scipts/report_results.py` to collate and report the evaluation results.
