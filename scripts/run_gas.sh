cd ..
mode=true_subclass_gdro
for seed in 101 102 103 104 105
do
python stratification/run_gas.py configs/pmx_config.json mode=$mode seed=$seed classification_config.dataset_config.seed=$seed
done