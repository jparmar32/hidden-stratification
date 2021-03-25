cd ..
for seed in 103 104 105
do
python stratification/run_gas.py configs/pmx_config.json seed=$seed classification_config.dataset_config.seed=$seed
done