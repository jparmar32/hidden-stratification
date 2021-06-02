#mode=true_subclass_gdro
mode=erm
for wd in 10 1 .1 .01 .001 .0001
    do
    for seed in 0 1 2 3 4 5 6 7 8 9
    do
        python stratification/run_gas.py configs/pmx_config.json mode=$mode seed=$seed classification_config.dataset_config.seed=$seed exp_dir="/mnt/gaze_robustness_results/gdro/cxr_p/wd_$wd"
    done
done