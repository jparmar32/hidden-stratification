import json
import numpy as np
import os


train_set = 'cxr_p'
test_set = 'cxr_p'
results_dir = f'/mnt/gaze_robustness_results/gdro/cxr_p/true_subclass_gdro'

seeds = [x for x in range(10)] 
wds = [ 1, .1]

'''

best_subclass_acc = 0.0
best_wd = 0
for wd in wds:
    subclass_acc_means = []
    for cv in seeds: 
        wd_string = str(wd)
        wd_string = wd_string.strip("0")
           
        if wd == 10:
            res_file = f"/mnt/gaze_robustness_results/gdro/cxr_p/wd_10/true_subclass_gdro/seed_{cv}/metrics.json"
        else:
            res_file = f"/mnt/gaze_robustness_results/gdro/cxr_p/wd_{wd_string}/true_subclass_gdro/seed_{cv}/metrics.json"
        
        with open(res_file) as data_file:
            results = json.load(data_file)
            subclass_acc_means.append(results['test']['subclass_rob_acc'])
    mean = np.mean(subclass_acc_means)
    print(f"wd: {wd} with mean subclass acc: {mean}")
    if mean >= best_subclass_acc:
        best_subclass_acc = mean
        best_wd = wd
print(f"best wd: {best_wd} with mean subclass acc: {best_subclass_acc}")

avg_acc_means = []
subclass_acc_means = []

#do overall as well as subclass
results_dir = f'/mnt/gaze_robustness_results/gdro/cxr_p/wd_{best_wd}/true_subclass_gdro'''

results_dir = f'/mnt/gaze_robustness_results/gdro/cxr_p/true_subclass_gdro'
avg_acc_means = []
subclass_acc_means = []

for cv in seeds:
    res_file = os.path.join(results_dir, f"seed_{cv}/metrics.json")

    with open(res_file) as data_file:
        results = json.load(data_file)
        avg_acc_means .append(results['test']['acc'])
        subclass_acc_means .append(results['test']['subclass_rob_acc'])


print(f"\nMean Average Acc: {np.mean(avg_acc_means):.3f}")
print(f"\nStd: {np.std(avg_acc_means):.3f}")

print(f"\nMean Subclass Acc: {np.mean(subclass_acc_means):.3f}")
print(f"\nStd: {np.std(subclass_acc_means):.3f}")