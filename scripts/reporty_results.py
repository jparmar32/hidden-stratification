import json
import numpy as np
import os


train_set = 'cxr_p'
test_set = 'cxr_p'
results_dir = f'/mnt/gaze_robustness_results/gdro/cxr_p/erm'

seeds = [x for x in range(10)] 
lrs = [.1, .01, .001, .0001]
wds = [1, .1, .01, .001, .0001, 0]

avg_acc_means = []
subclass_acc_means = []

#do overall as well as subclass

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