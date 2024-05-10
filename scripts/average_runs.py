import numpy as np
import sys
import os
import json
from pdb import set_trace as bp
sizes = [1,2,3,4,5,6,7,8,9,10]
seeds = [0,1,2,3,4,5]#[0,1,2,3,4,5]
res_path = '/home/prashant/unilm/layoutlmv3/results'
json_path = os.path.join(res_path, 'results_json')
rotate_ang = 3
shift = 10
mod, mod_val = 'scale',4
for sz in sizes:
    precs, f1s, accs, recalls = [], [], [], []
    for sd in seeds:
        try:
            # FUNSD 
            # results = json.loads(open(os.path.join(res_path, f'test-lm-fs-{sz}-{sd}','all_results.json'),'r').read())
            # CORD
            results = json.loads(open(os.path.join(res_path, f'test-lm-cord-fs-{sz}-{sd}','all_results.json'),'r').read())
            precs.append(float(results['test_precision']))
            f1s.append(float(results['test_f1']))
            accs.append(float(results['test_accuracy']))
            recalls.append(float(results['test_recall']))
        except:
            print("sz: {} \t Sd: {}".format(sz, sd))
    save_res = {'precision_mean':np.mean(precs), 'f1_mean':np.mean(f1s), 'accuracy_mean':np.mean(accs), 'recall_mean':np.mean(recalls),
                'precision_std':np.std(precs), 'f1_std':np.std(f1s), 'accuracy_std':np.std(accs), 'recall_std':np.std(recalls)}
    # FUNSD
    # json.dump(save_res, open(os.path.join(json_path,f'test-lm-fs-sz-{sz}-sd-0-5.json'), 'w'))
    # CORD
    json.dump(save_res, open(os.path.join(json_path,f'test-lm-cord-fs-sz-{sz}-sd-0-5.json'), 'w'))


