import json
import pandas as pd
import os
from collections import defaultdict

results_path = '/home/prashant/unilm/layoutlmv3/results/results_json'

cols = ['model','sz']
cols.extend([f'{x}_{y}' for x in ['FUNSD','CORD'] for y in ['precision','recall','f1']])
df = pd.DataFrame(columns=cols)
sizes = [i for i in range(1,11)]
seeds = [i for i in range(6)]


def get_row(model,sz, funsd_results, cord_results=None):
    
    def format_num(num):
        return round(num*100, 2)
    if cord_results==None:
        cord_results = defaultdict(lambda:0)
    if funsd_results==None:
        funsd_results = defaultdict(lambda:0)
    row = [model,sz, "{}$\pm${}".format(format_num(funsd_results['precision_mean']), format_num(funsd_results['precision_std'])),
           "{}$\pm${}".format(format_num(funsd_results['recall_mean']), format_num(funsd_results['recall_std'])),
           "{}$\pm${}".format(format_num(funsd_results['f1_mean']), format_num(funsd_results['f1_std'])),
           "{}$\pm${}".format(format_num(cord_results['precision_mean']), format_num(cord_results['precision_std'])),
           "{}$\pm${}".format(format_num(cord_results['recall_mean']), format_num(cord_results['recall_std'])),
           "{}$\pm${}".format(format_num(cord_results['f1_mean']), format_num(cord_results['f1_std']))]
    return row
mod, mod_val = 'scale', 4
for sz in sizes:
    funsd_results = json.load(open(os.path.join(results_path,f'test-lm-fs-sz-{sz}-sd-{seeds[0]}-{seeds[-1]}.json'),'r'))
    cord_results  = json.load(open(os.path.join(results_path,f'test-lm-cord-fs-sz-{sz}-sd-{seeds[0]}-{seeds[-1]}.json'),'r'))
    df.loc[len(df)] = get_row('FS',sz, funsd_results, cord_results)
    
    funsd_results = None
    cord_results = None
    # funsd_results = json.load(open(os.path.join(results_path,f'test-roberta-gat-closest-sz-{sz}-h4-d4-sd-{seeds[0]}-{seeds[-1]}.json'),'r'))
    # cord_results = json.load(open(os.path.join(results_path,f'test-roberta-cord-gat-closest-sz-{sz}-h4-d4-sd-{seeds[0]}-{seeds[-1]}.json'),'r'))
    df.loc[len(df)] = get_row('Gat-closest-h4-d4',sz, funsd_results, cord_results)

    # funsd_results = json.load(open(os.path.join(results_path,f'test-layoutlmv3-gat-closest-h8-d4-sz-{sz}-sd-{seeds[0]}-{seeds[-1]}.json'),'r'))
    # # cord_results = json.load(open(os.path.join(results_path,f'test-layoutlmv3-cord-gat-closest-sz-{sz}-h4-d4-sd-{seeds[0]}-{seeds[-1]}.json'),'r'))
    # df.loc[len(df)] = get_row('Gat-closest-h8-d4',sz, funsd_results, None)

    # funsd_results = json.load(open(os.path.join(results_path,f'test-layoutlmv3-gat-45-angles-v2-h4-d4-sz-{sz}-sd-{seeds[0]}-{seeds[-1]}.json'),'r'))
    # # cord_results = json.load(open(os.path.join(results_path,f'test-layoutlmv3-cord-gat-60-angles-v2-{sz}-h4-d4-sd-{seeds[0]}-{seeds[-1]}.json'),'r'))
    # df.loc[len(df)] = get_row('Gat-45-angles-h4-d4',sz, funsd_results, None)

    funsd_results = None
    cord_results = None
    # funsd_results = json.load(open(os.path.join(results_path,f'test-roberta-gat-angles-sz-{sz}-h4-d4-sd-{seeds[0]}-{seeds[-1]}.json'),'r'))
    # cord_results = json.load(open(os.path.join(results_path,f'test-roberta-cord-gat-angles-sz-{sz}-h4-d4-sd-{seeds[0]}-{seeds[-1]}.json'),'r'))
    df.loc[len(df)] = get_row('Gat-60-angles-h4-d4',sz, funsd_results, cord_results)

csv_name = f'results_funsd_cord_lm.csv'
df.to_csv(csv_name, index=False)