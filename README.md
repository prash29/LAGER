# LAGER - Layout Aware Graph-based Entity Recognition model

## Long-Paper accepted to LREC-COLING 2024

This is the PyTorch implemention for the LREC-COLING '24 Long-Paper: [Towards Few-shot Entity Recognition in Document Images: A Graph Neural Network Approach Robust to Image Manipulation](https://arxiv.org/abs/2305.14828)

## Quick Setup

```
git clone https://github.com/prash29/LAGER.git
cd LAGER
bash shell_scripts/setup.sh
```

Check `requirements.txt` for the required packages. Please let us know if there are any issues!

This repository uses LayoutLMv3 as the backbone model and is based on the setup in the [unilm](https://github.com/microsoft/unilm/tree/master/layoutlmv3) repository

## How to Run?

Inside the `shell_scripts` directory, `run_funsd.sh` or `run_cord.sh` allow you to train the LAGER model based on the few-shot sizes and other parameters defined

```
bash shell_scripts/run_funsd.sh
```

or

```
bash shell_scripts/run_cord.sh
```

### Explanation about the run_funsd.sh parameters
<!--- Parameters in the run_funsd.sh file --->
The `run_funsd.sh` (same as `run_cord.sh`) file contains several parameters that you can modify based on how you want to fine-tune the LAGER model. Here's a brief explanation of some of them

- `--sz`: Specifies the few-shot size
- `--sd`: Specifies the few-shot seed
- `--heuristic`: Specifies which heuristic to use - _nearest_, _angles_ or _baseline_. These are based on the k-nearest neighbors in space, k-nearest neighbors at multiple angles or the vanilla baseline respectively
- `--deg`: Specifies the degree of the graph that you construct
- `--hd`: Specifies the number of attention heads in the GAT
- `--output_dir`: Specifies the directory path where the trained model and evaluation results will be saved.

Feel free to modify these parameters according to your specific requirements.

<!--- End of Parameters in the run_funsd.sh file --->

### Other important files

- `data/path_config.json` : JSON file specifying paths for things such as input instances for the few_shot sizes or seeds, results, etc. For the CORD dataset, `data/path_config_cord.json` is used.
- `data/gat_params.json`: JSON file specifying the parameters for the Graph Attention Network (GAT) model used in the code. It contains configuration settings such as the number of attention heads, the degree of the graph, the angle (theta) used in the k-nearest neighbors at multiple angles heuristic, etc.

## To test image manipulation (shifting, rotation or scaling)

We show that existing coordinate based layout aware models are not as robust as LAGER that exploits the topological/spatial relationship of the tokens. To this effect, we aim to simulate three scenarios in which the document images are not in ideal, regular conditions: shifting, rotation or scaling. Note that there is no fine-tuning that happens here, just evaluation of an already fine-tuned model. To do this, you can run

```
bash shell_scripts/test_funsd.sh
``` 
or 
```
bash shell_scripts/test_cord.sh

```

### Explanation about additional parameters in test_funsd.sh

There are some parameters in the `test_funsd.sh` (same as `test_cord.sh`) in addition to the `run_funsd.sh` or `run_cord.sh` file contains that you can modify based on how you want to test the results on image manipulation. Here's a brief explanation of them:

- `--manip_type`: Specifies the type of image manipulation - _shift_, _rotate_ or _scale_
- `--manip_param`: Specifies the magnitude of the image manipulation. For e.g. if `manip_type_ = rotate`, and `manip_param = 5`, this implies a rotation of the document image by 5&deg; . This parameter is an additional loop in the script in case you want to run experiments for different values.
- `dir_path` : Specifies the path of the results directory that contains the fine-tuned model for which the image manipulation is to be performed.


## Acknowledgements

This code utilizes the setup based on [Unilm](https://github.com/microsoft/unilm/tree/master)
For the Graph Attention Network (GAT) model, we use the PyTorch implementation of GAT - [pyGAT](https://github.com/Diego999/pyGAT/tree/master) by [@Diego999](https://github.com/diego999)
