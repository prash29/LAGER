# LAGER - Layout Aware Graph-based Entity Recognition model

## Long-Paper accepted to LREC-COLING 2024! 

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

```bash shell_scripts/run_funsd.sh``` or ```bash shell_scripts/run_cord.sh```

### run_funsd.sh parameters explanation
<!--- Parameters in the run_funsd.sh file --->
The `run_funsd.sh` (same as `run_cord.sh`) file contains several arguments that you can modify based on how you want to fine-tune the LAGER model. Here's a brief explanation of some of them

- `--sz`: Specifies the few-shot size
- `--sd`: Specifies the few-shot seed
- `--heuristic`: Specifies which heuristic to use - _nearest_, _angles_ or _baseline_. These are based on the k-nearest neighbors in space, k-nearest neighbors at multiple angles or the vanilla baseline respectively
- `--deg`: Specifies the degree of the graph that you construct
- `--hd`: Specifies the number of attention heads in the GAT
- `--output_dir`: Specifies the directory path where the trained model and evaluation results will be saved.

Feel free to modify these parameters according to your specific requirements.

<!--- End of Parameters in the run_funsd.sh file --->

### Other important files

- `data/path_config.json` : JSON file specifying paths of important aspects such as input instances for the few_shot sizes or seeds, results, etc. For the CORD dataset, `data/path_config_cord.json` is used.
- `data/gat_params.json`: JSON file specifying the parameters for the Graph Attention Network (GAT) model used in the code. It contains configuration settings such as the number of attention heads, the degree of the graph, the angle (theta) used in the k-nearest neighbors at multiple angles heuristic, etc.


## To test image manipulation (shifting, rotation or scaling)


## Acknowledgements
This code utilizes the setup based on [Unilm](https://github.com/microsoft/unilm/tree/master)
For the Graph Attention Network (GAT) model, we use the PyTorch implementation of GAT - [pyGAT][https://github.com/Diego999/pyGAT/tree/master] by @Diego999 

### Paper Abstract:
Recent advances of incorporating layout information, typically bounding box coordinates, into pre-trained language models have achieved significant performance in entity recognition from document images. Using coordinates can easily model the absolute position of each token, but they might be sensitive to manipulations in document images (e.g., shifting, rotation or scaling), especially when the training data is limited in few-shot settings. In this paper, we propose to further introduce the topological adjacency relationship among the tokens, emphasizing their relative position information. Specifically, we consider the tokens in the documents as nodes and formulate the edges based on the topological heuristics from the k-nearest bounding boxes. Such adjacency graphs are invariant to affine transformations including shifting, rotations and scaling. We incorporate these graphs into the pre-trained language model by adding graph neural network layers on top of the language model embeddings, leading to a novel model LAGER. Extensive experiments on two benchmark datasets show that LAGER significantly outperforms strong baselines under different few-shot settings and also demonstrate better robustness to manipulations.


