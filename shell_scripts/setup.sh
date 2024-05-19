# Setup conda environment
conda create --name lager python=3.7
conda activate lager
pip install -r requirements.txt
pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
pip install -e .

# Download the FUNSD and CORD datasets (used by one of the LAGER heuristics)
$path=$(pwd)
gdown --id 1rplB2l2a_T4qIzG077nW5miw8Fg6nvzF -O $path/data/DocDatasets.zip

