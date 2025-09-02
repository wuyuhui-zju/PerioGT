## System requirements
This package has been tested on the following system:
- Linux: Ubuntu22.04
- CUDA 12.2

## Environment configuration
Clone this repository:
```
git clone https://github.com/wuyuhui-zju/PerioGT.git
```
 Create the environment with Anaconda in few minutes:
```
cd PerioGT
conda env create
```
Activate the created enviroment:
```
conda activate PerioGT
```
## Data preprocessing
Since the calculation of the properties of polymers is time-consuming, we prepare the pre-training data before pre-training:
```
cd PerioGT_common/scripts
bash prepare_pt_dataset.sh
```
The processed files can be downloaded from the Zenodo link given in the muniscript. 
Unzip all four `pretrain_data.zip` parts individually, and merge their contents into the `datasets/` directory.

## Pre-training
We provide a script to pre-train the base model (100M). Switch to the scripts directory.
```
bash pretrain.sh
```
Here we show some option arguments inside the pre-training scripts. The backbone of PerioGT can be specified by `--backbone`argument, and the capacity of a model can be specified by the `--config` argument.
```
python -u -m torch.distributed.run \
    --nproc_per_node=3 \
    --nnodes=1 \
    pretrain.py \
    --backbone $BACKBONE \
    --config $CONFIG \
```
We implement the light and GraphGPS architectures. The pre-training process takes approximately 50 hours on three RTX4090 GPUs, so we provide the pre-trained checkpoint, which you can download from the given link. Download the pretrained weights `checkpoints.zip`and put `checkpoints/pretrained/light/base.pth` in `PerioGT_common/checkpoints/pretrained/light` directory.
## Fine-tuning
After run the pre-training script or download the pre-trained weights, we fine-tune the models on downstream tasks. First, we use `egc` dataset as an example and provide a script to prepare PolymerGraphs:
```
bash prepare_ft_dataset.sh
```
You can change the dataset by specifying the `--dataset` argument in the script. The computation process of prompts is time-consuming, and we provide the processed dataset at link. Please unzip the downloaded `egc.zip` file and place the `egc` directory in the `PerioGT_common/datasets/` directory. Finally, model can be fine-tuned by running:
```
bash finetune.sh
```
The `--mode freeze` argument can be used to fix the weights of backbone.
## Evaluation
After fine-tuning, evaluate the model by running the following script:
```
bash evaluation.sh
```
You can also download our fine-tuned checkpoints from link. Please place the downloaded weights in the `checkpoints/egc` directory.
