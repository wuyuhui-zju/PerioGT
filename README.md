

## Environment configuration
Clone this repository:
```
git clone https://github.com/wuyuhui-zju/PerioGT.git
```
 Create the enviroment with Anaconda:
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
The processed files can be downloaded from [link](https://doi.org/10.5281/zenodo.12705754). Place the files downloaded from the `pretrain` directory into the `datasets` directory.
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
We implement the light and GraphGPS architectures. We also provide pre-trained models, which you can download from [link](https://doi.org/10.5281/zenodo.12705754). Download the pretrained weights `base.pth`and put it in `PerioGT_common/checkpoints/pretrained/light` directory.
## Fine-tuning
After run the pre-training script or download the pre-trained weights, we fine-tune the models on downstream tasks. First, we use `egb` dataset as an example and provide a script to prepare PolyGraphs:
```
bash prepare_ft_dataset.sh
```
You can change the dataset by specifying the `--dataset` argument in the script. Notably, the computation process of prompts is time-consuming, and we provide the processed dataset at [link](https://doi.org/10.5281/zenodo.12705754). Or you can also use the version without prompt by specifying `--no_prompt` argument to save time, which in some cases also can get close to the results. Please place the downloaded `egb` file in the `datasets` directory. Finally, model can be fine-tuned by running:
```
bash finetune.sh
```
The `--mode freeze` argument can be used to fix the weights of backbone.
## Evaluation
After fine-tuning, evaluate the model by running the following script:
```
bash evaluation.sh
```
You can also download our fine-tuned model from [link](https://doi.org/10.5281/zenodo.12705754). Please place the downloaded weights in the `checkpoints/egb` directory.
