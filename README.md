# MoE-TCR: Mixture-of-Experts Framework for pan-specific TCR-Epitope Binding Prediction

This repository provides a framework for training and predicting TCR-epitope binding using the TPNet model. It leverages PyTorch Lightning and WandB for model training, logging, and evaluation.

## Installation

Before running the training or prediction scripts, ensure that all dependencies are installed. You can do so by following the steps below:

```bash
conda create -n moe-tcr python==3.9.16
conda activate moe-tcr
pip install -r requirements.txt
```

Set up the environment with your WandB API key:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
```

Alternatively, you can directly set the WandB API key in the train.py script by modifying the following line:
```python
os.environ["WANDB_API_KEY"] = "<your_wandb_api_key>"  # Input your WandB API key for online logging
```


## Training

The `train.py` script handles the training process, including model checkpoints and early stopping, with support for nested cross-validation.

### Command-line Arguments

The script accepts the following arguments:

- `--config`: Path to the configuration YAML file.
- `--run_name`: Name of the current run.
- `--mode`: Training mode, currently supporting `nested_CV` (nested cross-validation).
- `--online`, `-o`: Run with WandB online logging (default: offline mode).
- `--gpu`: GPU device index to use.
- `--resume_from`: Resume training from a specific checkpoint (optional).

### Example Usage

In the `./configs/*.yaml` directory, we provide multiple configuration files tailored for various training needs. For example, to train the model on the IMMREP22 benchmark, you can use the following command:

```bash
python train.py --config ./configs/moe-tecc-IMMREP22.yaml --run_name moe-tecc-IMMREP22 --mode nested_CV --gpu 0 --online
```

Here's the refined continuation along with a visual representation of the directory structure:

This will generate a directory inside ./log named after your run_name. Within this directory, you’ll find the following structure:

```bash
./log/
   └── <run_name>/
       ├── checkpoints/   # Stores model checkpoint files
       └── <config_name>.yaml  # A copy of the input configuration file
```
The checkpoints subdirectory contains the model's saved parameters, while the YAML file is a copy of the input configuration, making it easy to organize and track experiments for future reference.


#### Model Checkpoints and Early Stopping

The training process automatically saves model checkpoints and uses early stopping to halt training if no improvements are observed after a specified number of epochs. Checkpoints are saved in the format `t_<train_fold>_v_<valid_fold>_{epoch:02d}.ckpt`.

#### Logging

Training runs are logged with WandB. If `--online` is passed, logs are uploaded to the WandB platform; otherwise, they are stored locally.

---

## Model Prediction

The `predict.py` script handles model inference using trained checkpoints.

### Command-line Arguments

The script accepts the following arguments:

- `--config`: Path to the configuration YAML file.
- `--ckpt_path`: Path to the directory or specific checkpoint file(s) for prediction.

### Example Usage

To perform prediction with a trained model, run the following command:

```bash
python predict.py --config <path_to_config> --ckpt_path <path_to_checkpoint>
```

For previously trained models, you can run predictions using the following command:

```bash
python predict.py --config ./log/moe-tecc-IMMREP22/moe-tecc-IMMREP22.yaml --ckpt_path ./log/moe-tecc-IMMREP22/checkpoints
```


#### Output

- **Pickle File**: The results are saved in a `.pickle` file containing the predictions, true targets, and epitope names.
- **Text Files**: Two text files summarizing the overall and per-epitope performance are generated.

## Declaration
It is free for non-commercial use. For commercial use, please contact Mr.Wei Qu and Prof.Shanfeng Zhu (zhusf@fudan.edu.cn).

