# SSL Pretraining

## Installation

Ensure you have all the necessary dependencies installed:

 ```bash
pip install -r requirements.txt
```

## Training on Single GPU

Run the following command to train SimCLR on single GPU.

```bash
python scripts/train.py --config <path-to-yaml-config>
```

## Distributed Training on Multiple GPUs

Run the following command to train SimCLR on multiple GPUs.

```bash
torchrun --nproc_per_node=N_GPUs --standalone scripts/multigpu_train_simclr.py --config <path-to-yaml-config>
```

Replace `N_GPUs` with the number of GPUs you want to use and `<path-to-yaml-config>` with the path to your configuration file.

## Configuration File

The YAML configuration file should look something like this:

```yaml
experiment_name: "simclr/cifar10"
method_type: "simclr"
supervision: "SSL"

dataset:
  name: "cifar10"
  path: "./data/"
  num_output_classes: 10

training:
  batch_size: 1024
  num_epochs: 1000
  lr: 0.3
  augmentations_type: "cifar"
  augment_both: True
  save_every: 100
  log_every: 10
  track_performance: True
  multi_gpu: True
  world_size: 2

model:
  encoder_type: "resnet50"
  pretrained: False
  width_multiplier: 2
  hidden_dim: 2048
  projection_dim: 64

loss:
  temperature: 0.5

evaluation:
  K: [5, 10, 20]
  checkpoints_dir: "experiments/simclr/cifar10/checkpoints/"
  perform_knn: False
  perform_cdnv: True
  perform_nccc: True
  perform_linear: False
  perform_tsne: False
```

### Key Configuration Parameters

- **Dataset**: Defines dataset path and number of classes.
- **Training**: Specifies batch size, epochs, learning rate, and augmentation strategy. The augmentation strategies can be defined at `utils/augmentations.py`
- **Model**: Determines encoder type, width multiplier, and projection head dimensions.
- **Loss**: Defines contrastive loss temperature. Set `$\tau = 0.5$` for CIFAR and `$\tau = 0.1` for Imagenet
- **Evaluation**: Sets evaluation methods and checkpoint directory.

Modify these settings based on your specific requirements. Happy pretraining! 🚀

---

## Experiment Results: SimCLR Pretraining on CIFAR-10

In this section, we present the results of multiple experiments conducted with different hyperparameter configurations. We evaluate the models using several metrics: CDNV (Contrastive Distance Norm Validation), NCCC (Normalized Cross-Correlation), Training Accuracy, and Test Accuracy. All experiments were conducted with a common set of training configurations unless specified otherwise.

Below are the results for the experiments conducted with different hyperparameter settings. Training and Test Accuracies are reported after Linear Probing experiments.

| Batch Size | LR  | GPUs | Projection Dim | Augmentation   | CDNV | NCCC | Training Acc. | Test Acc. |
|---|---|---|---|---|---|---|---|---|
| 1024       | 0.3 | 2    | 128             | Type #1 | | |          |      |
| 1024       | 0.3 | 2    | 64            | Type #1 | | |          |      |
| 1024       | 0.3 | 2    | 64             | Type #2 | | |         |      |

`Type #1`: Color-jitter with `p = 1.0` and no grayscale \
`Type #2`: Color-jitter with `p = 0.8` + grayscale
