# Validating the bound during training

To get the plots generated in figure 1, run the following command:

```bash
python evaluate.py --config <path-to-config-file> --ckpt_path <path-to-ckpts-dir> --output_path <path-to-log-csv-file> --perform_loss_diff True
```

### Notes:

- currently, we set `temperature=1` during evaluation
- `ckpt_path` expects a directory and not a single .pth file
- if output logs exist, this function will not overwrite but will update any missing epochs' information

# Few-Shot Error Analysis

This experiment involves two part: NCCC and LP (Linear Probing)

## NCCC

Run the following command for NCCC analysis:

```bash
python scripts/evaluate.py --config <path> --ckpt_path <path> --ckpt_path_nscl <path> --output_path <path>
```

For example,

```bash
python scripts/nccc_eval.py --config configs/simclr_DCL_cifar10_b1024.yaml --ckpt_path experiments/simclr/cifar10_dcl/checkpoints/ --ckpt_path_nscl experiments/simclr/cifar10_nscl/checkpoints/ --output_path logs/cifar10/
```

### Notes:

- `ckpt_paths` are directories, the code automatically selects the latest model, i.e., more recent epoch.
- You can find logs in `few_shot_analysis.csv` and reproduce NCCC error curves shown in figure 4.

## Linear Probing

For linear probing, run the following command:

```bash
python scripts/linear_probe.py --config <path-to-config-file> --ckpt_path <path-to-ckpt-dir> --output_path <path-to-save-logs> --N <n_samples>
```

For example,

```bash
python scripts/linear_probe.py --config configs/simclr_DCL_cifar10_b1024.yaml --ckpt_path experiments/simclr/cifar10_dcl/checkpoints/ --output_path logs/cifar10/ --N 500
```

### Notes:

- You can find logs in `few_shot_analysis_{supervision}.csv` where `supervision` is set in config file
- Using logs, you can reproduce LP error curves shown in figure 4.
- Again, `ckpt_paths` are directories, the code automatically selects the latest model, i.e., more recent epoch.

# Validating the bound as a function of C

To train many models for different values of C:

```bash
python exp2_multigpu_train_simclr.py --config <path-to-config-file>
```

To evaluate those models:

```bash
python exp2_eval.py --config <path-to-config-file>
```

### Notes:

- in the above commands, the config file should also have `classes_groups`
- `TODOs:` add arguments for logs_file.