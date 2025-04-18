# Validating the bound during training

To get the plots generated in figure 1, run the following command:

```bash
python evaluate.py --config <path-to-config-file> --ckpt_path <path-to-ckpts-dir> --output_path <path-to-log-csv-file> --perform_loss_diff True
```

### Notes:
- currently, we set `temperature=1` during evaluation
- `ckpt_path` expects a directory and not a single .pth file
- if output logs exist, this function will not overwrite but will update any missing epochs' information

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


# Few-Shot Error Analysis
This experiment involves two part: NCCC and LP (Linear Probing)

For NCCC, run:
```bash
python scripts/evaluate.py --config <path> --ckpt_path <path> --ckpt_path_nscl <path> --output_path <path>
```

### Notes:
- `ckpt_paths` are again just directories, the code automatically selects the latest model

For Linear Probing, run
```bash
python scripts/linear_probe.py --config <path-to-config-file> --ckpt_path <path-to-pth-file>
```

### Notes:
- Only in this case, our model expects exact path to checkpoint file to be loaded.
- `TODOs`: make this similar to other evaluation scripts - load ckpts in a similar fashion