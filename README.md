# Understanding SSL

In the following section, we have defined the code directory structure for easily understanding our work and reproducing our experiments:

```text
understanding-ssl/
| —— configs/                            # Config files for different algorithms and datasets
|    | —— byol.yaml                          
|    | —— datasets.yaml
|    | —— simclr.yaml
| —— data/
|    | —— cifar10/
| —— experiments/                        # Experiments directory 
|    | —— byol/
|    |    | —— logs/
|    |    | —— checkpoints/
|    |    | —— visualizations/
|    | —— simclr/
|    |    | —— logs/
|    |    | —— checkpoints/
|    |    | —— visualizations/
| —— models/                           
|    | —— base_encoder.py               # Backbones (ResNet50, ViT, etc.)
|    | —— byol.py                       # BYOL-specifc model
|    | —— projector.py                  # Projection head
|    | —— simclr.py                     # SimCLR-specific model
| —— notebooks/                         # Jupyter notebooks for random experiments
|    | —— simclr.ipynb
|    | —— byol.ipynb
| —— scripts/                        
|    | —— evaluate.py
|    | —— train.py
|    | —— visualize.py
| —— utils/                             # Utility functions
|    | —— augmentations.py
|    | —— dataset_loader.py
|    | —— losses.py
|    | —— metrics.py
| —— README.md
| —— requirements.txt
```
