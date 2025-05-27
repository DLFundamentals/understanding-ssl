# Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning

In our work, we make progress towards addressing the following question:

> How does self-supervsied CL learn representations similar to supervised learning,
> despite lacking explicit supervision?

Instructions on how to pretrain, linear probe, and evaluate can be found in `./docs/`

In the following section, we have defined the code directory structure for easily understanding our work and reproducing our experiments:

```text
understanding-ssl/
| —— configs/                            # Config files for different algorithms and datasets
|    | —— moco.yaml                          
|    | —— datasets.yaml
|    | —— simclr.yaml
| —— data/
|    | —— cifar10/
| —— docs/
| —— | —— / evaluation.md
| —— | —— / linear_probing.md
| —— | —— / pretraining.md
| —— experiments/                        # Experiments directory 
|    | —— moco/
|    |    | —— logs/
|    |    | —— checkpoints/
|    |    | —— visualizations/
|    | —— simclr/
|    |    | —— logs/
|    |    | —— checkpoints/
|    |    | —— visualizations/
| —— models/                           
|    | —— base_encoder.py               # Backbones (ResNet50, ViT, etc.)
|    | —— moco.py                       # moco-specifc model
|    | —— projector.py                  # Projection head
|    | —— simclr.py                     # SimCLR-specific model
| —— notebooks/                         # Jupyter notebooks for random experiments
|    | —— simclr.ipynb
|    | —— moco.ipynb
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
