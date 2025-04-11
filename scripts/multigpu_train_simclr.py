"""
torchrun --nproc_per_node=#GPUs --standalone multigpu_train_simclr.py --config config/simclr_cifar10.yaml
#
# The `torchrun` command is a wrapper around `python -m torch.distributed.run` that simplifies the process of launching distributed training jobs.
# The `--standalone` flag is used to run the script as a standalone script, rather than as a module.
# The `--config` flag is used to specify the path to the configuration file.
# The `nproc_per_node` flag is used to specify the number of GPUs per node.

Arguments can be found here:
https://github.com/pytorch/pytorch/blob/bbe803cb35948df77b46a2d38372910c96693dcd/torch/distributed/run.py#L401
"""
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils import clip_grad_norm_
# from torchlars import LARS
from torch.amp import GradScaler, autocast
import wandb

# distributed training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# utils
from utils.augmentations import get_transforms
from utils.dataset_loader import get_dataset
from utils.losses import NTXentLoss, WeakNTXentLoss
from utils.optimizer import LARS

# model
from models.simclr import SimCLR

import argparse
import yaml
from tqdm import tqdm
from collections import namedtuple

# # set seed
# torch.manual_seed(123)
# torch.cuda.manual_seed(123)

# initialize distributed training
def ddp_setup():
    local_rank = int(os.environ.get("LOCAL_RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))
    # rank = int(os.environ.get("RANK"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", 
                            world_size=world_size, 
                            rank=local_rank)
    

def cleanup():
  dist.destroy_process_group()

class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_loader: torch.utils.data.DataLoader,
            criterion: torch.nn.Module,
            save_every: int,
            log_every: int,
            snapshot_dir: str,
            **kwargs,
    ) -> None:
        
        # set seed
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)

        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(f'cuda:{self.gpu_id}')
        self.train_loader = train_loader
        self.test_loader = kwargs.get("test_loader", None)
        self.criterion = criterion
        self.save_every = save_every
        self.log_every = log_every
        self.epochs_run = 0
        self.snapshot_dir = snapshot_dir
        if os.path.exists(self.snapshot_dir):
            self._load_snapshot(self.snapshot_dir)
            print(f"Loaded model from {self.snapshot_dir}")

        self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)

        # optimizer and scheduler
        effective_lr = kwargs.get("effective_lr", 0.1)
        total_epochs = kwargs.get("total_epochs", 100)
        self.optimizer, self.scheduler = self.configure_optimizers(self.model, effective_lr, total_epochs)
        if os.path.exists(self.snapshot_dir):
            self._load_optimizer_scheduler(self.snapshot_dir)
            print(f"Loaded optimizer and scheduler from {self.snapshot_dir}")

        self.track_performance = kwargs.get("track_performance", False)
        self.settings = kwargs.get("settings", None)
        self.perform_knn = kwargs.get("perform_knn", False)
        self.perform_cdnv = kwargs.get("perform_cdnv", False)
        self.perform_nccc = kwargs.get("perform_nccc", False)
        self.wandb_defined = False

        # mixed precision training
        self.scaler = GradScaler()

        # support weak NTXentloss as well
        self.supervision = kwargs.get("supervision", "SSL")
    
    def _load_snapshot(self, snapshot_dir: str) -> None:
        loc = f"cuda:{self.gpu_id}"
        # load the latest snapshot
        dir_list = os.listdir(snapshot_dir)
        if len(dir_list) == 0:
            print("No snapshots found!")
            return
        latest_snapshot = sorted(dir_list, reverse=True)[0]
        snapshot_path = os.path.join(snapshot_dir, latest_snapshot)

        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resume training from snapshot at epoch {self.epochs_run}")

    def _load_optimizer_scheduler(self, snapshot_dir: str) -> None:
        loc = f"cuda:{self.gpu_id}"
        # load the latest snapshot
        dir_list = os.listdir(snapshot_dir)
        if len(dir_list) == 0:
            print("No snapshots found!")
            return
        latest_snapshot = sorted(dir_list, reverse=True)[0]
        snapshot_path = os.path.join(snapshot_dir, latest_snapshot)

        snapshot = torch.load(snapshot_path, map_location=loc)
        self.optimizer.load_state_dict(snapshot["OPTIMIZER"])
        self.scheduler.load_state_dict(snapshot["SCHEDULER"])
    
    def _save_snapshot(self, snapshot_dir: str, epoch: int) -> None:
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "OPTIMIZER": self.optimizer.state_dict(),
            "SCHEDULER": self.scheduler.state_dict()
        }
        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_path = os.path.join(snapshot_dir, f"snapshot_{epoch}.pth")
        torch.save(snapshot, snapshot_path)
        print(f"Saved model to {snapshot_path} at epoch {epoch}")

    def _run_epoch(self, epoch: int) -> None:
        print(f"[GPU {self.gpu_id}] Training epoch {epoch}...")
        if isinstance(self.train_loader.sampler, DistributedSampler):
            self.train_loader.sampler.set_epoch(epoch)

        loss_per_epoch = 0.0
        for batch in tqdm(self.train_loader):
            self.optimizer.zero_grad()

            # enable mixed precision training
            with autocast(device_type='cuda'):
              loss = self.model.module.run_one_batch(batch,
                                                self.criterion,
                                                self.optimizer,
                                                self.gpu_id)
            
            # backward + update using gradscaler
            self.scaler.scale(loss).backward()
            # torch.cuda.synchronize()
            self.scaler.unscale_(self.optimizer)  # Unscale gradients before clipping
            #clip model gradients
            # clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)         
            self.scaler.update()
            torch.cuda.synchronize()

            loss_per_epoch += loss.item()
        
        # update learning rate
        self.scheduler.step()

        return loss_per_epoch / len(self.train_loader)
            

    def train(self, max_epochs: int) -> None:
        self.model.train()
        loss_per_epoch = 0.0
        # if self.epochs_run + 100 >= max_epochs:
        #     return
        for epoch in range(self.epochs_run, max_epochs):

            loss_per_epoch = self._run_epoch(epoch)

            # On GPU 0 do extra logging, snapshot saving, and evaluation
            if self.gpu_id == 0:
                # Save a snapshot
                if epoch % self.log_every == 0:
                    self._save_snapshot(self.snapshot_dir, epoch)
                    print(f"Saved model at epoch {epoch}")

                # Evaluate and log performance every self.save_every epochs
                if epoch % self.save_every == 0:
                    print(f"Loss per epoch: {loss_per_epoch}")
                    if self.track_performance:
                        # Switch to eval mode and run evaluation with no_grad
                        self.model.eval()
                        with torch.no_grad():
                            eval_outputs = self.model.module.custom_eval(
                                                            self.test_loader,
                                                            settings=self.settings,
                                                            perform_knn=self.perform_knn,
                                                            perform_cdnv=self.perform_cdnv,
                                                            perform_nccc=self.perform_nccc
                                                        )
                        # Log the performance metrics
                        self.log_metrics(eval_outputs, epoch, loss_per_epoch)
                        # Return the model to training mode
                        self.model.train()

                # Optionally, if using distributed training, you might call a barrier here:
                if dist.get_world_size() > 1:
                    dist.barrier()

        print("Training complete! 🎉")

    def configure_optimizers(self, ssl_model, effective_lr,
                             total_epochs, warmup_epochs = 10):

        # LARS optimizer
        # base_optimizer = optim.SGD(ssl_model.parameters(), lr=effective_lr, weight_decay=1e-6)
        # optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)
        optimizer = LARS(
            ssl_model.parameters(),
            lr=effective_lr,
            weight_decay=1e-6,
            exclude_from_weight_decay=["batch_normalization", "bias"]
        )

        # Learning rate warmup + cosine decay
        scheduler = lr_scheduler.LambdaLR(
            optimizer, 
            lambda epoch: min(1.0, (epoch + 1) / warmup_epochs) * 0.5 * (1 + torch.cos(torch.tensor(epoch / total_epochs * 3.1416)))
        )

        return optimizer, scheduler
    
    def log_metrics(self, eval_outputs, cur_epoch, cur_loss_per_epoch):
        # define epoch as x-axis
        if not self.wandb_defined:
            wandb.define_metric("epoch")
            wandb.define_metric("learning_rate", step_metric="epoch")
            wandb.define_metric("knn_accuracy", step_metric="epoch")
            wandb.define_metric("cdnv_0", step_metric="epoch")
            wandb.define_metric("log_cdnv_0", step_metric="epoch")
            wandb.define_metric("cdnv_1", step_metric="epoch")
            wandb.define_metric("log_cdnv_1", step_metric="epoch")
            wandb.define_metric("nccc_0", step_metric="epoch")
            wandb.define_metric("nccc_1", step_metric="epoch")

            # define loss per epoch
            wandb.define_metric("loss_per_epoch", step_metric="epoch")
            self.wandb_defined = True

        # collect all logs in one dictionary
        log_data = {"epoch": cur_epoch}
        log_data["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        log_data["loss_per_epoch"] = cur_loss_per_epoch

        if self.perform_knn:
            log_data["knn_accuracy"] = eval_outputs["knn_train_acc"]

        if self.perform_cdnv:
            if isinstance(eval_outputs["cdnv"], list):
                for i, cdnv in enumerate(eval_outputs["cdnv"]):
                    log_data[f'cdnv_{i}'] = cdnv
                    log_data[f'log_cdnv_{i}'] = torch.log10(torch.tensor(cdnv))
            else:
                log_data["cdnv_0"] = eval_outputs["cdnv"]
                log_data["log_cdnv_0"] = torch.log10(torch.tensor(eval_outputs["cdnv"]))

        if self.perform_nccc:
            if isinstance(eval_outputs["nccc"], list):
                for i, nccc in enumerate(eval_outputs["nccc"]):
                    log_data[f'nccc_{i}'] = nccc
            else:
                log_data["nccc"] = eval_outputs["nccc"]

        # log all metrics
        wandb.log(log_data)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='SimCLR Training')
    parser.add_argument('--config', '-c', required=True, help='path to yaml config file')
    args = parser.parse_args()

    # load config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # load config parameters
    experiment_name = config['experiment_name']
    method_type = config['method_type']
    supervision = config['supervision']

    dataset_name = config['dataset']['name']
    dataset_path = config['dataset']['path']
    num_output_classes = config['dataset']['num_output_classes']
    
    batch_size = config['training']['batch_size']
    epochs = config['training']['num_epochs']
    lr = config['training']['lr']
    augmentations_type = config['training']['augmentations_type'] # imagenet or cifar or other dataset name
    augment_both = config['training']['augment_both']
    save_every = config['training']['save_every']
    log_every = config['training']['log_every']
    # save_model = config['training']['save_model']
    track_performance = config['training']['track_performance']
    multi_gpu = config['training']['multi_gpu']
    world_size = config['training']['world_size']

    encoder_type = config['model']['encoder_type']
    width_multiplier = config['model']['width_multiplier']
    hidden_dim = config['model']['hidden_dim']
    projection_dim = config['model']['projection_dim']

    temperature = config['loss']['temperature']

    K = config['evaluation']['K'] if track_performance else None
    perform_knn = config['evaluation']['perform_knn']
    perform_cdnv = config['evaluation']['perform_cdnv']
    perform_nccc = config['evaluation']['perform_nccc']
    perform_tsne = config['evaluation']['perform_tsne']
    checkpoints_dir = config['evaluation']['checkpoints_dir']

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    Settings = namedtuple("Settings", ["batch_size", "device", "num_output_classes"])
    settings = Settings(batch_size=batch_size, 
                        device=device,
                        num_output_classes=num_output_classes)

    # initialize distributed training
    ddp_setup()
    print(f"Local rank: {os.environ.get('LOCAL_RANK')}, World size: {os.environ.get('WORLD_SIZE')}")

    if dist.get_rank() == 0 and track_performance:
        # wandb init
        wandb.init(
            project = "simclr",
            config = {
                "experiment_name": experiment_name,
                "dataset_name": dataset_name,
                "batch_size": batch_size,
                "lr": lr,
                "augment_both": augment_both,
                "world_size": world_size,
                "encoder_type": encoder_type,
                "width_multiplier": width_multiplier,
                "hidden_dim": hidden_dim,
                "projection_dim": projection_dim,
                "temperature": temperature,
                "K": K
            }
        )
    
    # load dataset
    world_size = int(os.environ.get('WORLD_SIZE'))
    print(f"Dataset: {dataset_name}")
    _, train_loader, _, test_loader = get_dataset(dataset_name=dataset_name, 
                                    dataset_path=dataset_path,
                                    augment_both_views=augment_both,
                                    batch_size=batch_size, multi_gpu=multi_gpu,
                                    world_size=world_size, supervision=supervision,
                                    test=True)


    # define model
    if encoder_type == 'resnet50':
        encoder = torchvision.models.resnet50(weights=None)
    else:
        raise NotImplementedError(f"{encoder_type} not implemented")
    
    if method_type == 'simclr':
        ssl_model = SimCLR(model=encoder,
                           dataset=dataset_name,
                           width_multiplier=width_multiplier,
                           hidden_dim=hidden_dim,
                           projection_dim=projection_dim,
                           track_performance=track_performance,
                            K=K)
    else:
        raise NotImplementedError(f"{method_type} not implemented")

    # convert all BatchNorm layers to SyncBatchNorm
    ssl_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ssl_model)
    dist.barrier() # wait for all processes to catch up

    # define loss & optimizer
    if supervision == 'SSL':
        print("Using Self-Supervised Contrastive Learning")
        criterion = NTXentLoss(temperature, device) 
    elif supervision == 'SCL':
        print("Using Weakly-Supervised Contrastive Learning")
        criterion = WeakNTXentLoss(temperature, device)
    else:
        raise NotImplementedError(f"{supervision} not implemented")

    # train model

    effective_lr = lr*world_size*(batch_size//256)
    # effective_lr = lr * 2.0 * (batch_size // 256)
    trainer = Trainer(
        model=ssl_model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        save_every=save_every,
        log_every=log_every,
        snapshot_dir=checkpoints_dir,
        track_performance=track_performance,
        effective_lr = effective_lr,
        settings = settings,
        perform_knn = perform_knn,
        perform_cdnv = perform_cdnv,
        perform_nccc = perform_nccc,
        total_epochs = epochs
    )
    
    trainer.train(epochs)

    dist.destroy_process_group()