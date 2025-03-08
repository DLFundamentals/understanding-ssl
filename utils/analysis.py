import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from collections import defaultdict

@torch.no_grad()
def cal_cdnv(model, settings, loader):
    model.eval()

    initialized_results = False
    num_embs = 0
    N = []
    mean = []
    mean_s = []
    cdnvs = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            _, data, target = batch
            # if data.shape[0] != settings.batch_size:
            #     continue

            data, target = data.to(settings.device), target.to(settings.device)
            h, g_h = model(data)
            embeddings = [h, g_h]
            # embeddings = embeddings.unsqueeze(0)
            num_embs = len(embeddings)

            if initialized_results == False: # if we did not init means to zero
                N = [settings.num_output_classes*[0] for _ in range(num_embs)]
                mean = [settings.num_output_classes*[0] for _ in range(num_embs)]
                mean_s = [settings.num_output_classes*[0] for _ in range(num_embs)]
                initialized_results = True

            for i in range(num_embs): # which top layer to investigate
                h = embeddings[i]

                for c in range(settings.num_output_classes):
                    idxs = (target == c).nonzero(as_tuple=True)[0]
                    if len(idxs) == 0:  # If no class-c in this batch
                        continue

                    h_c = h[idxs, :]
                    mean[i][c] += torch.sum(h_c, dim=0)
                    N[i][c] += h_c.shape[0]
                    mean_s[i][c] += torch.sum(torch.square(h_c))
    
    for i in range(num_embs):
        for c in range(settings.num_output_classes):
            mean[i][c] /= N[i][c]
            mean_s[i][c] /= N[i][c]

        avg_cdnv = 0
        total_num_pairs = settings.num_output_classes * (settings.num_output_classes - 1) / 2
        for class1 in range(settings.num_output_classes):
            for class2 in range(class1 + 1, settings.num_output_classes):
                variance1 = abs(mean_s[i][class1].item() - torch.sum(torch.square(mean[i][class1])).item())
                variance2 = abs(mean_s[i][class2].item() - torch.sum(torch.square(mean[i][class2])).item())
                variance_avg = (variance1 + variance2) / 2
                dist = torch.norm((mean[i][class1]) - (mean[i][class2]))**2
                dist = dist.item()
                cdnv = variance_avg / dist
                avg_cdnv += cdnv / total_num_pairs

        cdnvs += [avg_cdnv]
    
    return cdnvs


def embedding_performance(model, settings, train_loader, test_loader=None):

    model.eval()
    batch = next(iter(train_loader))
    _, data, target = batch
    data, target = data.to(settings.device), target.to(settings.device)
    h, g_h = model(data)
    embeddings = [h]
    num_embs = len(embeddings)
    linear_projs = []
    loss_function = nn.CrossEntropyLoss()
    params = list()

    for i in range(num_embs):

        emb = embeddings[i]
        emb = emb.view(emb.size()[0], -1)
        emb_dim = emb.shape[1]

        # init the linear classifiers
        linear_proj = nn.Linear(emb_dim, settings.num_output_classes, bias=False).to(settings.device)
        linear_projs += [linear_proj]
        params += list(linear_proj.parameters())

    # init the optimizer
    # optimizer = optim.SGD(params, lr=settings.top_lr, momentum=settings.momentum, weight_decay=settings.weight_decay)
    optimizer = optim.Adam(params, lr=settings.top_lr)

    # train phase
    wandb_defined = False
    for i in range(settings.epochs):
        linear_projs = [proj.train() for proj in linear_projs]
        print(f"Epoch {i+1}/{settings.epochs}")
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            _, data, target = batch

            loss = 0.0
            data, target = data.to(settings.device), target.to(settings.device)
            h, g_h = model(data)
            embeddings = [h]
            
            for j in range(num_embs):
                embedding = embeddings[j]
                embedding = embedding.view(embedding.size(0), -1)
                outputs = linear_projs[j](embedding.view(embedding.size(0), -1))
                loss += loss_function(outputs, target)

            loss.backward()
            optimizer.step()

        if i % settings.save_every == 0 and settings.track_performance:
            train_accuracy_rates, train_losses = test(settings, num_embs, model, linear_projs, train_loader)
            test_accuracy_rates, test_losses = test(settings, num_embs, model, linear_projs, test_loader)
            print(f"Train accuracy: {train_accuracy_rates}")
            log_metrics(train_accuracy_rates, train_losses, i, wandb_defined)
            log_metrics_test(test_accuracy_rates, test_losses, i, wandb_defined)
            wandb_defined = True

    train_accuracy_rates, train_losses = test(settings, num_embs, model, linear_projs, train_loader)
    test_accuracy_rates, test_losses = test(settings, num_embs, model, linear_projs, test_loader)

    return train_accuracy_rates, train_losses

def log_metrics(acc_rates, losses, epoch, wandb_defined=False):
    if not wandb_defined:
        wandb.define_metric("epoch")
        num_embs = len(acc_rates)
        for i in range(num_embs):
            wandb.define_metric(f"train_accuracy_{i}", step_metric="epoch")
            wandb.define_metric(f"lin_prob_loss_{i}", step_metric="epoch")

    log_data = defaultdict()

    log_data["epoch"] = epoch
    num_embs = len(acc_rates)
    for i in range(num_embs):
        log_data[f"train_accuracy_{i}"] = acc_rates[i]
        log_data[f"lin_prob_loss_{i}"] = losses[i]


    wandb.log(log_data)
    
def log_metrics_test(acc_rates, losses, epoch, wandb_defined=False):
    if not wandb_defined:
        wandb.define_metric("epoch")
        num_embs = len(acc_rates)
        for i in range(num_embs):
            wandb.define_metric(f"test_accuracy_{i}", step_metric="epoch")
            wandb.define_metric(f"test_lin_prob_loss_{i}", step_metric="epoch")

    log_data = defaultdict()

    log_data["epoch"] = epoch
    num_embs = len(acc_rates)
    for i in range(num_embs):
        log_data[f"test_accuracy_{i}"] = acc_rates[i]
        log_data[f"test_lin_prob_loss_{i}"] = losses[i]


    wandb.log(log_data)


@torch.no_grad()
def test(settings, num_embs, model, linear_projs, test_loader):

    loss_function = nn.CrossEntropyLoss()
    
    # test phase
    test_losses = num_embs * [0.0]
    corrects = num_embs * [0.0]

    for batch in test_loader:
        _, data, labels = batch

        if settings.device == 'cuda':
            data = data.cuda()
            labels = labels.cuda()

        h, g_h = model(data)
        embeddings = [h]
        
        for i in range(num_embs):
            embedding = embeddings[i]
            embedding = embedding.view(embedding.size(0), -1)
            outputs = linear_projs[i](embedding)
            test_losses[i] += loss_function(outputs, labels).item()
            _, preds = outputs.max(1)
            corrects[i] += preds.eq(labels).sum().item()

    dataset_size = len(test_loader.dataset)
    accuracy_rates = [corrects[i] / dataset_size for i in range(num_embs)]
    losses = [test_losses[i] / dataset_size for i in range(num_embs)]

    return accuracy_rates, losses

@torch.no_grad()
def embedding_performance_nearest_mean_classifier(model, settings,
                                                  train_loader, test_loader=None):

    model.eval()

    batch = next(iter(train_loader))
    _, data, target = batch
    data, target = data.to(settings.device), target.to(settings.device)
    h, g_h = model(data)
    embeddings = [h, g_h]
    num_embs = len(embeddings)

    means = []
    N = [settings.num_output_classes * [0] for i in range(num_embs)]

    for i in range(num_embs):

        emb = embeddings[i]
        emb = emb.view(emb.size()[0], -1)
        emb_dim = emb.shape[1]
        means += [torch.zeros(settings.num_output_classes, emb_dim).to(settings.device)]

    with torch.no_grad():
        # train phase
        for batch_idx, batch in enumerate(train_loader, start=1):
            _, data, target = batch

            data, target = data.to(settings.device), target.to(settings.device)
            h, g_h = model(data)
            embeddings = [h, g_h]

            for i in range(num_embs):
                embedding = embeddings[i]
                embedding = embedding.view(embedding.size(0), -1)

                for c in range(settings.num_output_classes):
                    idxs = (target == c).nonzero(as_tuple=True)[0]
                    if len(idxs) == 0:  # If no class-c in this batch
                        continue

                    h_c = embedding[idxs, :]
                    means[i][c] += torch.sum(h_c, dim=0)
                    N[i][c] += h_c.shape[0]

    for i in range(num_embs):
        for c in range(settings.num_output_classes):
            means[i][c] /= N[i][c]

    train_accuracy_rates = test_nearest_mean(settings, num_embs, model, means, train_loader)
    if test_loader is not None:
        test_accuracy_rates = test_nearest_mean(settings, num_embs, model, means, test_loader)
        return train_accuracy_rates, test_accuracy_rates

    return train_accuracy_rates


@torch.no_grad()
def test_nearest_mean(settings, num_embs, model, means, test_loader):

    corrects = num_embs * [0.0]
    # testing phase
    for batch_idx, batch in enumerate(test_loader, start=1):
        _, data, target = batch
        # if data.shape[0] != settings.batch_size:
        #     continue

        data, target = data.to(settings.device), target.to(settings.device)
        h, g_h = model(data)
        embeddings = [h, g_h]
        
        for i in range(num_embs):
            embedding = embeddings[i]
            embedding = embedding.view(embedding.size(0), -1)
            outputs = torch.cdist(embedding.unsqueeze(0), means[i].unsqueeze(0)).squeeze(0)
            _, preds = outputs.min(1)
            corrects[i] += preds.eq(target).sum().item()

    dataset_size = len(test_loader.dataset)
    accuracy_rates = [corrects[i] / dataset_size for i in range(num_embs)]

    return accuracy_rates