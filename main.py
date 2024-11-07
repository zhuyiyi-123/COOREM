import numpy as np
import random
import torch
import wandb
import pandas as pd
import os
import matplotlib.pyplot as plt
from Taskdata import OfflineTask
from core.coorem.nets import DualThreeSurogateModel
from core.coorem.trainers import DualHeadSurogateTrainer
from core.utils import RiskSuppressionFactor, sample_langevin, neg_cons, setup_seed
from core.data import TaskDataset
from core.coorem.optimizesa import Optimizer
from constraint.CEC_problem import cec20_func
from constraint import gtopx_data
from exper import get_parser

def stand_scores(data):
    # normalization 
    mean = np.mean(data)
    std = np.std(data)
    stand_scores = (data - mean) / std
    return stand_scores

def build_data(args):
    # build datasets
    Task = OfflineTask(task=args.Task, benchmark=args.benchmark)
    data_set = pd.read_csv(os.path.join(args.csvname+ '.csv')).values
    ori_x = data_set[:, 1:Task.var_num + 1].astype(np.float64)
    ori_y = -data_set[:, Task.var_num + 1:Task.var_num + Task.obj_num + 1].reshape(len(data_set[:, Task.var_num + 1:Task.var_num + Task.obj_num + 1]), ).astype(np.float64)
    cons = data_set[:, Task.var_num + Task.obj_num + 1:Task.var_num + Task.obj_num + Task.con_num + 1].astype(np.float64)
    cons_mean = np.mean(cons).astype(np.float64)
    cons_std = np.std(cons).astype(np.float64)

    x = torch.tensor(stand_scores(ori_x).astype(np.double)).cuda()
    y = torch.tensor(stand_scores(ori_y.reshape(len(ori_y), 1)).astype(np.double)).cuda()
    cons = torch.tensor(stand_scores(cons).astype(np.double)).cuda()
    return ori_x, x, y , cons, cons_mean, cons_std, Task

def build_model(x, y, cons, cons_mean, cons_std, args):
    # build the three-head surrogate model
    dhs_model = DualThreeSurogateModel(np.prod(x.shape[1:]), 2048, int(np.prod(y.shape[1:])), cons).cuda()
    init_m = args.init_m * np.sqrt(np.prod(x.shape[1:]))
    trainer = DualHeadSurogateTrainer(dhs_model, cons=cons,
                                      dhs_model_prediction_opt=torch.optim.Adam, dhs_model_energy_opt=torch.optim.Adam,
                                      surrogate_lr=0.001, init_m=init_m,
                                      ldk=args.ldk, cons_mean=cons_mean, cons_std=cons_std)
    return dhs_model, trainer

def construct_dataloader(x, y, cons, cons_mean, cons_std):
    # create data loaders
    dataset = TaskDataset(x, y, cons)
    train_dataset_size = int(len(dataset) * (1 - 0.2))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                                               [train_dataset_size,
                                                                (len(dataset) - train_dataset_size)])
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    validate_dl = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True)

    # select the top k initial designs from the dataset
    new_x, new_y = neg_cons(x, y, cons, cons_mean, cons_std).feasible()
    indice = torch.topk(new_y[:, 0], 128)[1].unsqueeze(1)
    init_xt = new_x[indice].squeeze(1)
    init_yt = new_y[indice].squeeze(1)
    return train_dl, validate_dl, init_xt, init_yt

def train_model(trainer, dhs_model, train_dl, validate_dl, args):
    # train the surrogate model
    trainer.launch(train_dl, validate_dl, 1, True)

    if args.save_model:
        torch.save(dhs_model.state_dict(), os.path.join(args.save_model_name))
    if args.load_model:
        dhs_model.load_state_dict(torch.load(args.load_model_name))

def build_energy(dhs_model, init_xt, x, y, cons, cons_mean, cons_std, args):
    # get energy scalar
    energy_min = dhs_model(init_xt)[1].mean().detach().cpu().numpy()
    energy_max = dhs_model(sample_langevin(init_xt, dhs_model, stepsize=args.init_m, n_steps=64, noise=False))[1].mean().detach().cpu().numpy()
    uc_ood = RiskSuppressionFactor(energy_min, energy_max, init_m=args.init_m)
    energy_min = dhs_model(init_xt)[2].mean().detach().cpu().numpy()
    energy_max = dhs_model(neg_cons(x, y, cons, cons_mean, cons_std).cons_x())[2].mean().detach().cpu().numpy()
    uc_inf = RiskSuppressionFactor(energy_min, energy_max, init_m=args.init_m)
    return uc_ood, uc_inf

def optimizer(args, Task, trainer, init_xt, init_yt, dhs_model, uc_ood, uc_inf, ori_x):
    # the optimization process
    scores = []
    optimizer = Optimizer({"energy_opt": True, "opt_steps": 1000}, Task,
                        trainer, init_xt, init_yt, dhs_model=dhs_model)
    score = optimizer.optimize(args, uc_ood, uc_inf, mean=torch.tensor(np.mean(ori_x)).cuda(),
                        std=torch.tensor(np.std(ori_x)).cuda())
    scores.append(score)

def run(args):
    ori_x, x, y, cons, cons_mean, cons_std, Task = build_data(args)
    dhs_model, trainer = build_model(x, y, cons, cons_mean, cons_std, args)
    train_dl, validate_dl, init_xt, init_yt = construct_dataloader(x, y, cons, cons_mean, cons_std)
    train_model(trainer, dhs_model, train_dl, validate_dl, args)
    uc_ood, uc_inf = build_energy(dhs_model, init_xt, x, y, cons, cons_mean, cons_std, args)
    optimizer(args, Task, trainer, init_xt, init_yt, dhs_model, uc_ood, uc_inf, ori_x)

if __name__ == '__main__':
    args = get_parser()
    setup_seed(args.seed)
    run_name="COOREM-{}-False-{}".format(args.ldk, "CSM")
    wandb.init(project="COOREM", name=run_name, config=vars(args))
    run(args)                                                                                                  
