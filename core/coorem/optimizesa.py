import numpy as np
import torch
import wandb
import torch.autograd as autograd


class Optimizer(object):
    def __init__(self, config, task,
                 trainer, init_xt, init_yt,
                 pre_model=None, dhs_model=None, is_normalized_x=True):
        
        self.config = config
        self.task = task
        self.trainer = trainer
        self.init_xt = init_xt
        self.init_yt= init_yt
        self.predictive_model = pre_model
        self.dhs_model = dhs_model
        self.is_normalized_x = is_normalized_x

    def optimize(self, args, uc_ood, uc_inf, mean, std):
        self.uc_ood = uc_ood
        self.uc_inf = uc_inf
        self.mean = mean
        self.std = std
        self.dhs_model.eval()
        
        xt = self.init_xt
        solution = xt
        if self.is_normalized_x:
            solution = solution * self.std + self.mean
        score, cons = self.task.predict(solution.detach().cpu().numpy())
        init_score = - score
        ood = []
        inf = []
        ucood = []
        ucinf = []
        for step in range(1, 1 + self.config['opt_steps']):
            energy_ood = self.dhs_model(xt)[1]
            energy_inf = self.dhs_model(xt)[2]
            uc_e_ood = self.uc_ood.normalize(energy_ood.detach().cpu().numpy())
            uc_e_inf = self.uc_inf.normalize(energy_inf.detach().cpu().numpy())
            ucood.append(max(uc_e_ood))
            ucinf.append(max(uc_e_inf))
            uc_e = (uc_e_ood - max(uc_e_ood) * args.opt_ood) * (max(uc_e_inf) * args.opt_inf - uc_e_inf)
            xt = self.optimize_step(xt, 1, uc_e, self.config['energy_opt'])
            # evaluate the solutions found by the model
            if self.is_normalized_x:
                xtt = xt * self.std + self.mean
            score, cons = self.task.predict(xtt.detach().cpu().numpy())
            flag = 1
            for qq in range(len(cons[0])):
                if cons[0][qq] < 0:
                    flag = 0
            print(f"step:{step}xt{xtt[0].tolist()}uc_ood{uc_e_ood[0]}uc_inf{uc_e_inf[0]}score{-score[0]}cons{flag}")
            print("uc:", uc_e[0])
            worst, best = feasible(score, cons)
            wandb.log({"offline_init": init_score,
                       "score": -torch.tensor(score),
                       "opt/best": best,
                       "opt/worst": worst,
                       "opt/energy_ood": energy_ood,
                       "opt/energy_inf": energy_inf,
                       "opt/risk suppression factor": torch.tensor(uc_e)})

        worst, best = feasible(score, cons)
        wandb.log({"final/best": best, "final/worst":worst})
        return best

    def optimize_step(self, xt, steps, uc_e, energy_opt=False):
        self.dhs_model.eval()
        
        for step in range(steps):
            if energy_opt:
                uc_e = torch.tensor(uc_e).cuda()
                if len(xt.shape) > 2:
                    uc_e = uc_e.expand(xt.shape[0], xt.shape[1]*xt.shape[2])
                    uc_e = torch.reshape(uc_e, xt.shape)
                xt.requires_grad = True

                loss = self.dhs_model(xt)[0]
                grad = autograd.grad(loss.sum(), xt)[0]
                xt = xt + uc_e * grad
            else:
                xt.requires_grad = True
                loss = self.dhs_model(xt)[0]
                grad = autograd.grad(loss.sum(), xt)[0]
                xt = xt + self.init_m * grad
            print("grad:", (uc_e * grad)[0])
        return xt.detach()
    
    def optimize_tr(self, xt, steps, grad_scale=1):
        self.dhs_model.eval()
        xt.requires_grad = True

        for step in range(steps):
            loss_p = self.dhs_model(xt)[0]
            grad_p = autograd.grad(loss_p.sum(), xt, retain_graph=True)[0]
            xt_p = xt + grad_scale * grad_p

            loss = loss_p - 0.9 * self.dhs_model(xt_p)[0]
            grad = autograd.grad(loss.sum(), xt)[0]
            xt = xt + grad_scale * grad

        return xt.detach()

def feasible(score, cons):
    w = 0
    good_y = []
    for a in range(len(score)):
        for b in range(len(cons[0])):
            if cons[a][b] < 0:
                w = w + 1
        if w == 0:
            good_y.append(score[a])
    print("best:", max(good_y), min(good_y))
    return max(good_y), min(good_y)