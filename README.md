# COOREM

The implementation for the paper "Constrained Offline Black-Box Optimization via Risk Evaluation and Management". 

## Setup

Create the running environment with conda `4.10.3` with Python `3.9.0`:

```
conda create -n coorem python==3.9
conda activate coorem
```

Install the requirements for running COOREM:

```
pip install -r requirements.txt
```

## Run experiments

All the tasks of COOREM can be find in `gtopx.py` and `CEC_problem.py`. The following command line are available to run experiments:

```
./run.sh
```

you can define different parameters in `exper.py` and modify different hyperparameters and use `run.sh` to conduct experiments.

```
for i in $(seq 0 9); do
    for opt_ood in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
        for opt_inf in 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9; do
        	for ldk in 2 16 32 64; do
            	python main.py --opt_ood=$opt_ood --opt_inf=$opt_inf --seed=$i --ldk=$ldk
            done
        done
    done
done
```

#### Alation study of key componets

COOREM w/o CSM:

```
trainer.py Line 105:             loss = mse.to(dtype=torch.float32) + (self.alpha * score_neg.reshape(len(score_neg), )).mean().to(dtype=torch.float32) --> loss = mse.to(dtype=torch.float32)
```

COOREM w/o ASM:

```
trainer.py Line 105: loss = mse.to(dtype=torch.float32) + (self.alpha * score_neg.reshape(len(score_neg), )).mean().to(dtype=torch.float32) --> loss = mse.to(dtype=torch.float32) + (score_neg.reshape(len(score_neg), )).mean().to(dtype=torch.float32)
```

COOREM w/o ARM:

```
optimizesa.py Line 83: add uc_e=0.001/0.01
```

COOREM w/on FARM:

```
optimizesa.py Line 45:             uc_e = (uc_e_ood - max(uc_e_ood) * args.opt_ood) * (max(uc_e_inf) * args.opt_inf - uc_e_inf)-->uc_e = uc_e_ood * uc_e_inf
```

COOREM w/o ARMO:

```
optimizesa.py Line 45:             uc_e = (uc_e_ood - max(uc_e_ood) * args.opt_ood) * (max(uc_e_inf) * args.opt_inf - uc_e_inf)-->uc_e = uc_e_ood
```

COOREM w/o ARMC:

```
optimizesa.py Line 45:             uc_e = (uc_e_ood - max(uc_e_ood) * args.opt_ood) * (max(uc_e_inf) * args.opt_inf - uc_e_inf)-->uc_e = uc_e_inf
```

## The project structure

```
COOREM
├─ constraint
│  ├─ gtopx_data.py
│  ├─ gtopx.so
│  ├─ CEC_problem.py
├─ core
│  ├─ data.py
│  ├─ coorem
│  │  ├─ nets.py
│  │  ├─ optimizesa.py
│  │  ├─ trainers.py
│  │  └─ __init__.py
│  └─ utils.py
├─ README.md
├─ exper.py
├─ main.py
├─ run.sh
└─ requirements.txt
```