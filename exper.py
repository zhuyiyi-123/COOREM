import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Task", type=int, default=5)
    parser.add_argument("--csvname", type=str, default='dataset_c&g/Welded_new')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--Task_name", type=str, default='cec20_func')
    parser.add_argument("--save_model", type=bool, default=False)
    parser.add_argument("--save_model_name", type=str, default='model_para_surrogate_gtopx1.pkl')
    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--load_model_name", type=str, default='model_para_surrogate_gtopx1.pkl')
    parser.add_argument("--init_m", type=float, default=0.02)
    parser.add_argument("--iterepoches", type=int, default=125)
    parser.add_argument("--opt_ood", type=float, default=0.4)
    parser.add_argument("--opt_inf", type=float, default=1.1)
    parser.add_argument("--con_model", type=str, default='CSM')
    parser.add_argument("--ldk", type=int, default=128)
    return parser.parse_args()