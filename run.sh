for i in $(seq 0 9); do
    for opt_ood in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
        for opt_inf in 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9; do
            python main.py --opt_ood=$opt_ood --opt_inf=$opt_inf --seed=$i
        done
    done
done

# for i in $(seq 0 9); do
#     # for opt_ood in 0.1 0.2 0.3 0.4 0.5; do
#     #     for opt_inf in 1.1 1.2 1.3 1.4 1.5; do
#         # for ldk in 2 16 32 64; do
#             python main.py --seed=$i
#         # done
#     # done
# done