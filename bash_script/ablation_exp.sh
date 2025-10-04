

CUDA_VISIBLE_DEVICES=0 python ablation.py --loss_type 0 > logs/ablation_0.txt &
CUDA_VISIBLE_DEVICES=1 python ablation.py --loss_type 1 2 > logs/ablation_1.txt &
CUDA_VISIBLE_DEVICES=2 python ablation.py --loss_type 3 > logs/ablation_2.txt &
CUDA_VISIBLE_DEVICES=3 python ablation.py --loss_type 0 1 2 > logs/ablation_3.txt &
CUDA_VISIBLE_DEVICES=4 python ablation.py --loss_type 0 3 > logs/ablation_4.txt &
CUDA_VISIBLE_DEVICES=5 python ablation.py --loss_type 1 2 3 > logs/ablation_5.txt &

wait
