# ## best hyper-parameter for german dataset
# echo '============German============='
# CUDA_VISIBLE_DEVICES=1 python train.py --dropout 0.5 --hidden 16 --lr 1e-2 --epochs 1000 --model adagcn --dataset german --seed_num 5 --alpha 0.1 --beta 1.0

# ## best hyper-parameter for bail dataset
echo '============Bail============='
CUDA_VISIBLE_DEVICES=1 python train.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model adagcn --dataset bail --seed_num 5 --alpha 0.001 --beta 0.2

# ## best hyper-parameter for credit dataset
# echo '============Credit============='
# CUDA_VISIBLE_DEVICES=1 python train.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model adagcn --dataset credit --seed_num 5 --alpha 0.5 --beta 0.1

# ## best hyper-parameter for pokec_z dataset
# echo '============Pokec_z============='
# CUDA_VISIBLE_DEVICES=1 python train.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model adagcn --dataset pokec_z --seed_num 5 --alpha 0.001 --beta 0.05

## best hyper-parameter for pokec_n dataset
# echo '============Pokec_n============='
# CUDA_VISIBLE_DEVICES=1 python train.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model adagcn --dataset pokec_n --seed_num 5 --alpha 0.05 --beta 0.001
