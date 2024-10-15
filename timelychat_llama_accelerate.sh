#!/bin/sh
	
#SBATCH -J timelychat-llama
#SBATCH -o /home/minjinj/timely-chat/outputs/log/timelychat_llama%j.txt
#SBATCH -t 3-00:00:00

#SBATCH -p A100-80GB
#SBATCH -q hpgpu
#SBATCH --gres=gpu:3

#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4

echo "Swap cuda version from 11.8 to 12.1"
ml swap cuda/11.8 cuda/12.1
echo "Set complier to g++"
export CXX=g++

echo "Activate conda"
source $HOME/.bashrc
conda activate timelychat

cd $SLURM_SUBMIT_DIR

GPU_NUM=3

accelerate launch --num_processes $GPU_NUM --config_file /home/minjinj/timely-chat/accelerate_config.yaml  scripts/run_train_accelerate.py  --run-name accelerate  --pretrained-model meta-llama/Llama-3.1-8B-Instruct  --train-dataset-path /home/minjinj/timely-chat/resources/data/train_augmented.json  --val-dataset-path /home/minjinj/timely-chat/resources/data/valid_augmented.json --use-amp --log-output-dir ./outputs/log --weight-output-dir ./outputs --train-batch-size 16 --val-batch-size 16
conda deactivate
echo "Done!"