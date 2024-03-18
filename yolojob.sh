#!/bin/bash

##Job Script for FYP

#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --nodelist=SCSEGPU-TC1-04
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=yoloOG
#SBATCH --output=./output/output_%x_%j.out
#SBATCH --error=./error/error_%x_%j.err

module load cuda/12.1
module load anaconda
source activate torch
python3 -m torch.distributed.launch --use-env --nproc_per_node=1 main.py --train
# python3 ./tools/test.py -md retina_fpn_baseline -r 35

# python3 ./tools/eval_json.py -f /home/FYP/ryu007/CrowdDet/model/rcnn_fpn_baseline/outputs/eval_dump/dump-20.json
# python3 ./tools/inference.py -md rcnn_emd_simple -r 20 -i /home/FYP/ryu007/CrowdDet/data/CrowdHuman/images/273271,1b9eb00089049cd6.jpg
# python3 ./tools/visulize_json.py -f /home/FYP/ryu007/CrowdDet/model/rcnn_emd_simple/outputs/eval_dump/dump-20.json -n 3