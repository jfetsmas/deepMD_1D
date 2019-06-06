for Nnear in {1..7..1}
do
cat > "script_train_v7_Nnear_${Nnear}_more.sh" << EOF
#!/bin/bash
# Job name:
#SBATCH --job-name=training
#
# Account:
#SBATCH --account=ac_esmath
#
# Partition:
#SBATCH --partition=savio2_gpu
#
# Quality of Service:
#SBATCH --qos=savio_normal
#
#SBATCH --nodes=1
#
#SBATCH --ntasks=1
#
#SBATCH --cpus-per-task=2
#
#SBATCH --gres=gpu:1
#
# Wall clock limit:
#SBATCH --time=48:00:00
#
## Command(s) to run:

module load ml/tensorflow/1.7.0-py35
python MD_rho_1D_v7_fulloscillation.py --lr 1e-3 --batch-size 80 --epoch 50 --Nnear ${Nnear} --n-train 100 --input-prefix 'KS_MD_scf_8_sigma_6.0'
EOF
       sbatch script_train_v7_Nnear_${Nnear}_more.sh
done
