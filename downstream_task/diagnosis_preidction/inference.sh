root='/mnt/pool/ovariancancer/mSTAR_results/UHN_mSTAR/pt_files' #change here
study='UHN_5folds_mSTAR_HKUST_infTestonTrain'
ckpt_paths=(
    '/mnt/pool/ovariancancer/mSTAR_results/results_1/UHN_5folds_mSTAR_HKUST_noDropout/[ABMIL]/[mSTAR]-[2025-07-09]-[14-01-24]/fold_0/model_best_1.0000_16.pth.tar'
)
# ckpt for pretrained aggregator
# aggregator='aggregator'
# export WANDB_MODE=dryrun
# cd ..

for ckpt in "${ckpt_paths[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python main.py --model ABMIL \
                                        --root $root \
                                        --csv_file ./dataset_csv/UHN_5folds.csv \
                                        --feature mSTAR \
                                        --study $study \
                                        --evaluate \
                                        --resume $ckpt 
done

# nohup bash inference.sh > "/home/jma/Documents/Beatrice/logs/HKUST_mSTAR_UHN_infflipped$(date +%Y%m%d_%H%M%S).txt" 2>&1 &


