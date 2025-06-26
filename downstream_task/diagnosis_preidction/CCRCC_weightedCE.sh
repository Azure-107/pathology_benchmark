root='/mnt/pool/ovariancancer/mSTAR_results/CCRCC/pt_files' #change here
feature='uni'
studies='CCRCC_5folds_univ1_HKUST_weightedCE '
models='ABMIL'
# ckpt for pretrained aggregator
# aggregator='aggregator'
# export WANDB_MODE=dryrun
# cd ..

for study in $studies
do
    for model in $models
    do
        CUDA_VISIBLE_DEVICES=0 python main.py --model $model \
                                            --root $root \
                                            --csv_file ./dataset_csv/CCRCC_5folds.csv \
                                            --feature $feature \
                                            --study $study \
                                            --num_epoch 50 \
                                            --batch_size 1 \
                                            --loss weighted_ce 
    done
done

# nohup bash CCRCC_weightedCE.sh > "/home/jma/Documents/Beatrice/logs/ABMIL_mSTAR_univ1_CCRCC_weightedCE_$(date +%Y%m%d_%H%M%S).txt" 2>&1 &