
# model="resnet50"
# models='plip'
# models='uni'
# models='conch'
# models='mSTAR'
models='uni'
declare -A gpus

gpus['resnet50']=0
gpus['conch']=0
gpus['uni']=0
gpus['mSTAR']=0

CSV_FILE_NAME="./dataset_csv/CCRCC.csv"

DIR_TO_COORDS="/mnt/pool/ovariancancer/CCRCC_results/univ1/20x_256px_0px_overlap"
DATA_DIRECTORY="/mnt/pool/ovariancancer/raw_data/CCRCC"

FEATURES_DIRECTORY="/mnt/pool/ovariancancer/mSTAR_results/CCRCC"

ext=".svs"
for model in $models
do
        echo $model", GPU is:"${gpus[$model]}
        export CUDA_VISIBLE_DEVICES=${gpus[$model]}

        python extract_feature.py \
                --data_h5_dir $DIR_TO_COORDS \
                --data_slide_dir $DATA_DIRECTORY \
                --csv_path $CSV_FILE_NAME \
                --feat_dir $FEATURES_DIRECTORY \
                --batch_size 256 \
                --model $model \
                --slide_ext $ext
done

# nohup bash CCRCC.sh > "/home/jma/Documents/Beatrice/logs/mSTAR_extractfeature_CCRCC_$(date +%Y%m%d_%H%M%S).txt" 2>&1 &