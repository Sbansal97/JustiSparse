#!/bin/sh
#SBATCH --job-name=trigger
#SBATCH --mail-user=adityasv@cs.cmu.edu
#SBATCH --mail-type=ALL
#SBATCH -t 0 # unlimited time for executing

jid1=""
jid2=""
jid3=""
jid4=""
jid5=""
jid6=""

DATASET=bias-bios
AXIS=gender
LABEL_COLUMN=p
for PEFT in sft pfeiffer prefix_tuning_flat; do
    DEBIAS_METHOD=cda
    echo $DATASET $AXIS $PEFT
    PATCH_PATH=models/$PEFT/$DEBIAS_METHOD/$AXIS/$DATASET
    if [[ $PEFT == "sft" ]];then
        PATCH_PATH=models/$PEFT/$DEBIAS_METHOD/$AXIS/$DATASET/sft
        echo $PATCH_PATH
    else
        PATCH_PATH=models/$PEFT/$DEBIAS_METHOD/$AXIS/$DATASET
    fi
    if [ ! -d $PATCH_PATH ]; then
        echo "not found locally, looking into alternate path"
        PATCH_PATH=/projects/tir6/general/srijanb/Spr23/JustiSparse/$PATCH_PATH
    fi
    if [[ $jid1 == "" && $jid2 == "" && $jid3 == "" ]]; then
        jid1=$(sbatch --exclude=tir-1-28 train_extrinsic-debias_after.sh $PEFT $DATASET $PATCH_PATH  $LABEL_COLUMN $DEBIAS_METHOD | cut -d ' ' -f4 )
        jid2=$(sbatch --exclude=tir-1-28 train_extrinsic-debias_before.sh $PEFT $DATASET $PATCH_PATH  $LABEL_COLUMN $DEBIAS_METHOD | cut -d ' ' -f4 )
        jid3=$(sbatch --exclude=tir-1-28 train_extrinsic-no_debias.sh $PEFT $DATASET $PATCH_PATH  $LABEL_COLUMN $DEBIAS_METHOD | cut -d ' ' -f4 )
    else
        jid1=$(sbatch --exclude=tir-1-28 --dependency=afterany:${jid1} train_extrinsic-debias_after.sh $PEFT $DATASET $PATCH_PATH  $LABEL_COLUMN $DEBIAS_METHOD | cut -d ' ' -f4 )
        jid2=$(sbatch --exclude=tir-1-28 --dependency=afterany:${jid2} train_extrinsic-debias_before.sh $PEFT $DATASET $PATCH_PATH  $LABEL_COLUMN $DEBIAS_METHOD | cut -d ' ' -f4 )
        jid3=$(sbatch --exclude=tir-1-28 --dependency=afterany:${jid3} train_extrinsic-no_debias.sh $PEFT $DATASET $PATCH_PATH  $LABEL_COLUMN $DEBIAS_METHOD | cut -d ' ' -f4 )
    fi
    
    # DEBIAS_METHOD=adv
    # echo $DATASET $AXIS $PEFT
    # PATCH_PATH=models/$PEFT/$DEBIAS_METHOD/$AXIS/$DATASET
    # if [ ! -d $PATCH_PATH ]; then
    #     echo "not found locally, looking into alternate path"
    #     PATCH_PATH=/projects/tir6/general/srijanb/Spr23/JustiSparse/models/$PEFT/$DEBIAS_METHOD/$AXIS/$DATASET
    # fi
    # if [[ $jid4 == "" && $jid5 == "" && $jid6 == "" ]]; then
    #     jid4=$(sbatch --exclude=tir-1-28 train_extrinsic-debias_after.sh $PEFT $DATASET $PATCH_PATH  $LABEL_COLUMN $DEBIAS_METHOD | cut -d ' ' -f4 )
    #     jid5=$(sbatch --exclude=tir-1-28 train_extrinsic-debias_before.sh $PEFT $DATASET $PATCH_PATH  $LABEL_COLUMN $DEBIAS_METHOD | cut -d ' ' -f4 )
    #     jid6=$(sbatch --exclude=tir-1-28 train_extrinsic-no_debias.sh $PEFT $DATASET $PATCH_PATH  $LABEL_COLUMN $DEBIAS_METHOD | cut -d ' ' -f4 )
    # else
    #     jid4=$(sbatch --exclude=tir-1-28 --dependency=afterany:${jid4} train_extrinsic-debias_after.sh $PEFT $DATASET $PATCH_PATH  $LABEL_COLUMN $DEBIAS_METHOD | cut -d ' ' -f4 )
    #     jid5=$(sbatch --exclude=tir-1-28 --dependency=afterany:${jid5} train_extrinsic-debias_before.sh $PEFT $DATASET $PATCH_PATH  $LABEL_COLUMN $DEBIAS_METHOD | cut -d ' ' -f4 )
    #     jid6=$(sbatch --exclude=tir-1-28 --dependency=afterany:${jid6} train_extrinsic-no_debias.sh $PEFT $DATASET $PATCH_PATH  $LABEL_COLUMN $DEBIAS_METHOD | cut -d ' ' -f4 )
    # fi

done

DATASET=gab
AXIS=group
LABEL_COLUMN=label
for PEFT in sft pfeiffer prefix_tuning_flat; do
    DEBIAS_METHOD=cda
    echo $DATASET $AXIS $PEFT
    PATCH_PATH=models/$PEFT/$DEBIAS_METHOD/$AXIS/$DATASET
    if [[ $PEFT == "sft" ]];then
        PATCH_PATH=models/$PEFT/$DEBIAS_METHOD/$AXIS/$DATASET/sft
        echo $PATCH_PATH
    else
        PATCH_PATH=models/$PEFT/$DEBIAS_METHOD/$AXIS/$DATASET
    fi
    if [ ! -d $PATCH_PATH ]; then
        echo "not found locally, looking into alternate path"
        PATCH_PATH=/projects/tir6/general/srijanb/Spr23/JustiSparse/$PATCH_PATH
    fi
    if [[ $jid4 == "" && $jid5 == "" && $jid6 == "" ]]; then
        jid4=$(sbatch --exclude=tir-1-28 train_extrinsic-debias_after.sh $PEFT $DATASET $PATCH_PATH  $LABEL_COLUMN $DEBIAS_METHOD | cut -d ' ' -f4 )
        jid5=$(sbatch --exclude=tir-1-28 train_extrinsic-debias_before.sh $PEFT $DATASET $PATCH_PATH  $LABEL_COLUMN $DEBIAS_METHOD | cut -d ' ' -f4 )
        jid6=$(sbatch --exclude=tir-1-28 train_extrinsic-no_debias.sh $PEFT $DATASET $PATCH_PATH  $LABEL_COLUMN $DEBIAS_METHOD | cut -d ' ' -f4 )
    else
        jid4=$(sbatch --exclude=tir-1-28 --dependency=afterany:${jid4} train_extrinsic-debias_after.sh $PEFT $DATASET $PATCH_PATH  $LABEL_COLUMN $DEBIAS_METHOD | cut -d ' ' -f4 )
        jid5=$(sbatch --exclude=tir-1-28 --dependency=afterany:${jid5} train_extrinsic-debias_before.sh $PEFT $DATASET $PATCH_PATH  $LABEL_COLUMN $DEBIAS_METHOD | cut -d ' ' -f4 )
        jid6=$(sbatch --exclude=tir-1-28 --dependency=afterany:${jid6} train_extrinsic-no_debias.sh $PEFT $DATASET $PATCH_PATH  $LABEL_COLUMN $DEBIAS_METHOD | cut -d ' ' -f4 )
    fi
    
    # DEBIAS_METHOD=adv
    # echo $DATASET $AXIS $PEFT
    # PATCH_PATH=models/$PEFT/$DEBIAS_METHOD/$AXIS/$DATASET
    # if [ ! -d $PATCH_PATH ]; then
    #     echo "not found locally, looking into alternate path"
    #     PATCH_PATH=/projects/tir6/general/srijanb/Spr23/JustiSparse/models/$PEFT/$DEBIAS_METHOD/$AXIS/$DATASET
    # fi
    # if [[ $jid4 == "" && $jid5 == "" && $jid6 == "" ]]; then
    #     jid4=$(sbatch --exclude=tir-1-28 train_extrinsic-debias_after.sh $PEFT $DATASET $PATCH_PATH  $LABEL_COLUMN $DEBIAS_METHOD | cut -d ' ' -f4 )
    #     jid5=$(sbatch --exclude=tir-1-28 train_extrinsic-debias_before.sh $PEFT $DATASET $PATCH_PATH  $LABEL_COLUMN $DEBIAS_METHOD | cut -d ' ' -f4 )
    #     jid6=$(sbatch --exclude=tir-1-28 train_extrinsic-no_debias.sh $PEFT $DATASET $PATCH_PATH  $LABEL_COLUMN $DEBIAS_METHOD | cut -d ' ' -f4 )
    # else
    #     jid4=$(sbatch --exclude=tir-1-28 --dependency=afterany:${jid4} train_extrinsic-debias_after.sh $PEFT $DATASET $PATCH_PATH  $LABEL_COLUMN $DEBIAS_METHOD | cut -d ' ' -f4 )
    #     jid5=$(sbatch --exclude=tir-1-28 --dependency=afterany:${jid5} train_extrinsic-debias_before.sh $PEFT $DATASET $PATCH_PATH  $LABEL_COLUMN $DEBIAS_METHOD | cut -d ' ' -f4 )
    #     jid6=$(sbatch --exclude=tir-1-28 --dependency=afterany:${jid6} train_extrinsic-no_debias.sh $PEFT $DATASET $PATCH_PATH  $LABEL_COLUMN $DEBIAS_METHOD | cut -d ' ' -f4 )
    # fi

done