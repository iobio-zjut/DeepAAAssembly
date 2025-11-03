#!/bin/bash

# -----------------------------------------------------------------------------------------------------------------------------
# input variables
target1=$1
target2=$2
target=$3

# -----------------------------------------------------------------------------------------------------------------------------
# generate the paired MSA
 if [ ! -f ${target}_paired.a3m ];then
 	python $SCRIPT_DIR/bin/gen_paired_msa.py ${target1}.a3m ${target2}.a3m ${target}_paired.a3m $target
 fi

 # generate the PSSM, DCA feats with the paired MSA
 if [ -f ${target}_paired.a3m ];then

 #     # generate HHM file for PSSM
     if [ ! -f ${target}_paired.hhm ];then
         printf "*%.0s" {1..128}; printf '%s\n'
         echo "Generating the HHM file with ${target}_paired.a3m"
         ${hhblits_bin}/hhmake -i ${target}_paired.a3m -o ${target}_paired.hhm
     fi


 else
     echo "Processing PSSM feats : Missing the input MSA (${target}_paired.a3m) file"
     exit 2
 fi

# -----------------------------------------------------------------------------------------------------------------------------
# generate the ESM-MSA feats
if [ -f ${target}_paired.a3m ];then

    if [ ! -f ${target}_paired_esm_msa.pkl ];then
        printf "*%.0s" {1..128}; printf '%s\n'
        echo "Generating ESM-MSA feats with ${target}_paired.a3m"
        ${hhblits_bin}/hhfilter -i ${target}_paired.a3m -o ${target}_paired_filter.a3m -diff 512
        python $SCRIPT_DIR/bin/get_esm_msa_feature.py ${esm_msa_model} . ${target}_paired
    fi
else
    echo "Processing ESM-MSA feautres : Missing the input MSA (${target}_paired.a3m) file"
    exit 2
fi
