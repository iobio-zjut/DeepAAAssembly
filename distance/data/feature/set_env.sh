#!/bin/bash
#Users need to properly set the following variables after installatin
# -------------------------------------------------------------------------------------

export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#export HHsuite_root=$SCRIPT_DIR/bin/hhsuite-3.2.0
export ccmpred=$SCRIPT_DIR/bin/ccmpred
#export sa_software=$SCRIPT_DIR/third_party/naccess/naccess
export sa_software=$SCRIPT_DIR/bin/freesasa
export UniRef_database=./distance/data/uniref30_former/UniRef30_2021_03
export esm_msa_model=$SCRIPT_DIR/bin/esm_pretrain_models/esm_msa1_t12_100M_UR50S.pt
# -------------------------------------------------------------------------------------
#Users usually do not need to change the following variables
export hhblits_bin=$SCRIPT_DIR/bin/

export reformat=$SCRIPT_DIR/bin/reformat.pl
export mon_dismap=$SCRIPT_DIR/bin/dis_rec_lig_all
export clean_pdb=$SCRIPT_DIR/bin/clean_pdb
