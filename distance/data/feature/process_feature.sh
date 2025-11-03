#!/bin/bash

# -----------------------------------------------------------------------------------------------------------------------------
# source the defined variables and the operating environment of DeepAAAssembly
# 获取当前脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 进入脚本所在目录
cd "$SCRIPT_DIR"
module load anaconda
source set_env.sh
source activate DeepTMP


# -----------------------------------------------------------------------------------------------------------------------------
# command args
if [ $# -lt 3 ];then
	echo ""
	echo "USAGE: `basename $0` prot1.pdb prot2.pdb target_name [options]"
	echo ""
	echo "Descriptions:"
	echo "    prot1.pdb   : input, the file of prot1 strutcture(*.pdb)"
    echo "    prot2.pdb   : input, the file of prot2 strutcture(*.pdb)"
    echo "    target_name : the name of the dimers"
	echo "    -cov        : the coverage of hhblits, default --> 0.4"
	echo "    -ecut       : the e-value cutoff of hhblits, default --> 0.001"
	echo "    -ncpu       : the number of cpu for hhblits, default --> 3"
	echo "    -db         : the database for hhblits, default --> UniRef30_2020_03"
	echo "    -ntop       : output the top n predicted contacts, default --> all"
	echo "    -out        : the output filename for predicted contacts, default --> target_prediction.txt"
    echo "    -gpu        : use the gpu to calculate the inter-protein contacts, default-->False"
	echo ""
        exit 1
fi

# -----------------------------------------------------------------------------------------------------------------------------
# !!! The lines after this normally do not need to be modified. !!!

# default parameters
cov=0.4
ecut=0.001
ntop=all
ncpu=25
gpu=True

input_prot1=$1
input_prot2=$2
target1=$(basename $input_prot1 .pdb)
target2=$(basename $input_prot2 .pdb)
target=$3
fout=$3_prediction.txt


while [ $# -gt 3 ];do
    case $4 in

    -cov)
        shift
        cov=$4;;
    -ecut)
        shift
        ecut=$4;;
    -ncpu)
        shift
        npcu=$4;;
    -db)
        shift
        UniRef_database=$4;;
    -ntop)
        shift
        ntop=$4;;
    -out)
        shift
        fout=$4;;
    -gpu)
        shift
        gpu=$4;;
	-dp)
        shift
        data_path=$4;;
    *)
    	echo " ERROR: wrong command argument \"$2\" !!"
    	echo " Type \"$0\" for help !!"
        exit 2;;
    esac
    shift
done

# Check the path of required programs or variables

if [ `which $mon_dismap 2>/dev/null | wc -l` -eq 0 ];then
  echo ""
  echo "ERROR: \"$mon_dismap\" is not found !!"
  echo "Please set the \"SCRIPT_DIR\" parameter in set_env.sh properly!"
  echo ""
  exit 2
fi

if [ `which $reformat 2>/dev/null | wc -l` -eq 0 ];then
  echo ""
  echo "ERROR: \"$reformat\" is not found !!"
  echo "Please set the \"HHsuite_root\" parameter in set_env.sh properly!"
  echo ""
  exit 2
fi

if [ `which $ccmpred 2>/dev/null | wc -l` -eq 0 ];then
  echo ""
  echo "ERROR: \"$ccmpred\" is not found !!"
  echo "Please set the \"ccmpred\" parameter in set_env.sh properly!"
  echo ""
  exit 2
fi

if [ `which $sa_software 2>/dev/null | wc -l` -eq 0 ];then
  echo ""
  echo "ERROR: \"$sa_software\" is not found !!"
  echo "Please set the \"$sa_software\" parameter in set_env.sh properly!"
  echo ""
  exit 2
fi

if [ `ls ${UniRef_database}.* 2>/dev/null | wc -l` -eq 0 ];then
  echo ""
  echo "ERROR: \"$UniRef_database\" is not found !!"
  echo "Please set the \"UniRef_database\" parameter in set_env.sh properly!"
  echo ""
  exit 2
fi

if [ `ls $esm_msa_model 2>/dev/null | wc -l` -eq 0 ];then
  echo ""
  echo "ERROR: \"$esm_msa_model\" is not found !!"
  echo "Please set the \"esm_msa_model\" parameter in set_env.sh properly!"
  echo ""
  exit 2
fi

if [ `ls $SCRIPT_DIR/bin/LoadHHM.py 2>/dev/null | wc -l` -eq 0 ];then
  echo ""
  echo "ERROR: \"LoadHHM.py\" is not found !!"
  echo "Please install the \"LoadHHM.py\" file in the proper directory!"
  echo ""
  exit 2
fi


# -----------------------------------------------------------------------------------------------------------------------------
# the main process of DeepAAAssembly

# mkdir temporaty directory to save the data
cdir=$(pwd)
data_path=$data_path
echo "$data_path"
echo "$ptype"
mkdir $data_path 2>/dev/null
cd $data_path

# compare the two input proteins to distinguish the homo and hetero and genereate the needed feats
if [ $input_prot1 == $input_prot2 ];then
    # model=$model1
    ptype="homo"
    # ln -sf $cdir/${input_prot1}
#     $SCRIPT_DIR/gen.sh ${target1} $cov $ecut $ncpu
else
    # model=$model2
    ptype="hetero"

    $SCRIPT_DIR/gen.sh ${target1} $cov $ecut $ncpu
    $SCRIPT_DIR/gen.sh ${target2} $cov $ecut $ncpu
    $SCRIPT_DIR/gen_paired.sh ${target1} ${target2} ${target}
fi


# -----------------------------------------------------------------------------------------------------------------------------
# integrated the preprocess feats into target.npz file
 echo "Packing Features"
 python $SCRIPT_DIR/bin/GenerateFeaturesPkl.py  -d .  -t1 $target1  -t2 $target2 -t $target -m $ptype

