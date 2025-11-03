#!/bin/bash

# -----------------------------------------------------------------------------------------------------------------------------
# the input variables
target=$1
cov=$2
ecut=$3
ncpu=$4


# -----------------------------------------------------------------------------------------------------------------------------
# clean the unstandard residues of target structure --> target_clean.pdb
 egrep -v "HOH|WAT" ${target}.pdb > ${target}_clean.pdb

 # # modify the unstandard residues to standard reisdues --> target_modified.pdb
 sed "s/MEX/CYS/g; s/HID/HIS/g; s/HIE/HIS/g; s/HIP/HIS/g; s/MSE/MET/g; s/ASX/ASN/g; s/GLX/GLN/g; s/TYS/TRP/g" \
 ${target}_clean.pdb > ${target}_modified.pdb

 # # generate the sequence --> target.fasta
 $SCRIPT_DIR/bin/getseq.awk ${target}_modified.pdb > ${target}.fasta

 # # check the length of the sequence
 length=`awk '{if(NR>1){len=len+length($0)}}END{print len}' ${target}.fasta`

 if [ $length -eq 0 ];then
     echo "The length of the ${target}.fasta is empty"
     exit 2
 fi

# -----------------------------------------------------------------------------------------------------------------------------
# generate the MSA from hhblits
 if [ -f ${target}.fasta ];then
 	if [ ! -f ${target}.a3m ];then
         printf "*%.0s" {1..128}; printf '%s\n'
 		echo "Generating MSA of ${target}.fasta"
 		$hhblits_bin/hhblits -i ${target}.fasta -d $UniRef_database -cpu $ncpu -oa3m  ${target}.a3m -n 3 -e $ecut -id 99 -cov $cov
 	fi
 else
 	echo "Missing the sequence file of ${target}"
         exit 2
 fi

# generate MSA feats from target.a3m
 if [ -f ${target}.a3m ];then

     # generate HHM file for PSSM
     if [ ! -f ${target}.hhm ];then
         printf "*%.0s" {1..128}; printf '%s\n'
	 	echo "Generating the HHM file with ${target}.a3m"
	     ${hhblits_bin}/hhmake -i ${target}.a3m -o ${target}.hhm
     fi


 else
     echo "Processing PSSM feats : Missing the input MSA (${target}.a3m) file"
     exit 2
 fi
# -----------------------------------------------------------------------------------------------------------------------------
# generate the sequence feats
 if [ -f ${target}.a3m ];then
 	if [ ! -f ${target}_seqembedding.txt ];then
 	  echo "Generating onehot feats"
 		python $SCRIPT_DIR/bin/get_onehot.py -t $target -d $(pwd) -o $(pwd)
 	fi
 	if [ ! -f ${target}_seqproperty.txt ];then
 	  echo "Generating physicochemical feats"
 		python $SCRIPT_DIR/bin/get_physicochemical.py -t $target -d $(pwd) -o $(pwd)
 	fi
 else
     echo "Processing sequence  feautres : Missing the input MSA (${target}.a3m) file"
     exit 2
 fi
 -----------------------------------------------------------------------------------------------------------------------------
# generate the ESM-MSA feats
if [ -f ${target}.a3m ];then

    if [ ! -f ${target}_esm_msa.pkl ];then
        printf "*%.0s" {1..128}; printf '%s\n'
		echo "Generating ESM-MSA feats with ${target}.a3m"
     	${hhblits_bin}/hhfilter -i ${target}.a3m -o ${target}_filter.a3m -diff 512
	    python $SCRIPT_DIR/bin/get_esm_msa_feature.py ${esm_msa_model} . $target
	fi
else
    echo "Processing ESM-MSA feautres : Missing the input MSA (${target}.a3m) file"
    exit 2
fi


# -----------------------------------------------------------------------------------------------------------------------------
# renumber the residue id from target_modified.pdb --> target_renum.pdb
 awk '{s=substr($0,18,10);
     if(substr($1,1,4)=="ATOM"||substr($1,1,6)=="HETATM"){
         if(s!=s0)n++;
         printf"%s%4d %s\n",substr($0,1,22),n,substr($0,28);
         s0=s
     }
     else{
              print
     }
 }'  ${target}_modified.pdb > ${target}_renum.pdb

# generate the structure feats including intra-dist, SA
 if [ -f ${target}_renum.pdb ];then
     printf "*%.0s" {1..128}; printf '%s\n'
     echo "Generating SA features"

     name_software=$(basename $sa_software)
     if [ $name_software = "naccess" ];then
     	${sa_software} ${target}_renum.pdb
     elif [ $name_software = "freesasa" ];then
         ${sa_software} ${target}_renum.pdb --format=seq -o ${target}_renum.freesasa
     else
         echo "Please set the right software (naccess/freesasa) for SA"
         exit 2
     fi

 	if [ ! -f ${target}_pesto.txt ];then
 		echo "Generating pesto feats with ${target}_renum.pdb"
      	source activate pesto
 	    python $SCRIPT_DIR/bin/get_binding_site.py -t $target -d $(pwd) -o $(pwd)
 		conda deactivate
 	fi

 	if [ ! -f ${target}_USR.txt ];then
 		echo "Generating USR feats with ${target}_renum.pdb"
      	source activate pytorch
 	    python $SCRIPT_DIR/bin/get_usr_feature.py $target $(pwd) $(pwd)
 		conda deactivate
 	fi


 	if [ ! -f ${target}_mon_distance.out ];then
 	  echo "Generating MonDistance feature"
 		$mon_dismap ${target}_renum.pdb ${target}_renum.pdb -o ${target}_mon_distance.out
 	fi

 else
  	echo "Processing structure features: Missing the target pdb file"
 	exit 2
 fi

# -----------------------------------------------------------------------------------------------------------------------------
# generate the antiberty feats
if [ -f ${target}.a3m ];then

    if [ ! -f ${target}_antiberty.txt ];then
        printf "*%.0s" {1..128}; printf '%s\n'
		echo "Generating antiberty feats with ${target}.a3m"
		  source activate pytorch
     	python $SCRIPT_DIR/bin/get_antiberty_embedding.py -d $(pwd) -o $(pwd)
	fi
else
    echo "Processing antiberty feautres : Missing the input MSA (${target}.a3m) file"
    exit 2
fi
