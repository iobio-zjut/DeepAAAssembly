#!/bin/awk -f 

BEGIN{
	split("ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET MSE ASN PRO GLN ARG SER THR VAL TRP TYR", resa, " ");
	split("A C D E F G H I K L M M N P Q R S T V W Y", resc, " ");
	nfile=0;
} 
{
	if(FNR==1){nfile++; 
		if(NR>1)print""
		print ">"FILENAME;
	};
	s=substr($0,18,10);
	atmn=substr($0,13,4);
	gsub(/ /,"",atmn);
	if((substr($1,1,4)=="ATOM"||substr($1,1,6)=="HETATM") && s!=s0){
		nres++;flag=0;
		resn=substr($0,18,3);
		gsub(/ /,"",resn);
		for(i in resa){if(resa[i]==resn){flag=1;break}};
		if(flag>0){
			printf"%s",resc[i]
		}else{
                        printf "\n";
			printf"nonstandard residue %s \n", resn;
                        printf"please delete the nonstandard residue of targe pdb"
                        exit;

		}
		if(nres%60==0)printf"\n";
	};
	if($1=="TER"){nter++;printf"\n";nres=0};
	s0=s;
} END{
	printf"\n"
} 
