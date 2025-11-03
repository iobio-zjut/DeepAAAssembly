import os,sys
import string
import numpy as np

def extract_taxid(file, gap_cutoff=0.8):

    lines = open(file, 'r').readlines()
    query = lines[1].strip().translate(translation)
    seq_len = len(query)

    msas = [ query ]
    sid = [0]
    for line in lines[2:]:

        if line[0] == ">":
            if "TaxID=" in line:
                content = line.split("TaxID=")[1]
                if len(content) > 0:
                    try:
                        sid.append(int(content.split()[0]))
                    except:
                        sid.append(0)
            elif "OX=" in line:
                content = line.split("OX=")[1]
                if len(content) > 0:
                    try:
                        sid.append(int(content.split()[0]))
                    except:
                        sid.append(0)
            else:
                sid.append(0)
            continue

        seq = line.strip().translate(translation)
        gap_fra = float(seq.count('-'))/seq_len
        if gap_fra <= gap_cutoff:
            msas.append(seq)
        else:
            sid.pop(-1)

    if len(msas) != len(sid):
        print("ERROR: len(msas) != len(sid)")
        print(len(msas), len(sid))
        exit()

    return msas, np.array(sid)


def cal_identity(query, sub_msas):
    """
    Args:
        query : str
        sub_msas : List[str]
    Return:
        identity : np.array
    """

    identity = np.zeros((len(sub_msas)))
    seq_len = len(query)
    ones = np.ones(seq_len)
    for idx, seq in enumerate(sub_msas):
        match =  [ query[i] == seq[i] for i in range(seq_len) ] 
        counts = np.sum( ones[match])
        identity[idx] = counts/seq_len

    return identity

def alignment(msas1, sid1, msa2, sid2, top=True):

    # obtain the same species and delete species=0
    smatch = np.intersect1d(sid1, sid2)
    smatch = smatch[np.argsort(smatch)]
    smatch = np.delete(smatch, 0)

    query1 = msas1[0]
    query2 = msas2[0]
    aligns = [ query1 + query2 ]

    for id in smatch:

        index1 = np.where(sid1==id)[0]
        sub_msas1 = [ msas1[idx] for idx in index1]
        identity1 = cal_identity(query1, sub_msas1)
        sort_idx1 = np.argsort(-identity1)

        index2 = np.where(sid2==id)[0]
        sub_msas2 = [ msas2[idx] for idx in index2]
        identity2 = cal_identity(query2, sub_msas2)
        sort_idx2 = np.argsort(-identity2)

        if top == True:
            aligns.append( sub_msas1[sort_idx1[0]] + \
                           sub_msas2[sort_idx2[0]] )
        else:
            num = min(len(sub_msas1), len(sub_msas2))
            for i in range(num):
                aligns.append( sub_msas1[sort_idx1[i]] + \
                               sub_msas2[sort_idx2[i]] )
 
    return aligns


def write_a3m(pdb, aligns, out_path):

    n = len(aligns)
    with open(out_path, 'w') as f:
        f.write(">" + pdb + "\n")
        f.write(aligns[0] + "\n")

        for idx, aligned_seq in enumerate(aligns[1:]):
            f.write(">" + str(idx+1) + "\n")
            f.write(aligned_seq + "\n")
            
if __name__ == "__main__":


    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)


    a3m1 = sys.argv[1]
    a3m2 = sys.argv[2]
    paired_msa = sys.argv[3]
    pdb = sys.argv[4]

    msas1, sid1 = extract_taxid( a3m1 )
    msas2, sid2 = extract_taxid( a3m2 )
    aligns = alignment(msas1, sid1, msas2, sid2, top=True)
    write_a3m(pdb, aligns, paired_msa)



