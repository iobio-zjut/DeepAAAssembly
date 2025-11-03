import os
import numpy as np
import re
import sys
def read_file(file_path):
    residues = {}
    solvent_areas = {}
    chain1_number = {}
    i = 1  # 初始化计数器
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split()
            residue1 = data[0]
            area = float(data[2])
            pattern = r'c<(\w+)>r<(\d+)>'
            matches1 = re.search(pattern, residue1)

            if matches1:
                chain1_name = matches1.group(1)
                residue1_number = matches1.group(2)
            else :
                print("2area不正确",file_path,line)
                residues = None
                solvent_areas = None
                return residues,solvent_areas
            residue1_id = chain1_name + " " + residue1_number
            if residue1_id not in chain1_number.keys():
                chain1_number[residue1_id] = i
                i += 1  # 更新计数器

    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split()
            residue1 = data[0]
            residue2 = data[1]
            area = float(data[2])

            pattern = r'c<(\w+)>r<(\d+)>'
            matches1 = re.search(pattern, residue1)
            matches2 = re.search(pattern, residue2)
            if matches1 and matches2:
                chain1_name = matches1.group(1)
                residue1_number = matches1.group(2)
                chain2_name = matches2.group(1)
                residue2_number = matches2.group(2)
            residue1_id = chain1_name + " " + residue1_number
            residue2_id = chain2_name + " " + residue2_number


            if residue2 == "c<solvent>":
                if residue1_id in solvent_areas:
                    solvent_areas[chain1_number[residue1_id]] += area
                else:
                    solvent_areas[chain1_number[residue1_id]] = area
                continue
            
            if residue1_id != residue2_id:

                residue_pair = tuple(sorted([chain1_number[residue1_id], chain1_number[residue2_id]]))
                residues.setdefault(residue_pair, 0)
                residues[residue_pair] += area

    return residues, solvent_areas


def write_output(output_npz_file, residues, solvent_areas):
    max_residue_id = max(
        max(int(pair[0]), int(pair[1])) for pair in residues.keys()
    )
    voro_area = np.zeros((max_residue_id, max_residue_id))
    for residue_pair, area in residues.items():
        residue1, residue2 = residue_pair
        voro_area[int(residue1) - 1][int(residue2) - 1] = area
        voro_area[int(residue2) - 1][int(residue1) - 1] = area

    max_residue_id2 = max(map(int, solvent_areas.keys()))
    voro_solvent_area = np.zeros(max_residue_id2)
    for residue_id in range(1, max_residue_id2):
        area = solvent_areas.get(residue_id, 0)
        voro_solvent_area[residue_id] = area

    np.savez(output_npz_file, voro_area=voro_area, voro_solvent_area=voro_solvent_area)


def process_folder(folder_path):
    folder_name = os.path.basename(folder_path)
    output_folder_path = os.path.join(result_folder , folder_name)
    os.makedirs(output_folder_path, exist_ok=True)
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            input_file = os.path.join(folder_path, file_name)
            # 检查文件大小是否不为0
            if os.path.getsize(input_file) != 0:
                
                residues_area, solvent_areas = read_file(input_file)
                if residues_area != None:
                    output_npz_file = os.path.join(output_folder_path, f"{os.path.splitext(file_name)[0]}_voro_area.npz")
                    write_output(output_npz_file, residues_area, solvent_areas)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python 2.area.complex.py subfolder_file root_folder")
        sys.exit(1)
subfolder_file = sys.argv[1]
root_folder = sys.argv[2]
result_folder = sys.argv[3]

input_folder_list = subfolder_file
with open(input_folder_list, 'r') as f:
    for folder_path in f:
        folder_path = root_folder + "/" + folder_path.strip()
        process_folder(folder_path)
