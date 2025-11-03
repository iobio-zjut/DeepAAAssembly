#!/bin/bash


python ./simulation/run_rosetta/utils/coord_trans_all.py \
        --base_path ./simulation/run_rosetta/DB5.5/stage_1 \
        --trans_mat_subdir trans_mat_5model_bestscore --output_subdir trans_pos_5model_bestscore



python ./simulation/run_rosetta/utils/union_to_complex.py \
        --base_path ./simulation/run_rosetta/DB5.5/stage_1 \
        --input_subdir trans_pos_5model_bestscore --output_subdir after_tran_complex_5model_bestscore
