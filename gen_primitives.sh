cd build
DYN=$1
PRIM=$2
OUT_FILE=$3
VIZ_FILE=$4

./main_primitives --mode_gen_id 0 --dynamics $DYN --models_base_path ../dynobench/models/ --max_num_primitives $PRIM --out_file $OUT_FILE --solver_id 0 --cfg_file ../opt_params.yaml 

./main_primitives --mode_gen_id 1 --dynamics $DYN --models_base_path ../dynobench/models/ --max_num_primitives $PRIM --in_file $OUT_FILE --solver_id 1 --cfg_file ../opt_params.yaml 

./main_primitives --mode_gen_id 2 --in_file  ${OUT_FILE}.im.bin  --max_num_primitives -1  --max_splits 4  --max_length_cut 40  --min_length_cut 20 --dynamics $DYN --models_base_path ../dynobench/models/
python3 ../scripts/visualize_prims.py --prims ${OUT_FILE}.im.bin.sp.bin.yaml --output $VIZ_FILE --num_samples 20

