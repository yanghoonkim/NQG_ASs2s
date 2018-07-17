PARAMS=h100_setting7_m1_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h100_setting7_m1_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h100_setting7_m2_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h100_setting7_m2_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h100_setting7_m4_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h100_setting7_m4_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h100_setting7_m8_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h100_setting7_m8_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h100_setting7_m20_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h100_setting7_m20_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h100_setting7_m50_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h100_setting7_m50_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h100_setting7_m100_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h100_setting7_m100_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h100_setting7_m256_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h100_setting7_m256_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h200_setting7_m1_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h200_setting7_m1_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h200_setting7_m2_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h200_setting7_m2_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h200_setting7_m4_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h200_setting7_m4_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h200_setting7_m8_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h200_setting7_m8_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h200_setting7_m20_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h200_setting7_m20_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h200_setting7_m50_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h200_setting7_m50_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h200_setting7_m100_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h200_setting7_m100_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h200_setting7_m256_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h200_setting7_m256_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h350_setting7_m1_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h350_setting7_m1_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h350_setting7_m2_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h350_setting7_m2_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h350_setting7_m4_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h350_setting7_m4_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h350_setting7_m8_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h350_setting7_m8_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h350_setting7_m20_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h350_setting7_m20_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h350_setting7_m50_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h350_setting7_m50_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h350_setting7_m100_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h350_setting7_m100_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h350_setting7_m256_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h350_setting7_m256_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h512_setting7_m1_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h512_setting7_m1_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h512_setting7_m2_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h512_setting7_m2_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h512_setting7_m4_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h512_setting7_m4_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h512_setting7_m8_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h512_setting7_m8_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h512_setting7_m20_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h512_setting7_m20_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h512_setting7_m50_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h512_setting7_m50_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h512_setting7_m100_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h512_setting7_m100_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h512_setting7_m256_batch128
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h512_setting7_m256_batch128 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h100_setting7_m1_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h100_setting7_m1_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h100_setting7_m2_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h100_setting7_m2_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h100_setting7_m4_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h100_setting7_m4_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h100_setting7_m8_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h100_setting7_m8_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h100_setting7_m20_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h100_setting7_m20_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h100_setting7_m50_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h100_setting7_m50_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h100_setting7_m100_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h100_setting7_m100_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h100_setting7_m256_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h100_setting7_m256_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h200_setting7_m1_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h200_setting7_m1_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h200_setting7_m2_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h200_setting7_m2_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h200_setting7_m4_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h200_setting7_m4_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h200_setting7_m8_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h200_setting7_m8_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h200_setting7_m20_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h200_setting7_m20_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h200_setting7_m50_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h200_setting7_m50_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h200_setting7_m100_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h200_setting7_m100_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h200_setting7_m256_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h200_setting7_m256_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h350_setting7_m1_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h350_setting7_m1_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h350_setting7_m2_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h350_setting7_m2_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h350_setting7_m4_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h350_setting7_m4_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h350_setting7_m8_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h350_setting7_m8_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h350_setting7_m20_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h350_setting7_m20_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h350_setting7_m50_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h350_setting7_m50_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h350_setting7_m100_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h350_setting7_m100_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h350_setting7_m256_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h350_setting7_m256_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h512_setting7_m1_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h512_setting7_m1_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h512_setting7_m2_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h512_setting7_m2_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h512_setting7_m4_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h512_setting7_m4_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h512_setting7_m8_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h512_setting7_m8_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h512_setting7_m20_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h512_setting7_m20_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h512_setting7_m50_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h512_setting7_m50_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h512_setting7_m100_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h512_setting7_m100_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h512_setting7_m256_batch256
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h512_setting7_m256_batch256 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h100_setting7_m1_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h100_setting7_m1_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h100_setting7_m2_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h100_setting7_m2_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h100_setting7_m4_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h100_setting7_m4_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h100_setting7_m8_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h100_setting7_m8_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h100_setting7_m20_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h100_setting7_m20_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h100_setting7_m50_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h100_setting7_m50_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h100_setting7_m100_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h100_setting7_m100_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h100_setting7_m256_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h100_setting7_m256_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h200_setting7_m1_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h200_setting7_m1_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h200_setting7_m2_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h200_setting7_m2_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h200_setting7_m4_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h200_setting7_m4_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h200_setting7_m8_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h200_setting7_m8_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h200_setting7_m20_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h200_setting7_m20_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h200_setting7_m50_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h200_setting7_m50_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h200_setting7_m100_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h200_setting7_m100_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h200_setting7_m256_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h200_setting7_m256_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h350_setting7_m1_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h350_setting7_m1_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h350_setting7_m2_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h350_setting7_m2_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h350_setting7_m4_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h350_setting7_m4_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h350_setting7_m8_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h350_setting7_m8_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h350_setting7_m20_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h350_setting7_m20_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h350_setting7_m50_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h350_setting7_m50_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h350_setting7_m100_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h350_setting7_m100_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h350_setting7_m256_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h350_setting7_m256_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h512_setting7_m1_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h512_setting7_m1_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h512_setting7_m2_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h512_setting7_m2_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h512_setting7_m4_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h512_setting7_m4_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h512_setting7_m8_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h512_setting7_m8_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h512_setting7_m20_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h512_setting7_m20_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h512_setting7_m50_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h512_setting7_m50_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h512_setting7_m100_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h512_setting7_m100_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

PARAMS=h512_setting7_m256_batch512
MODEL_DIR=./store_model/$PARAMS
python main.py --mode='pred' --model_dir=$MODEL_DIR --params=$PARAMS
echo h512_setting7_m256_batch512 >> bleu_all.txt
perl result/bleu result/mpqg_test.txt < result/squad.txt >> bleu_all.txt
echo -e >> bleu_all.txt

