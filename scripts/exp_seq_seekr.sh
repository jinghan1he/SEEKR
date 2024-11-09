#!/bin/bash
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

base_model=$1
dataset=$2

### base model
if [[ $base_model == llama2 ]]; then
cached_model="/path/to/base_models/Llama-2-7b-chat-hf/"
elif [[ $base_model == vicuna1.5 ]]; then
cached_model="/path/to/base_models/models--lmsys--vicuna-7b-v1.5/"
elif [[ $base_model == vicuna1.5-13b ]]; then
cached_model="/path/to/base_models/models--lmsys--vicuna-13b-v1.5"
fi

### benchmark and order
if [[ $dataset == tracer1 ]]; then
data_dir="/path/to/datasets/TRACE-Benchmark/LLM-CL-Benchmark_Reasoning"
task_order="C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten"
epochs=5,5,5,5,5,5,10,5
test_rounds=("0,7" "1,7" "2" "7" "3" "7" "4,5,7" "6,7")
test_tasks=("0" "1" "2" "2" "3" "3" "4,5" "6,7")

elif [[ $dataset == tracer2 ]]; then
data_dir="/path/to/datasets/TRACE-Benchmark/LLM-CL-Benchmark_Reasoning"
task_order="NumGLUE-cm,NumGLUE-ds,FOMC,20Minuten,C-STANCE,Py150,MeetingBank,ScienceQA"
epochs=10,10,10,5,5,5,5,5
test_rounds=("4,7" "2,7" "6" "7" "5" "7" "0,1,7" "3,7")
test_tasks=("4" "2" "6" "6" "5" "5" "0,1" "3,7")

elif [[ $dataset == superni1 ]]; then
data_dir="/path/to/datasets/SuperNI_CL"
task_order="task1572,task363,task1290,task181,task002,task1510,task073,task748,task511,task591,task195,task875"
epochs=10,10,10,10,10,10,10,10,10,10,10,10
test_rounds=("0,11,1" "2,11,3" "4,11,5" "6,11,7" "8,11,9" "10,11")
test_tasks=("1,0" "3,2" "5,4" "7,6" "9,8" "11,10")

elif [[ $dataset == superni2 ]]; then
data_dir="/path/to/datasets/SuperNI_CL"
task_order="task748,task073,task1572,task195,task591,task363,task1510,task181,task511,task002,task1290,task875"
epochs=10,10,10,10,10,10,10,10,10,10,10,10
test_rounds=("0,11,1" "2,11,3" "4,11,5" "6,11,7" "8,11,9" "10,11")
test_tasks=("1,0" "3,2" "5,4" "7,6" "9,8" "11,10")

fi

### continual learning with seekr
replay_ratio=0.01
attns_distill_weight=1e3

attn_layer_budget=24
attn_head_budget=128
attn_query_budget=0

method="seekr-$replay_ratio-alb$attn_layer_budget-ahb$attn_head_budget-aqb$attn_query_budget/"
output_dir="outputs_LLM-CL/$dataset/$base_model/$method"
mkdir -p $output_dir

port=$(shuf -i25000-30000 -n1)
deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port $port training/seekr.py  \
    --data_path $data_dir \
    --dataset_name $task_order \
    --model_name_or_path $cached_model \
    --per_device_train_batch_size 2 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --num_train_epochs $epochs \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --zero_stage 2 \
    --deepspeed \
    --past_task_ratio $replay_ratio \
    --attns_distill_weight $attns_distill_weight \
    --attn_layer_budget $attn_layer_budget \
    --attn_head_budget $attn_head_budget \
    --attn_query_budget $attn_query_budget \
    --enable_wandb \
    --output_dir $output_dir > $output_dir/train.log 2>&1

if [[ -f $output_dir/tmp.bin ]]; then
rm $output_dir/tmp.bin
fi

### inference
for i in {0..7}; do
CUDA_VISIBLE_DEVICES=$i python inference/infer_single.py  \
    --data_path $data_dir \
    --inference_tasks $task_order \
    --test_rounds ${test_rounds[i]} \
    --test_tasks ${test_tasks[i]} \
    --model_name_or_path  $cached_model \
    --inference_model_path $output_dir \
    --inference_batch 16 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --seed 1234 \
    --inference_output_path $output_dir/predictions &
done

wait

### remove the intermediate checkpoints
IFS=', ' read -r -a tasks <<< $task_order
for i in $(seq 0 $((${#tasks[*]}-1))); do
if [[ -f $output_dir/predictions/results-$i-$i-${tasks[i]}.json ]]; then
rm -r $output_dir/$i
fi
done

### show evaluation results
python re_eval.py --output_dir $output_dir
python show_results.py --output_dir $output_dir
