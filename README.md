# SEEKR: Selective Attention-Guided Knowledge Retention for Continual Learning of Large Language Models

Official implementation of our paper **"SEEKR: Selective Attention-Guided Knowledge Retention for Continual Learning of Large Language Models"** in **EMNLP 2024**.

## Install

```
conda create -n seekr python=3.10
conda activate seekr
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install flash-attn==2.5.5
```


## Prepare datasets and models

- Download [Trace Benchmark](https://github.com/BeyonderXX/TRACE)
- Download [SuperNI Benchmark](https://drive.google.com/file/d/18h8PNOKbjcaK5DpFlxF45M6of4qIsr-2/view?usp=sharing)

- Modify ``path/to/datasets`` in ``scripts/exp_seq_seekr.sh``
- Modify ``path/to/base_models`` in ``scripts/exp_seq_seekr.sh``

## Continual learning with SEEKR

```
bash scripts/exp_seq_seekr.sh llama2 tracer1
```

## Acknowledgement

This project is built on top of [TRACE](https://github.com/BeyonderXX/TRACE)

