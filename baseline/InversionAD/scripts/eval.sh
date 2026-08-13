export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
python -m src.evaluate \
    --eval_strategy inversion \
    --save_dir results/exp_dit_ad \
    --use_ema_model \
    --eval_step 3
