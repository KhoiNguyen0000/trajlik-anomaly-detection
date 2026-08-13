# Replace the config path below when training another dataset/category.
python ./main.py \
    --task train \
    --fname configs/exp_dit_ad/single_class.yml \
    --devices cuda:0 \
    --port 12346
