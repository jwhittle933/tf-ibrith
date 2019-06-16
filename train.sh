#bin/bash
NOW=$(date +"%m-%d-%Y|%T")
python src/retrain.py \
       --image_dir src/modules \
       --summaries_dir src/summaries/$NOW \
       --saved_model_dir src/saved_models/$NOW \
       --output_graph src/graphs/$NOW \
       --output_labels src/output_labels/$NOW \
       # --flip_left_right \
       # --random_crop 5 \
       # --random_scale 5 \
       # --random_brightness 5 \
