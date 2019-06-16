#bin/bash
python src/retrain.py \
       --image_dir src/modules \
       --summaries_dir src/summaries/$(date +"%m-%d-%Y|%T")
