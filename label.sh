#bin/bash
# $1 = dated dir of graph and label, which should be the same
# $2 = path to new image
sh ./setup.sh

python src/label_image.py \
       --graph=src/graphs/$1 \
       --labels=src/output_labels/$1 \
       --input_layer=Placeholder \
       --output_layer=final_result \
       --image=$2
