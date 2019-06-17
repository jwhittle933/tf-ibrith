# עברית

## Training Hebrew Character Classifier

## Getting Started
The `src` directory houses the source code for the repo, but the scripts at the root can be used as a proxy to access them. `setup.sh` checks for the needed deps (tensorflow and numpy) which are used in `retrain.py` and `label_image.py` and installs them if needed. You don't need to call this script directly, as `train.sh` and `label.sh` call these before ever running the script. `graph.sh` opens `tensorboard` and accepts a single argument: the path to the graph. Graphs are kept in `src/graphs` the file names are saved in the format date|timestamp. This will be the practice for all stored files (graphs, output labels, saved models directories, and summaries). 

## Links
* [Build a Classifier](https://www.youtube.com/watch?v=QfNvhPx5Px8)
* [TF in 5 Minutes](https://www.youtube.com/watch?v=2FmcHiLCwTU)
* [Image Classifier](https://www.tensorflow.org/hub/tutorials/image_retraining)
* [Docker Container ](https://hub.docker.com/r/tensorflow/tensorflow/)
* [Image Labels](https://towardsdatascience.com/multi-label-image-classification-with-inception-net-cbb2ee538e30)
