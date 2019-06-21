# עברית

## Training Hebrew Character Classifier

## Getting Started
The `src` directory houses the source code for the repo, but the scripts at the root can be used as a proxy to access them. `setup.sh` checks for the needed deps (tensorflow and numpy) which are used in `retrain.py` and `label_image.py` and installs them if needed. You don't need to call this script directly, as `train.sh` and `label.sh` call these before ever running the script. `graph.sh` opens `tensorboard` and accepts a single argument: the path to the graph. Graphs are kept in `src/graphs` the file names are saved in the format date|timestamp. This will be the practice for all stored files (graphs, output labels, saved models directories, and summaries). 

The `modules` directory contains our labeled images. Images are labeled by the directory name that holds them, i.e., `aleph` dir holds images with label `aleph` for training. It's important that we don't nest any directories below this label level, as TF won't be able to train and will error out. As well, more than 1 directory must be present to train (as `retrain.py` takes files at random from the various directories to train on), and there can be no invalid directories, i.e., a directory named `mem` that is empty. Moreover, each image must be a .jpg/.jpeg, as TF will uses pixels to train on the images (opposed to a .png, which uses lines). If you screen grab a image from a Mac, the image will be saved as a png and need to be converted. Open the image and export as jpeg, and set the resolution to the highest level. It's also important to mmake sure our images are very high quality. The screen grabs currently in `modules` were grabbed from the [Dead Sea Scrolls Archive](http://dss.collections.imj.org.il/community) online, and are very high quality. As well, there's lots of natural distortion in the images due to the decaying nature of the vellum/parchment/papyrus. TF can accept randomized image interference, but these images have some naturally. This is a great place to go grab more images if the modules need to be beefed up (and they definitely do). A good prelimary goal is to get at least __100__ images into each module, for all 22 Hebrew characters.

## Links
* [Build a Classifier](https://www.youtube.com/watch?v=QfNvhPx5Px8)
* [TF in 5 Minutes](https://www.youtube.com/watch?v=2FmcHiLCwTU)
* [Image Classifier](https://www.tensorflow.org/hub/tutorials/image_retraining)
* [Docker Container ](https://hub.docker.com/r/tensorflow/tensorflow/)
* [Image Labels](https://towardsdatascience.com/multi-label-image-classification-with-inception-net-cbb2ee538e30)
