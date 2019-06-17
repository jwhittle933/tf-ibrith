# Check for and install the necessary python packages.
TF=$(python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))")
NP=$(python3 -c "import numpy as np; print(np.version.version)")

if [ -z "$TF" ]
then
    echo "Installing tensorflow..."
    pip install "tensorflow" "tensorflow-hub" "numpy"
else
    echo "Tensorflow detected..."
fi

if [ -z "$NP" ]
then
    echo "Installing numpy..."
    pip install "numpy"
else
    echo "Numpy detected..."
fi
