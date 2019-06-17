# Install the necessary python packages.
TF=$(python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))")

[ -z "$test" ] && pip install "tensorflow" "tensorflow-hub" "numpy"
