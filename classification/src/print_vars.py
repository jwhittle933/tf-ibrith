import tensorflow as t

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('../coco/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('../coco/'))
