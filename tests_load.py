import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from train_variants import split_train_and_test
from train_loop import augment
import numpy as np
import net

size = 512
batch_size = 4

images = np.load('input' + '.npy', mmap_mode='r')
labels = np.load('input' + '_labels.npy', mmap_mode='r')
training, test = split_train_and_test(images, labels)
training_images, training_labels = training

with tf.Session() as sess:

    new_saver = tf.train.import_meta_graph('models/modelname.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('models'))

    print('trainables:', tf.trainable_variables())

    training_set = tf.data.Dataset.from_tensor_slices((training_images, training_labels), name="new1")\
        .apply(tf.data.experimental.shuffle_and_repeat(buffer_size=training_images.shape[0]))\
        .map(lambda im, lab: tf.py_func(augment, [im, lab, size], [im.dtype, lab.dtype]), num_parallel_calls=4, name="new")\
        .batch(batch_size)\
        .prefetch(1)

    next_training = training_set.make_one_shot_iterator().get_next()

    batch_imgs, batch_labs = sess.run(next_training)

    print("Before the loop..")

    graph = tf.get_default_graph()
    prob = graph.get_tensor_by_name('probabilities:0')

    # print("prob:", prob)

    prob_val = sess.run(prob, {'input:0': batch_imgs, 'labels:0': batch_labs})

    print("prob_val:", prob_val)

