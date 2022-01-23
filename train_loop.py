import os
import math
import random
import time
from datetime import datetime
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import net
from deformations import elastically_deform_image_2d
import progress

import wandb
wandb.init(project="tbcnn", entity="joeljosephjin")

import gc

# Remove tf annoying logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def augment(image, label, size):
    "Augments image and returns the augmented tensor"

    if random.random() > 0.2:
        # Augment with elastic deformations
        image = elastically_deform_image_2d(image[:,:,0], 2, 32)
        # Go back to initial image shape
        image = image.reshape(image.shape + (1,))
    
    # Assume image is square
    max_displacement = image.shape[0] - size
    displacement_x = int(random.random() * max_displacement)
    displacement_y = int(random.random() * max_displacement)

    image = image[
        displacement_y:displacement_y+size,
        displacement_x:displacement_x+size
    ]

    return image, label

# Use tensorflow Dataset API to improve the performances of the training set
# Shuffle, augment and created batches for each epoch
def create_dataloader(training_images, training_labels, size, batch_size):
    training_set = tf.data.Dataset.from_tensor_slices((training_images, training_labels)).apply(tf.data.experimental.shuffle_and_repeat(buffer_size=training_images.shape[0])).map(lambda im, lab: tf.py_func(augment, [im, lab, size], [im.dtype, lab.dtype]), num_parallel_calls=4).batch(batch_size).prefetch(1)

    del training_images
    gc.collect()

    return training_set.make_one_shot_iterator().get_next()

def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def train_net_mod(training, test, size=512, epochs=400, batch_size=4, logging_interval=5, run_name=None):
    """Train network using the given training and test data.
    """

    if run_name is None:
        run_name = datetime.now().strftime(r'%Y-%m-%d_%H:%M')

    training_images, training_labels = training
    test_images, test_labels = test

    # Crop center from test images
    border = (test_images.shape[1] - size) // 2
    test_images = test_images[:,border:border+size, border:border+size]

    epoch_size = int(math.ceil(training_images.shape[0] / batch_size))

    # Create network
    inp_var, labels_var, output = net.generate_network(size)
    error_fn, train_fn, metrics = net.generate_functions(inp_var, labels_var, output)

    print('Parameter number: {}'.format( np.sum([np.prod(v.shape) for v in tf.trainable_variables()]) ))

    # Create tensorboard summaries
    metrics_summary = progress.create_metrics_summary(metrics)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # Initialize weights
        sess.run(tf.global_variables_initializer())
        gc.collect()
        # Initialite tensorboard
        progress.init_run(run_name)
        gc.collect()

        # Training loop
        for e in range(epochs):
            start = time.time()

            # Initialize accuracy calculation
            sess.run(tf.local_variables_initializer())

            # Get needed functions
            accuracy_fn, accuracy_update = metrics['accuracy']
            auc_fn, auc_update = metrics['AUC']
            precision_fn, precision_update = metrics['precision']
            recall_fn, recall_update = metrics['recall']

            # splitting the training set into 4 or more pieces to avoid ram crashing on dataloader
            n_split=4
            training_images_list = list(split_list(training_images, n_split))
            training_labels_list = list(split_list(training_labels, n_split))
            for i, (training_images_i, training_labels_i) in enumerate(zip(training_images_list, training_labels_list)):
                print(f'Loading data set {i}...')
                next_training = create_dataloader(training_images_i, training_labels_i, size, batch_size)
                epoch_size = int(math.ceil(training_images_i.shape[0] / batch_size))
                print(f'Loaded data set {i}...')
            
                # for b in range(epoch_size):
                #     batch_imgs, batch_labs = sess.run(next_training)
                #     gc.collect()

                #     # Train
                #     sess.run([train_fn, accuracy_update, auc_update, precision_update], {
                #         'input:0': batch_imgs,
                #         'labels:0': batch_labs,
                #     })
                #     gc.collect()

                #     # Provide some feedback
                #     print('Batch {} / {}'.format(b + 1, epoch_size), end='\r')
                
                # time.sleep(5)

                del next_training
                gc.collect()

            print("Compute Metrics...")
            # Compute metrics
            accuracy = sess.run(accuracy_fn)
            print('Accuracy Calculated...')
            gc.collect()
            auc = sess.run(auc_fn)
            print('auc Calculated...')
            gc.collect()
            precision = sess.run(precision_fn)
            print('precision Calculated...')
            gc.collect()
            recall = sess.run(recall_fn)
            print('recall Calculated...')
            gc.collect()
            wandb.log({"accuracy":accuracy, "auc":auc, "precision": precision, "recall": recall})

            print("About to compute test metrics...")
            if True:
                # Every logging_interval epochs compute and save results on the test set

                # Reset accuracy and auc for the test set
                sess.run(tf.local_variables_initializer())
                gc.collect()

                # Accuracy on test
                for ti, (img, lab) in enumerate(zip(test_images, test_labels)):
                    sess.run([accuracy_update, auc_update, precision_update, recall_update], {
                        'input:0': img.reshape(1, size, size, -1),
                        'labels:0': [lab],
                    })
                    gc.collect()

                    print('Test image {} / {}'.format(ti + 1, len(test_images)), end='\r')

                # Compute test metrics
                test_accuracy = sess.run(accuracy_fn)
                test_auc = sess.run(auc_fn)
                test_precision = sess.run(precision_fn)
                test_recall = sess.run(recall_fn)
                wandb.log({"test_accuracy":test_accuracy, "test_auc":test_auc, "test_precision":test_precision, "test_recall":test_recall})

                # Collect summaries for tensorboard
                summ_data = sess.run(metrics_summary, {
                    'training_accuracy:0': accuracy,
                    'training_AUC:0': auc,
                    'training_precision:0': precision,
                    'training_recall:0': recall,
                    'test_accuracy:0': test_accuracy,
                    'test_AUC:0': test_auc,
                    'test_precision:0': test_precision,
                    'test_recall:0': test_recall,
                })
                gc.collect()
                # Write summaries to disk
                progress.add_summary(summ_data, e)

            elapsed = time.time() - start
            # Print progress
            print(
                'Epoch {:>3} | Time: {:>3.0f} s | Acc: {:>5.3f} (Test: {:>5.3f}) | AUC: {:>5.3f} (Test: {:>5.3f}) | Precision: {:>5.3f} (Test: {:>5.3f}) | Recall: {:>5.3f} (Test: {:>5.3f})'
                    .format(e, elapsed, accuracy, test_accuracy, auc, test_auc, precision, test_precision, recall, test_recall)
            )


def train_net(training, test, size=512, epochs=400, batch_size=4, logging_interval=5, run_name=None):
    """Train network using the given training and test data.
    """

    if run_name is None:
        run_name = datetime.now().strftime(r'%Y-%m-%d_%H:%M')

    training_images, training_labels = training
    test_images, test_labels = test

    # Crop center from test images
    border = (test_images.shape[1] - size) // 2
    test_images = test_images[:,border:border+size, border:border+size]

    epoch_size = int(math.ceil(training_images.shape[0] / batch_size))

    print("Loading the Training Dataset...")

    # Use tensorflow Dataset API to improve the performances of the training set
    # Shuffle, augment and created batches for each epoch
    training_set = (
        tf.data.Dataset.from_tensor_slices((training_images, training_labels))
            # .shuffle(buffer_size=training_images.shape[0])
            .apply(tf.data.experimental.shuffle_and_repeat(buffer_size=training_images.shape[0]))
            .map(lambda im, lab: tf.py_func(augment, [im, lab, size], [im.dtype, lab.dtype]), num_parallel_calls=4)
            .batch(batch_size)
            .prefetch(1)
    )

    print("Loaded the Training Dataset...")
    
    next_training = training_set.make_one_shot_iterator().get_next()

    print("Loaded the Training Data Iterator...")

    # Create network
    inp_var, labels_var, output = net.generate_network(size)
    error_fn, train_fn, metrics = net.generate_functions(inp_var, labels_var, output)

    print('Parameter number: {}'.format( np.sum([np.prod(v.shape) for v in tf.trainable_variables()]) ))

    # Create tensorboard summaries
    metrics_summary = progress.create_metrics_summary(metrics)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        # Initialize weights
        sess.run(tf.global_variables_initializer())
        # Initialite tensorboard
        progress.init_run(run_name)

        # Training loop
        for e in range(epochs):
            start = time.time()

            # Initialize accuracy calculation
            sess.run(tf.local_variables_initializer())

            # Get needed functions
            accuracy_fn, accuracy_update = metrics['accuracy']
            auc_fn, auc_update = metrics['AUC']
            precision_fn, precision_update = metrics['precision']
            recall_fn, recall_update = metrics['recall']

            for b in range(epoch_size):
                batch_imgs, batch_labs = sess.run(next_training)

                # Train
                sess.run([train_fn, accuracy_update, auc_update, precision_update, recall_update], {
                    'input:0': batch_imgs,
                    'labels:0': batch_labs,
                })

                # Provide some feedback
                # print('Batch {} / {}'.format(b + 1, epoch_size), end='\r')

            # Compute metrics
            accuracy = sess.run(accuracy_fn)
            auc = sess.run(auc_fn)
            precision = sess.run(precision_fn)
            recall = sess.run(recall_fn)
            wandb.log({"accuracy":accuracy, "auc":auc, "precision": precision, "recall": recall})

            if True:
                # Every logging_interval epochs compute and save results on the test set

                # Reset accuracy and auc for the test set
                sess.run(tf.local_variables_initializer())

                # Accuracy on test
                for ti, (img, lab) in enumerate(zip(test_images, test_labels)):
                    sess.run([accuracy_update, auc_update, precision_update, recall_update], {
                        'input:0': img.reshape(1, size, size, -1),
                        'labels:0': [lab],
                    })

                    # print('Test image {} / {}'.format(ti + 1, len(test_images)), end='\r')

                # Compute test metrics
                test_accuracy = sess.run(accuracy_fn)
                test_auc = sess.run(auc_fn)
                test_precision = sess.run(precision_fn)
                test_recall = sess.run(recall_fn)
                wandb.log({"test_accuracy":test_accuracy, "test_auc":test_auc, "test_precision":test_precision, "test_recall":test_recall})

                # Collect summaries for tensorboard
                summ_data = sess.run(metrics_summary, {
                    'training_accuracy:0': accuracy,
                    'training_AUC:0': auc,
                    'training_precision:0': precision,
                    'training_recall:0': recall,
                    'test_accuracy:0': test_accuracy,
                    'test_AUC:0': test_auc,
                    'test_precision:0': test_precision,
                    'test_recall:0': test_recall,
                })
                # Write summaries to disk
                progress.add_summary(summ_data, e)

            elapsed = time.time() - start
            # Print progress
            print(
                'Epoch {:>3} | Time: {:>3.0f} s | Acc: {:>5.3f} (Test: {:>5.3f}) | AUC: {:>5.3f} (Test: {:>5.3f}) | Precision: {:>5.3f} (Test: {:>5.3f}) | Recall: {:>5.3f} (Test: {:>5.3f})'
                    .format(e, elapsed, accuracy, test_accuracy, auc, test_auc, precision, test_precision, recall, test_recall)
            )

        saver.save(sess, "models/modelname")
            
