from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from alexnet import AlexNet
from preprocessingData import *
import tensorflow.contrib.slim as slim
from flip_gradient import flip_gradient
from utils import *
import numpy as np
from mmd import mix_rbf_mmd2

#  Network flags
# tf.app.flags.DEFINE_integer('batchsize',32, 'batchsize')
# tf.app.flags.DEFINE_string('source_filepath','./amazon31.tfrecords', 'path of source dataset')
# tf.app.flags.DEFINE_string('target_filepath','./dslr31.tfrecords', 'path of target dataset')
# tf.app.flags.DEFINE_integer('num_class',31, 'number of classes')
# tf.app.flags.DEFINE_integer('num_steps',5000, 'number of train steps')
# FLAGS = tf.app.flags.FLAGS

source_filepath = './amazon31.tfrecords'
target_filepath = './dslr31.tfrecords'
batchsize = 32
num_class =31
lr=0.00005
# preprocessing data
source_data,source_labels = read_and_decode(source_filepath,batchsize,is_batch=True)
target_data,target_labels = read_and_decode(target_filepath,batchsize,is_batch=True)
train_data = tf.concat([source_data,target_data],0)
domain_labels = np.vstack([np.tile([1., 0.], [batchsize, 1]),
                                   np.tile([0., 1.], [batchsize, 1])])
test_data, test_label = read_and_decode(target_filepath,batchsize,is_batch=False)
test_data = tf.convert_to_tensor(test_data)
test_label = tf.convert_to_tensor(test_label)

class DANN():
    """

    """
    def __init__(self,image,keep_prob, num_classes, skip_layer,
                 weights_path='DEFAULT',reuse=False):
        self.image = image

        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        self.REUSE = reuse

        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        self._build_model()

    def _build_model(self):

        self.l = tf.placeholder(tf.float32, [])
        self.domain = tf.placeholder(tf.float32, [None, 2])
        # Initialize  alexnet model
        with tf.variable_scope("model", reuse=self.REUSE):
            self.model = AlexNet(self.image, self.KEEP_PROB, self.NUM_CLASSES, self.SKIP_LAYER)

            # fc7 feature in alexnet
            fc7_feature = self.model.dropout7
            # add the fc layer :output 256
            self.final_feature = slim.fully_connected(fc7_feature,256,activation_fn=None)

            # Label predictor
            self.scores = slim.fully_connected(self.final_feature, num_class, activation_fn=None)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores[:batchsize,...],
                                                                     labels=source_labels)

            # MMD loss
            self.mmd_loss = mix_rbf_mmd2(self.final_feature[:batchsize,...],
                                         self.final_feature[batchsize:,...])

            # domain classifier
            with tf.variable_scope('domain_predictor'):
                feat = flip_gradient(self.final_feature, self.l)

                d_fc1 = slim.fully_connected(feat, 1024)
                d_drop1 = slim.dropout(d_fc1, self.KEEP_PROB)

                d_fc2 = slim.fully_connected(d_drop1, 1024)
                d_drop2 = slim.dropout(d_fc2, self.KEEP_PROB)

                self.d_logits = slim.fully_connected(d_drop2, 2,activation_fn=None)

                self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.d_logits,
                                                                           labels=self.domain)

alpha=0.003
# Build the model graph
graph = tf.get_default_graph()
with graph.as_default():
    model = DANN(image=train_data, keep_prob=0.5,num_classes=num_class,skip_layer=['fc7','fc8'])
    learning_rate = tf.placeholder(tf.float32, [])

    pred_loss = tf.reduce_mean(model.pred_loss)
    mmd_loss = model.mmd_loss
    domain_loss = tf.reduce_mean(model.domain_loss)
    total_loss = pred_loss + domain_loss + alpha * mmd_loss
    # tf.summary.scalar("loss", total_loss)
    # summ = tf.summary.merge_all()
    regular_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss)
    #dann_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)
    dann_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

    # Evaluation
    correct_domain_pred = tf.equal(tf.argmax(model.domain, 1), tf.argmax(model.d_logits, 1))
    domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))

num_steps = 12000
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# Load the pretrained weights into the non-trainable layer
model.model.load_initial_weights(sess)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
try:

    for step in range(num_steps):
        if coord.should_stop():
            break

        # Adaptation param and learning rate schedule as described in the paper
        p = float(step) / num_steps
        l = 2. / (1. + np.exp(-10. * p)) - 1
            # lr = 0.01 / (1. + 10 * p) ** 0.75


        _, loss, domain_accuracy = sess.run([dann_train_op, total_loss, domain_acc],
                                                feed_dict={model.domain: domain_labels,
                                                           model.l: l, learning_rate: lr})
        if step % 200 == 0:
            print('step: {:>6}, loss: {:>6}, domain_accuracy: {:.4f}'.format(step, loss, domain_accuracy))

    # Compute final evaluation on test data
    i = 0
    all_sum = 0
    num_test = 498
    while i < num_test:
        j = min(i + 300, num_test)
        image_batch = test_data[i:j, ...]
        labels_batch = test_label[i:j, ...]
        predict_model = DANN(image=image_batch, keep_prob=1.,
                             num_classes=num_class, skip_layer=['fc7', 'fc8'], reuse=True)
        predict_logits = predict_model.scores
        correct = tf.equal(tf.argmax(predict_logits, 1), labels_batch)
        correct_sum = tf.reduce_sum(tf.cast(correct, tf.float32))
        correct_sums = sess.run(correct_sum)
        all_sum += correct_sums

        i = j

    acc = float(all_sum) / num_test
    print('Target  accuracy: {:.4f}'.format(acc))


except tf.errors.OutOfRangeError:
    print('done')
finally:
    coord.request_stop()
coord.join(threads)










