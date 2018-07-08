from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from preprocessingData import *
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1

#  Network flags
tf.app.flags.DEFINE_integer('batchsize',32, 'batchsize')
tf.app.flags.DEFINE_string('filepath','./train.tfrecords', 'path of dataset')
tf.app.flags.DEFINE_integer('num_class',2, 'number of classes')
tf.app.flags.DEFINE_integer('num_steps',5000, 'number of train steps')
FLAGS = tf.app.flags.FLAGS

# preprocessing data
train_data,train_labels = read_and_decode(FLAGS.filepath,FLAGS.batchsize,is_batch=True)
test_data,test_labels = read_and_decode(FLAGS.filepath,FLAGS.batchsize,is_batch=False)

# our base model inference
def resnet_model(image,reuse,is_training):
    """
    our base model is resnet50.
    :param image:
    :param reuse: use True when test
    :param is_training: control batchnorm in restnet, and set True for training and test phase
    :return: logits of batch images
    """
    with tf.variable_scope("model",reuse=reuse):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            logits, endpoints = resnet_v1.resnet_v1_50(image,is_training=is_training)
            outputs = slim.flatten(logits)
            logits = slim.fully_connected(outputs, FLAGS.num_class, activation_fn=None)
    return logits,endpoints

# Build the model graph
graph = tf.get_default_graph()
with graph.as_default():
    # loss function for imbalanced class
    logits,endpoints = resnet_model(train_data,reuse=False,is_training=True)
    ratio = 60.0 / (60.0 + 120.0)
    class_weight = tf.constant([ratio, 1.0 - ratio])
    weighted_logits = tf.multiply(logits, class_weight) # shape [batch_size, 2]
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = weighted_logits,labels = train_labels))
    tf.summary.scalar("loss", loss)

    # train accuracy
    train_correct_label_pred = tf.equal(tf.argmax(logits,1), tf.argmax(train_labels, 1))
    train_label_acc = tf.reduce_mean(tf.cast(train_correct_label_pred, tf.float32))
    tf.summary.scalar("acc", train_label_acc)

    train_var = tf.trainable_variables()[60:]
    global_step = tf.Variable(0, trainable=False)
    opt = tf.train.MomentumOptimizer(0.0003, 0.9)
    train = slim.learning.create_train_op(loss, opt, global_step=global_step, variables_to_train=train_var)

    # Merges all summaries collected in the default graph
    summ = tf.summary.merge_all()

    # Evaluation
    # test_logits,test_endpoints = resnet_model(test_data, reuse=True, is_training=True)
    # correct_label_pred = tf.equal(tf.argmax(test_logits, 1), test_labels)
    # label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))


def train_and_evaluate(graph, num_steps):
    """Helper to run the model."""

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()

        # tensorboard
        writer = tf.summary.FileWriter('./checkpoints')
        writer.add_graph(sess.graph)

        # restore weights for fine-tune
        def name_in_checkpoint(var):
            if "model" in var.op.name:
                return var.op.name.replace("model/", "")

        variables_to_restore = slim.get_model_variables()
        variables_to_restore = {name_in_checkpoint(var): var for var in variables_to_restore if 'resnet' in var.op.name}
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, "../data/resnet_v1_50.ckpt")

        # initialize the saver
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Training loop
        try:
            for step in range(num_steps):
                if coord.should_stop():
                    break
                _,batch_loss,acc= sess.run([train, loss,train_label_acc])
                if (step+1)%50 == 0:
                    msg = 'Step:{}'.format(step + 1) + ' >>>  ' + 'loss:{0:.5}'.format(batch_loss) \
                          + ' >>>  ' + 'acc:{0:.4}'.format(acc)

                    print(msg)
                    saver.save(sess, './checkpoint/',global_step=step)
        except tf.errors.OutOfRangeError:
            print('done')
        finally:
            coord.request_stop()
        coord.join(threads)

def main(_):
    train_and_evaluate(graph,FLAGS.num_steps)

if __name__ == "__main__":
     tf.app.run()
