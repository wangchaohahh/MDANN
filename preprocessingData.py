import os
from scipy.misc import imread, imresize,imshow
import tensorflow as tf

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

def create_record(filePath):
    """
    :param filePath: your dataset path
    :return: tfrecords file
    """
    writer = tf.python_io.TFRecordWriter("./train.tfrecords")

    index = 0
    for datapath in os.listdir(filePath):
        for filename in os.listdir(filePath + '/' + datapath):
            filename = filePath + '/' + datapath + '/' + filename

            img = imread(filename)
            img = imresize(img, (256, 256))
            img_raw = img.tostring()
            if (len(img_raw) == 196608):
                # print (len(img_raw))
                example = tf.train.Example(features=tf.train.Features(
                    feature={"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                             "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}
                ))
            writer.write(example.SerializeToString())
        index += 1
    writer.close()
    print ("create record is finished")



#
def read_and_decode(filename, batch_size, is_batch = True):
    """

    :param filename: tfrecords path
    :param batch_size:
    :param is_batch: train or test
    :return: batch of iamges and  labels
    """
    if(is_batch):
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img = tf.cast(img, tf.float32)
        #print(img.shape)
        img = tf.reshape(img, [256, 256, 3])

        # data argumentation
        image = tf.random_crop(img, [227, 227, 3])# randomly crop the image size to 224 x 224
        image = tf.image.random_flip_left_right(image)
        #image = tf.image.random_flip_up_down(image)
        #image = tf.image.random_brightness(image, max_delta=63)
        #image = tf.image.random_contrast(image,lower=0.2,upper=1.8)

       # img = tf.image.per_image_standardization(image)
        # img = 2*( tf.cast(image, tf.float32) * (1. / 255) - 0.5)
        img = tf.subtract(image, IMAGENET_MEAN )
        # RGB -> BGR, for using pretrained alexnet
        img = img[..., ::-1]
        label = tf.cast(features['label'], tf.int32)

        images, label_batch = tf.train.shuffle_batch(
                                        [img, label],
                                        batch_size = batch_size,
                                        num_threads= 16,
                                        capacity = 5000,
                                        min_after_dequeue = 2000)
        ## ONE-HOT
        n_classes = 10
        label_batch = tf.one_hot(label_batch, depth= n_classes)
        label_batch = tf.cast(label_batch, dtype=tf.int32)
        labels = tf.reshape(label_batch, [batch_size, n_classes])
    else:

        labels = []
        images = []
        for s_example in tf.python_io.tf_record_iterator(filename):
            features = tf.parse_single_example(s_example,
                                               features={
                                              'label':tf.FixedLenFeature([],tf.int64),
                                              'img_raw':tf.FixedLenFeature([],tf.string),
                                               })
            img = tf.decode_raw(features['img_raw'], tf.uint8)
            img = tf.cast(img, tf.float32)
            img = tf.reshape(img, [256,256,3])
            #image = tf.random_crop(img,[227,227,3])
            image = tf.image.resize_images(img, (227,227), method=0)
            img = tf.subtract(image, IMAGENET_MEAN)
            # RGB -> BGR, for using pretrained alexnet
            image = img[..., ::-1]
            images.append(tf.expand_dims(image,0))

            ##labels
            n_classes =10
            label = tf.cast(features['label'], tf.int32)
            label = tf.one_hot(label, depth=n_classes)
            labels.append(tf.expand_dims(label,0))
    return tf.concat(images,0), tf.concat(labels,0)

# ## test
if __name__ == '__main__' :
    #create_record('./dataset/')
    img, labels = read_and_decode('./amazon31.tfrecords',32, is_batch=False)
    # img = tf.convert_to_tensor(img)
    # labels = tf.convert_to_tensor(labels)
    #
    with tf.Session()  as sess:
    ##
       coord = tf.train.Coordinator()
       threads = tf.train.start_queue_runners(sess=sess,coord=coord)
       sess.run(tf.global_variables_initializer())
       try:
           for i in range(1):

               image, label = sess.run([img, labels])

               print(image.shape)
               #imshow(image[0])


       except tf.errors.OutOfRangeError:
           print('done!')
       finally:
           coord.request_stop()
       coord.join(threads)

