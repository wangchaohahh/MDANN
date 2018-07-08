# -*- coding: utf-8 -*-
# @Time    : 18-6-1 上午10:11
# @Author  : wangchao
# @FileName: test.py
# @Project : tf-dann
# @Software: PyCharm

from __future__ import print_function

#!/usr/bin/env python3
#coding:utf-8
import tensorflow as tf

img = tf.reshape(tf.range(24), [2, 2, 2, 3])

img_channel_swap = img[..., ::-1]

sess = tf.Session()
print(sess.run(img))
print('bGR:', sess.run(img_channel_swap))