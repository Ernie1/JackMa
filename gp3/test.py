import tensorflow as tf
# a=tf.Variable(tf.truncated_normal([2,3,4,5],stddev=0.1))
# init=tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(a))

with tf.Session() as sess:
    input = tf.Variable(tf.random_normal([1, 5, 5, 5]))
    filter = tf.Variable(tf.ones([3, 3, 5, 2]))
    op = tf.nn.conv2d(input, filter, [1, 1, 1, 1], 'SAME')
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(input))
    print(sess.run(filter))
    print(sess.run(op))