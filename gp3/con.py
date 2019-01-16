import tensorflow as tf
from tf.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
x=tf.placeholder(tf,float32,[None,784])
y_=tf.placeholder(tf,float32,[None,10])
x_image=tf.reshape(x,[-1,28,28,1])
