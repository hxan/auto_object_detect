from PIL import Image
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import cv2
import sys

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1) 
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def train():
	mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
	
	x = tf.placeholder(tf.float32, shape=[None, 784]) 
	y_ = tf.placeholder(tf.float32, shape=[None, 10]) 
	
	x_image = tf.reshape(x, [-1,28,28,1]) 
	W_conv1 = weight_variable([5, 5, 1, 32]) 
	b_conv1 = bias_variable([32])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) 
	h_pool1 = max_pool_2x2(h_conv1)
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	
	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	keep_prob = tf.placeholder("float")
	
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])
	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
	
	cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))  
	
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	
	saver = tf.train.Saver() 
	
	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	
	    for i in range(500):
	      batch = mnist.train.next_batch(50)
	      if i%100 == 0:       
	        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
	        print("step %d, training accuracy %g"%(i, train_accuracy))
	
	      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}) 
	
	    saver.save(sess, './save/model.ckpt') 
	
	    print("final accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images [0:2000], y_: mnist.test.labels [0:2000], keep_prob: 1.0}))

def recognition():
	im = Image.open('./number_result.png')
	result = [ (255-x)*1.0/255.0  for x in list( im.getdata() ) ] 
	
	x = tf.placeholder("float", shape=[None, 784]) 

	x_image = tf.reshape(x, [-1,28,28,1]) 
	
	W_conv1 = weight_variable([5, 5, 1, 32]) 
	b_conv1 = bias_variable([32])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) 
	h_pool1 = max_pool_2x2(h_conv1)
	
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	
	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	
	keep_prob = tf.placeholder("float")
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])
	
	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
	
	saver = tf.train.Saver() 
	
	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    saver.restore(sess, "./save/model.ckpt")
	
	    prediction = tf.argmax(y_conv,1)
	    predint = prediction.eval(feed_dict={x: [result],keep_prob: 1.0}, session=sess)
	
	    print("recognize result: %d" %predint[0])

if len(sys.argv) > 1:
    if sys.argv[1]=='t':
        train()
        exit()
        
is_press=0    
src=cv2.imread('./white.png')
cv2.imshow('src',src)

def onMouse(event,x,y,flags,param):
    global is_press
    global src
    
    if event == 1:
        is_press=1
    elif event==3:
        img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) 
        ret,bin=cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        bin=cv2.resize(bin, (28,28))  
		  
        cv2.imwrite('./number_result.png',bin)
        
        recognition()
        cv2.waitKey(10000)
        exit()
    elif event == 4:
        is_press=0
    elif event == 0 and is_press == 1:
        cv2.circle(src,(x,y),15,(0,0,0),-1)
        cv2.imshow('src',src)
        
cv2.setMouseCallback('src',onMouse)
cv2.waitKey(0)
