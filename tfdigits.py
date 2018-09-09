
# coding: utf-8

# # Build a TensorFlow classifier for recognizing handwritten digits

# 1. Import MNIST Data using TensorFlow

# In[15]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[16]:


mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)


# 2. Check type of Dataset

# In[17]:


type(mnist)
#tensorflow.contrib.learn.python.learn.datasets.base.Datasets


# 3. Array of Training images

# In[18]:


mnist.train.images
#4. size of training data
mnist.train.num_examples


# 5. Visualize the Data

# In[19]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
mnist.train.images[1].shape


# In[20]:


plt.imshow(mnist.train.images[15].reshape(28,28))


# 6. Maximum and minimum value of the pixels in the image

# In[21]:


mnist.train.images[1].max()


# 7. Create the Model

# In[22]:


x = tf.placeholder(tf.float32,shape = [None,784])
#24*24 = 784 pixel images


# In[23]:


W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


# 8. Create the Graph

# In[24]:


y = tf.matmul(x,W) +b
#y = Wi*xi+b
y_true = tf.placeholder(tf.float32,[None,10])


# In[25]:


#Cross Entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)


# In[26]:


train = optimizer.minimize(cross_entropy)


# 8. Create the Session

# In[27]:


init = tf.global_variables_initializer()


# In[28]:


with tf.Session() as sess:
    sess.run(init)
    #Train the model for 1000 steps on the training set using built in batch feeder from mnist
    for  step in range(1000):
        batch_x,batch_y=mnist.train.next_batch(1000)
        sess.run(train, feed_dict = {x:batch_x,y_true:batch_y})
#9. Evaluate the Trained model on Test Data
    #Test the trained model
    matches = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))
    acc = tf.reduce_mean(tf.cast(matches,tf.float32))
    print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))

