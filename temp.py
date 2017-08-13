import tensorflow as tf
import tflearn.datasets.oxflower17 as oxflower17
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import os

dataset_X,dataset_y=oxflower17.load_data(one_hot=True,resize_pics=(227,227))
dataset_X-=dataset_X.min()
dataset_X/=dataset_X.max()
X_train,X_test,y_train,y_test=train_test_split(dataset_X,dataset_y,
                                               test_size=0.1, random_state=30)
labels_train=LabelBinarizer().fit_transform(y_train)
labels_test=LabelBinarizer().fit_transform(y_test)

def convlayers(inputs,title,kernel_size,input_level,output_level,step):
    with tf.name_scope(title) as scope:
        kernel = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, 
                                input_level, output_level], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(inputs, kernel, [1, step, step, 1], padding='SAME')
        biases = tf.Variable(tf.truncated_normal([output_level], dtype=tf.float32),
                             trainable=True, name='biases')
        conv_pred = tf.nn.bias_add(conv, biases)
        conv_o = tf.nn.relu(conv_pred, name=scope)
        return conv_o,kernel,biases
def pool(X,name):
    with tf.name_scope(name):
        pool_o= tf.nn.max_pool(X,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],
                         padding='VALID',name=name)
        return pool_o
def connlayer(x, inputD, outputD, activation, name):  
    """fully-connect"""  
    with tf.variable_scope(name) as scope:  
        w = tf.get_variable("w", shape = [inputD, outputD], dtype = "float")  
        b = tf.get_variable("b", [outputD], dtype = "float")  
        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)  
        if activation=="relu":  
            return tf.nn.relu(out)  
        if activation=="softmax":
            return tf.nn.softmax(out)
        else:  
            return out  
def vgg19(X,y):
    #conv conv pool  64 64
    #conv conv pool  128 128
    #conv conv conv pool  256 256
    #conv conv conv pool  512 512
    #conv conv conv pool  512 512
    #4096 4096 17
    conv1,kernel,biases=convlayers(X,"conv_1",3,3,64,1)
    conv2,kernel,biases=convlayers(conv1,"conv_2",3,64,64,1)
    pool1=pool(conv2,"pool_1")
    conv3,kernel,biases=convlayers(pool1,"conv_3",3,64,128,1)
    conv4,kernel,biases=convlayers(conv3,"conv_4",3,128,128,1)
    pool2=pool(conv4,"pool_1")
    conv5,kernel,biases=convlayers(pool2,"conv_5",3,128,256,1)
    conv6,kernel,biases=convlayers(conv5,"conv_6",3,256,256,1)
    conv7,kernel,biases=convlayers(conv6,"conv_7",3,256,256,1)
    conv8,kernel,biases=convlayers(conv7,"conv_8",3,256,256,1)
    pool3=pool(conv8,"pool_1")
    conv9,kernel,biases=convlayers(pool3,"conv_9",3,256,512,1)
    conv10,kernel,biases=convlayers(conv9,"conv_10",3,512,512,1)
    conv11,kernel,biases=convlayers(conv10,"conv_11",3,512,512,1)
    conv12,kernel,biases=convlayers(conv11,"conv_12",3,512,512,1)
    pool4=pool(conv12,"pool_1")
    conv13,kernel,biases=convlayers(pool4,"conv_13",3,512,512,1)
    conv14,kernel,biases=convlayers(conv13,"conv_14",3,512,512,1)
    conv15,kernel,biases=convlayers(conv14,"conv_15",3,512,512,1)
    conv16,kernel,biases=convlayers(conv15,"conv_16",3,512,512,1)
    pool5=pool(conv16,"pool_1")
    size=7*7*512
    conv_pred = tf.reshape(pool5, [-1,size]) 
    fc1_pred = connlayer(conv_pred, size, 4096, "relu", "fc1")
    fc1=tf.nn.dropout(fc1_pred, 0.5)  
    fc2_pred = connlayer(fc1, 4096, 4096, "relu", "fc2")
    fc2=tf.nn.dropout(fc2_pred, 0.5)  
    y_pred = connlayer(fc2, 4096, 17, "softmax", "fc3")
    
    return y_pred

X = tf.placeholder(tf.float32, [None, 227, 227, 3])  
y = tf.placeholder(tf.float32,[None,17])   
y_pred=vgg19(X,y)
loss=tf.reduce_mean(tf.reduce_sum(tf.square(y - y_pred),reduction_indices=[1]))
tf.summary.scalar("loss",loss)
train_op=tf.train.AdamOptimizer().minimize(loss)
correct_pred=tf.equal(tf.argmax(y,1),tf.argmax(y_pred,1))
acc_op=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

global_step = tf.Variable(0, name='global_step', trainable=False)
saver = tf.train.Saver()
non_storable_variable = tf.Variable(777)
ckpt_dir = './ckpt_dir'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

with tf.Session() as sess:  
    writer=tf.summary.FileWriter("./logs",sess.graph)
    tf.global_variables_initializer().run()
    merged = tf.summary.merge_all()
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print('Restoring from checkpoint: %s' % ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    start = global_step.eval()
    for epoch in range(10):
        for i in range(0, len(X_train), 64):
            X_batch=X_train[i: i + 64]
            y_batch=labels_train[i: i + 64]
            for j in range(64):
                feed_dict = {X: X_batch[j:j+2],y: y_batch[j:j+2]}
                j=j+1
                _,loss_train = sess.run([train_op,loss], feed_dict=feed_dict)
                print('Epoch: %04d, loss=%.9f' % (epoch + 1, loss_train))
                summary, accuracy = sess.run([merged, acc_op],
                                             feed_dict={X: X_test, y: labels_test})
                print('Accuracy on validation set: %.9f' % accuracy)
        writer.add_summary(summary, epoch)
        global_step.assign(epoch).eval()
        saver.save(sess, ckpt_dir + '/logistic.ckpt',global_step=global_step)
print('Training complete!')

        
        
        
        
        
        
        
        
        
            