from dataset import CatDog
import tensorflow as tf
data = CatDog()




X = tf.placeholder("float")
Y = tf.placeholder("float")
dropout = tf.placeholder("float")
def nn(x_data, drop_out):

    x_img = tf.reshape(x_data, [-1, 50, 50 , 3])

    layer1 = tf.layers.conv2d(x_img, filter=32,kernel_size=[5, 5], activation=tf.nn.relu)

    layer1_max_pool = tf.layers.max_pooling2d(layer1, pool_size=5, strides=(1, 1))

    layer2 = tf.layers.conv2d(layer1_max_pool, filter=64, kernel_size=[5, 5], strides=(1, 1))
    layer2_max_pool = tf.layers.max_pooling2d(layer2, pool_size=5, strides=(1, 1))

    layers_flattern = tf.layers.flatten(layer2_max_pool)

    fully_connected_1 = tf.layers.dense(layers_flattern, units=1024, activation=tf.nn.relu)
    fully_connected_1_drop = tf.layers.dropout(fully_connected_1, rate=drop_out)

    fully_connected = tf.layers.dense(fully_connected_1_drop, units=2, activation=None)
    
    return fully_connected


out = nn(X, dropout)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=out))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(1000):
    for i in range(23000//128):
        image, label = data.next_batch(128, i)
        _, train = sess.run([loss, train_op], feed_dict={X:image, Y:label, dropout:0.8})

        test_images, test_labels = data.get_test()
        acc = sess.run(accuracy, feed_dict={X:test_images, Y:test_labels, dropout:1.})

        print("Epoch ", epoch, " acc : ", acc)

        



