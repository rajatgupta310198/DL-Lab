from dataset import CatDog
import tensorflow as tf
data = CatDog()




X = tf.placeholder("float")
Y = tf.placeholder("float")
dropout = tf.placeholder("float")
def nn(x_data, drop_out):

    x_img = tf.reshape(x_data, [-1, 50, 50 , 3])

    layer1 = tf.layers.conv2d(x_img, filters=32, kernel_size=[3, 3], activation=tf.nn.relu, strides=(1, 1))
    print(layer1)
    layer1_max_pool = tf.layers.max_pooling2d(layer1, pool_size=3, strides=(2, 2))
    print(layer1_max_pool)
    layer2 = tf.layers.conv2d(layer1_max_pool, filters=64, kernel_size=[3, 3], activation=tf.nn.relu, strides=(2, 2))
    layer2_max_pool = tf.layers.max_pooling2d(layer2, pool_size=3, strides=(3, 3))
    print(layer2)
    print(layer2_max_pool)
    layers_flattern = tf.layers.flatten(layer2_max_pool)
    print(layers_flattern)
    fully_connected_1 = tf.layers.dense(layers_flattern, units=576, activation=tf.nn.relu)
    fully_connected_1_drop = tf.nn.dropout(fully_connected_1, keep_prob=drop_out)

    fully_connected_logits = tf.layers.dense(fully_connected_1_drop, units=2, activation=None)
    
    return fully_connected_logits


logits = nn(X, dropout)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))
train_op = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(1000):
    for i in range(4500//128):
        x_image, label = data.train_images, data.train_labels
        loss_, _ = sess.run([loss,train_op], feed_dict={X:x_image, Y:label, dropout:0.5})
        test_image, test_labels = data.test_images, data.test_labels
        print("Epoch :", i, "Acc :", sess.run(accuracy, feed_dict={X:test_image, Y:test_labels, dropout:1.}))

# 1 250 478 853 - 6599