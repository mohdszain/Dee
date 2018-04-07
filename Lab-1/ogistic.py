import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Step 1: read in data from the file
text = np.loadtxt('Samplestck.txt', skiprows=1)
book = text[:,:2]
sheet = text[:,2:] #Creating matrix
# Step 2: create placeholders for input a and label z
a = tf.placeholder(tf.float32, [None, 2])
z = tf.placeholder(tf.float32, [None, 1])

# Step 3: create weight and bias, initialized to 0
w = tf.Variable(tf.ones([2, 1]))
b = tf.Variable([1.0])

# Step 4: build model to predict Y
logits = a * w + b

# Step 5: use the square error as the loss function
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=z, logits=(a * w + b))
loss = tf.reduce_mean(loss)

# Step 6:Calculating the Accuracy
predictoper  = tf.greater_equal(logits, tf.zeros_like(logits))
correctoper  = tf.equal(tf.cast(predictoper, tf.float32), z)
accuracyoper = tf.reduce_mean(tf.cast(correctoper, tf.float32))
with tf.Session() as sess:
# Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./graphs/log_reg', sess.graph)
learng_rt = 0.01
epochs= 100
optimizer = tf.train.GradientDescentOptimizer(learng_rt)
train_op  = optimizer.minimize(loss)

#using session run
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Using random number generator
np.random.seed(0)

for epoch in range(epochs):
    # each point is  once present in random order
    index = np.random.permutation(text.shape[0])
    for i in index:
        feed_dictionary = {a: book[i:i+1], z: sheet[i:i+1]}
        sess.run(train_op, feed_dictionary)

    if (epoch+1) % 10 == 0:
        feed_dictionary = {a: book, z: sheet}
        accy = sess.run(accuracyoper, feed_dictionary)
        print("After {} epochs, accuracy = {}".format(epoch+1, accy))

# Print the result
Weigh_val, bais_val = sess.run([w, b])
Weigh_val = Weigh_val[:,0]
bais_val = bais_val[0]
print("w =", Weigh_val)
print("b =", bais_val)

def predict(a_):
    return 1 * sess.run(predictoper, {a: a_})

#predictions Models
labels = predict(book)[:,0]

#indices for two species
index_0, = np.where(labels == 0)
index_1, = np.where(labels == 1)

# Plotting
plt.plot(book[index_0,0], book[index_0,1], 'bo', label='I. versicolor')
plt.plot(book[index_1,0], book[index_1,1], 'ro', label='I. virginica')

# Plotting the hyperplane
x_sep = np.linspace(book[:,0].min(), book[:,0].max())
y_sep = (-bais_val - Weigh_val[0]*x_sep) / Weigh_val[1]
plt.plot(x_sep, y_sep, 'm', label="Decision boundary")

plt.legend()

plt.xlabel("Sepal length (cm)")
plt.ylabel("Petal legnth (cm)")

plt.show()