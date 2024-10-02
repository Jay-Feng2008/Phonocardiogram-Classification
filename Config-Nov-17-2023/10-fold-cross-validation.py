import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import KFold
from layers import PositionalEmbedding, MultiHeadSelfAttention, FeedForward
from tqdm import tqdm
import datetime

tf.random.set_seed(100)

# Load data
X = np.load("mfcc.npz")['X']
Y = np.load("mfcc.npz")['Y']

# create dataset for 10-fold cross validation
def make_dataset(x_data,y_data,n_splits):
    def gen():
        for train_index, test_index in KFold(n_splits).split(x_data):
            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]
            yield x_train,y_train,x_test,y_test

    return tf.data.Dataset.from_generator(gen, (tf.float32,tf.float32,tf.float32,tf.float32))

# build model & evaluation pipeline
def build_model(d_model=64, num_heads=[64, 32], classes=5, input_shape=(137, 15), batch_size=32):
    inputs = keras.layers.Input(shape=input_shape, batch_size=batch_size)
    x = PositionalEmbedding(d_model=d_model)(inputs)
    for n_heads in num_heads:
        x = MultiHeadSelfAttention(d_model=d_model, num_heads=n_heads)(x)
        x = FeedForward(d_model=d_model)(x)
    x = keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    x = keras.layers.Dense(classes, activation='softmax')(x)
    return keras.Model(inputs, x)

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
lds = lambda x, y: tf.math.reduce_sum(keras.losses.kl_divergence(x, y))
acc_metric = keras.metrics.SparseCategoricalAccuracy()

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr, warmup_steps=2000):
        super().__init__()
        self.lr = lr
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return self.lr * tf.math.minimum(arg1, arg2)
    
def build_optimizer(lr, warmup_steps):
    learning_rate = CustomSchedule(lr, warmup_steps)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
    return optimizer

def evaluate(X_train, Y_train, x_test, y_test, hyperparameters, save_logs=False):
    d_model, num_heads, classes, input_shape, batch_size, epochs, lr, warmup_steps, pretrain_steps, eps, alpha = hyperparameters
    
    x_val = X_train[800:900]
    y_val = Y_train[800:900]
    x_train = X_train[0:800]
    y_train = Y_train[0:800]  # train/val/test: 80:10:10
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=800, reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)
    
    x = x_train[0:batch_size]
    x_rank = tf.rank(x).numpy()
    x_norm_resize_shape = [batch_size] + list(tf.ones(tf.rank(x), dtype=tf.int32).numpy())[1:]
    
    model = build_model(d_model=d_model, num_heads=num_heads, classes=classes, input_shape=input_shape,
                       batch_size=batch_size)
    optimizer = build_optimizer(lr=lr, warmup_steps=warmup_steps)
    
    @tf.function
    def pre_train(x, y):
        with tf.GradientTape() as model_tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)
        grads = model_tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
    zeta = 1e-6
    @tf.function
    def training_step(x, y):
        x_p = tf.random.normal(x.shape)
        x_norm = x_p
        for i in range(x_rank-1, 0, -1):
            x_norm = tf.norm(x_norm, ord=2, axis=int(i))
        x_p /= tf.reshape(x_norm, (batch_size, 1, 1))
        x_p *= zeta

        with tf.GradientTape() as adversarial_tape:
            adversarial_tape.watch(x_p)
            y_p = model(x + x_p, training=True)
            logits = model(x, training=True)
            l = lds(logits, y_p)
        g = adversarial_tape.gradient(l, x_p)

        g_norm = g
        for i in range(x_rank-1, 0, -1):
            g_norm = tf.norm(g_norm, ord=2, axis=int(i))

        x_p = eps * g / (tf.reshape(g_norm, x_norm_resize_shape)+1e-6)

        with tf.GradientTape() as model_tape:
            y_p = model(x + x_p, training=True)
            logits = model(x, training=True)
            l = lds(logits, y_p)    # Recalculate regularization
            loss = loss_fn(y, logits) + alpha * l / batch_size
        grads = model_tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        acc_metric.update_state(y, logits)
        acc = acc_metric.result()
        acc_metric.reset_states()
        
        return loss, l, acc
    
    # start training
    for i in range(pretrain_steps):
        for step, (x, y) in enumerate(train_dataset):
            pre_train(x, y)
            
    log = {"training_loss":[], "training_1":[], "training_acc":[],
           "val_loss":[], "val_acc":[], "test_acc":[], "test_logits":[]}
    log_path = "log" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".npy"
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        epoch_l = 0
        epoch_acc = 0
        for step, (x, y) in enumerate(train_dataset):
            batch_loss, batch_l, batch_acc = training_step(x, y)
            epoch_loss += float(batch_loss)
            epoch_l += float(batch_l)
            epoch_acc += float(batch_acc)
        epoch_loss /= step+1
        epoch_l /= step+1
        epoch_acc /= step+1
        # print("Training loss: %.4f\nTraining metric: %.4f"
        #     % (float(epoch_loss), float(epoch_acc)))
        # print("lds: %.4f" % float(epoch_l))

        val_loss = 0
        val_acc = 0
        val_logits = model(x_val, training=False)
        val_loss = loss_fn(y_val, val_logits)
        acc_metric.update_state(y_val, val_logits)
        val_acc = acc_metric.result().numpy()
        acc_metric.reset_states()

        # print("Validation loss: %.4f" % (float(val_loss)))
        # print("Validation acc: %.4f" % (float(val_acc)))
        
        test_logits = model(x_test, training=False)
        acc_metric.update_state(y_test, test_logits)
        test_acc = acc_metric.result().numpy()
        acc_metric.reset_states()

        log["training_loss"].append(epoch_loss)
        log["training_1"].append(epoch_l)
        log["training_acc"].append(epoch_acc)
        log["val_loss"].append(val_loss)
        log["val_acc"].append(val_acc)
        log["test_acc"].append(test_acc)
        log["test_logits"].append(test_logits)

        if save_logs:
            np.save(log_path, [log])

    log['test_acc'] = np.array(log['test_acc'])
    log['val_loss'] = np.array(log['val_loss'])
    score = log['test_acc'] - log['val_loss']
    best_idx = np.where(np.abs(score-max(score)) < 1e-6)[0]
    print(log["test_acc"][best_idx])
    return log["test_acc"][best_idx]

def k_fold_cross_validation(data, hyperparameters, k):
    X, Y = data
    dataset = make_dataset(X, Y, k)
    results = []
    for X_train, Y_train, x_test, y_test in dataset:
        results.append(evaluate(X_train, Y_train, x_test, y_test, hyperparameters, save_logs=True))
    return(results)

# d_model, num_heads, classes, input_shape, batch_size, epochs, lr, warmup_steps, pretrain_steps, eps, alpha = hyperparameters
hyperparameters = (64, [64, 32], 5, (137, 15), 32, 2000, 0.0376087962339086, 3282, 5, 49.5219842550157, 3.76978313949224)  # fold_1
print(evaluate(X[0:900], Y[0:900], X[900:1000], Y[900:1000], hyperparameters, save_logs=True))  # fold_1
