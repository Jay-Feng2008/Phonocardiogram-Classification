import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
sys.path.append("../..")
from layers import PositionalEmbedding, MultiHeadSelfAttention, FeedForward

def build_model(d_model=64, num_heads=[64, 32], classes=5, input_shape=(137, 15), batch_size=32):
    inputs = keras.layers.Input(shape=input_shape, batch_size=batch_size)
    x = PositionalEmbedding(d_model=d_model)(inputs)
    for n_heads in num_heads:
        x = MultiHeadSelfAttention(d_model=d_model, num_heads=n_heads)(x)
        x = FeedForward(d_model=d_model)(x)
    x = keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    x = keras.layers.Dense(classes, activation='softmax')(x)
    return keras.Model(inputs, x)

class Model(keras.Model):
    def __init__(self, model, x_shape, eps=8.0, alph=1.0):
        super().__init__()
        self.model = model
        self.x_shape = x_shape
        self.x_rank = len(self.x_shape)
        self.batch_size = x_shape[0]

        self.x_norm_resize_shape = [self.batch_size] + list(tf.ones(self.x_rank, dtype=tf.int32).numpy())[1:]

        self.xi = 1e-6
        self.eps = eps     # the perturbation parameter
        self.alph = alph   # regularization coefficient
        self.lds = lambda y, y_p: tf.math.reduce_sum(keras.losses.kl_divergence(y, y_p))

    def train_step(self, data):
        x, y = data

        x_p = tf.random.normal(self.x_shape)
        x_norm = x_p
        for i in range(self.x_rank-1, 0, -1):
            x_norm = tf.norm(x_norm, ord=2, axis=int(i))
        x_p /= tf.reshape(x_norm, self.x_norm_resize_shape)
        x_p *= self.xi

        with tf.GradientTape() as adversarial_tape:
            adversarial_tape.watch(x_p)
            y_p = self.model(x + x_p, training=True)
            y_hat = self.model(x, training=True)
            l = self.lds(y_hat, y_p)                     # Calculate the local smoothness measure
        g = adversarial_tape.gradient(l, x_p)

        g_norm = g
        for i in range(self.x_rank-1, 0, -1):
            g_norm = tf.norm(g_norm, ord=2, axis=int(i))

        x_p = self.eps * g / tf.reshape(g_norm, self.x_norm_resize_shape)  # set x_p to be eps * normalized_grad

        with tf.GradientTape() as model_tape:
            y_p = self.model(x + x_p, training=True)
            y_hat = self.model(x, training=True)
            l = self.lds(y_hat, y_p)    # Recalculate regularization

            logits = self.model(x, training=True)
            loss = self.compiled_loss(y, logits) + self.alph * l / BATCH_SIZE

        self.optimizer.minimize(loss, self.trainable_variables, tape=model_tape)
        return self.compute_metrics(x, y, logits, None)

    def call(self, x):
        return self.model(x)

# load data
data = np.load("mfcc.npz")
X = data["X"]
Y = data["Y"]

x_train = X[0:700]
y_train = Y[0:700]
x_val = X[700:850]
y_val = Y[700:850]
x_test = X[850:1000]
y_test = Y[850:1000]  # 70/15/15 (train/val/test) split

# Hyperparameter to search
pretraining_epochs_all = [1, 2, 3, 4, 5]
eps_all = [10, 20, 30, 40]

# fixed Hyperparameter
d_model = 64
num_heads = [64, 32]
BATCH_SIZE = 50   # 50 is used so that no traing and testing data will be wasted.
lr = 0.115
warmup_steps = 6000

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr, warmup_steps=4000):
        super().__init__()

        self.warmup_steps = warmup_steps
        self.lr = lr

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return self.lr * tf.math.minimum(arg1, arg2)

def eval_model(d_model, num_heads, classes, input_shape, batch_size, epochs, lr, warmup,
              pretraining_epochs, eps):
    model = build_model(d_model=d_model, num_heads=num_heads, classes=classes, input_shape=input_shape,
                        batch_size=batch_size)

    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    learning_rate = CustomSchedule(lr=lr, warmup_steps=warmup)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
    model.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    # pretraining:
    model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=pretraining_epochs,
              validation_data=(x_val, y_val))
    model.save_weights("pretrained_weights")

    # rebuild model for vat:
    model = build_model(d_model=d_model, num_heads=num_heads, classes=5, input_shape=(137, 15), batch_size=BATCH_SIZE)
    model.load_weights("pretrained_weights")
    model = Model(model, x_shape=(BATCH_SIZE, 137, 15), eps=eps, alph=1.0)

    # VAT
    learning_rate = CustomSchedule(lr=lr, warmup_steps=warmup_steps)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    earlystop_callback = keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                       patience=500,
                                                       verbose=1,
                                                       restore_best_weights=True,
                                                       start_from_epoch=450)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=["accuracy"])
    model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=epochs, validation_data=(x_val, y_val),
         callbacks=[earlystop_callback])

    training_loss = float(loss(y_train, model(x_train)))

    m = keras.metrics.Accuracy()
    m.update_state(y_train, tf.math.argmax(model(x_train), axis=1))
    training_acc = m.result().numpy()
    m.reset_states()

    testing_loss = float(loss(y_val, model(x_val)))

    m.update_state(y_val, tf.math.argmax(model(x_val), axis=1))
    testing_acc = m.result().numpy()
    m.reset_states()

    return {"training_loss": training_loss, "training_acc": training_acc,
            "testing_loss": testing_loss, "testing_acc": testing_acc}


# Start grid search
history = []
for pretraining_epochs in pretraining_epochs_all:
    for eps in eps_all:
        logs = eval_model(d_model=d_model, num_heads=num_heads, classes=5, input_shape=(137, 15),
                          batch_size=BATCH_SIZE, epochs=2000, lr=lr, warmup=warmup_steps,
                          pretraining_epochs=pretraining_epochs, eps=eps)
        history.append({"pretraining_epochs": pretraining_epochs, "eps": eps, "logs": logs})
        np.save("history1.npy", history)

np.save("history1.npy", history)