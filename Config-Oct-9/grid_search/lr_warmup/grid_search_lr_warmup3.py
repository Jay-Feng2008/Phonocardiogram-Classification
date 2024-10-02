import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
sys.path.append("../..")
from layers import PositionalEmbedding, MultiHeadSelfAttention, FeedForward

# Hyperparameter to search
warmup_all = [4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000]
lrc_all = [0.125, 0.1166, 0.1083, 0.1]  # range defined by result from "grid_search_lr_warmup2.py"
epochs = [int(i/4000 * 1000) for i in warmup_all]
earlyStoppingStartEpoch = [int(i/4000 * 300) for i in warmup_all]

# fixed Hyperparameter
d_model = 64
num_heads = [64, 32]
BATCH_SIZE = 50

# load data
data = np.load("mfcc.npz")
X = data["X"][0:850]
Y = data["Y"][0:850]

x_train = X[0:700]
y_train = Y[0:700]
x_test = X[700:]
y_test = Y[700:]   # 70/15/15 split

def build_model(d_model=64, num_heads=[64, 32], classes=5, input_shape=(137, 15), batch_size=32):
    inputs = keras.layers.Input(shape=input_shape, batch_size=batch_size)
    x = PositionalEmbedding(d_model=d_model)(inputs)
    for n_heads in num_heads:
        x = MultiHeadSelfAttention(d_model=d_model, num_heads=n_heads)(x)
        x = FeedForward(d_model=d_model)(x)
    x = keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    x = keras.layers.Dense(classes, activation='softmax')(x)
    return keras.Model(inputs, x)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lrc, warmup_steps=4000):
        super().__init__()

        self.lrc = lrc
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return self.lrc * tf.math.minimum(arg1, arg2)

def eval_model(d_model, num_heads, classes, input_shape, batch_size, epochs, lrc, warmup, early_stopping_start):
    model = build_model(d_model=d_model, num_heads=num_heads, classes=classes, input_shape=input_shape,
                        batch_size=batch_size)
    
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    learning_rate = CustomSchedule(lrc=lrc, warmup_steps=warmup)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
    model.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    earlystop_callback = keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                       patience=300,
                                                       verbose=1,
                                                       restore_best_weights=True,
                                                       start_from_epoch=early_stopping_start)
    model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=epochs, validation_data=(x_test, y_test),
         callbacks=[earlystop_callback])

    training_loss = float(loss(y_train, model(x_train)))

    m = keras.metrics.Accuracy()
    m.update_state(y_train, tf.math.argmax(model(x_train), axis=1))
    training_acc = m.result().numpy()
    m.reset_states()

    testing_loss = float(loss(y_test, model(x_test)))

    m.update_state(y_test, tf.math.argmax(model(x_test), axis=1))
    testing_acc = m.result().numpy()
    m.reset_states()

    return {"training_loss": training_loss, "training_acc": training_acc,
            "testing_loss": testing_loss, "testing_acc": testing_acc}


# Start grid search
history = []
for i in range(len(warmup_all)):
    for lrc in lrc_all:
        logs = eval_model(d_model=d_model, num_heads=num_heads, classes=5, input_shape=(137, 15),
                          batch_size=BATCH_SIZE, epochs=epochs[i], lrc=lrc, warmup=warmup_all[i],
                          early_stopping_start=earlyStoppingStartEpoch[i])
        history.append({"lrc": lrc, "warmup": warmup_all[i], "logs": logs})
        np.save("history2.npy", history)

np.save("history2.npy", history)