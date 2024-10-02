import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
sys.path.append("../../..")
from layers import PositionalEmbedding, MultiHeadSelfAttention, FeedForward
import datetime

def build_model(d_model=64, num_heads=[64, 32], classes=5, input_shape=(137, 15), batch_size=32):
    inputs = keras.layers.Input(shape=input_shape, batch_size=batch_size)
    x = PositionalEmbedding(d_model=d_model)(inputs)
    for n_heads in num_heads:
        x = MultiHeadSelfAttention(d_model=d_model, num_heads=n_heads)(x)
        x = FeedForward(d_model=d_model)(x)
    x = keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    x = keras.layers.Dense(classes, activation='softmax')(x)
    return keras.Model(inputs, x)

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

# Hyperparameters
d_model = 64
num_heads = [64, 32]
BATCH_SIZE = 50   # 50 is used so that no traing and testing data will be wasted.
lrc = 0.115
warmup_steps = 6000

# Build model
model = build_model(d_model=d_model, num_heads=num_heads, classes=5, input_shape=(137, 15), batch_size=BATCH_SIZE)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lrc, warmup_steps=4000):
        super().__init__()

        self.warmup_steps = warmup_steps
        self.lrc = lrc

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return self.lrc * tf.math.minimum(arg1, arg2)
    

learning_rate = CustomSchedule(lrc=lrc, warmup_steps=warmup_steps)
optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=["accuracy"])
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_path = "training_1/cp.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 monitor = "val_accuracy",
                                                 save_best_only=True,
                                                 save_weights_only=True,
                                                 verbose=1)

# Training
model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=300, validation_data=(x_val, y_val))

model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=1200, validation_data=(x_val, y_val),
         callbacks=[tensorboard_callback, cp_callback])