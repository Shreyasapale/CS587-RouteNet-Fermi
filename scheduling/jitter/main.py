import os
import tensorflow as tf
from data_generator import input_fn

# from jitter_model import RouteNet_Fermi
from jitter_model_BiLSTM import RouteNet_Fermi
# from jitter_model_RNN import RouteNet_Fermi

def denorm_MAPE(y_true, y_pred):
    denorm_y_true = tf.math.exp(y_true)
    denorm_y_pred = tf.math.exp(y_pred)
    mape = tf.abs((denorm_y_pred - denorm_y_true) / denorm_y_true) * 100
    return mape

TRAIN_PATH = '/home/ssapale/Documents/Workspace/DeepLearning/project/routenet_fermi-main/data/scheduling/train'
VALIDATION_PATH = '/home/ssapale/Documents/Workspace/DeepLearning/project/routenet_fermi-main/data/scheduling/test'
TEST_PATH = '/home/ssapale/Documents/Workspace/DeepLearning/project/routenet_fermi-main/data/scheduling/test'

ds_train = input_fn(TRAIN_PATH, shuffle=True)
# ds_train = ds_train.map(lambda x, y: transformation(x, y))
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
ds_train = ds_train.repeat()

ds_validation = input_fn(VALIDATION_PATH, shuffle=False)
# ds_validation = ds_validation.map(lambda x, y: transformation(x, y))
ds_validation = ds_validation.prefetch(tf.data.experimental.AUTOTUNE)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model = RouteNet_Fermi()

# loss_object = tf.keras.losses.MeanSquaredError()
loss_object = tf.keras.losses.MeanAbsolutePercentageError()

model.compile(loss=loss_object,
              optimizer=optimizer,
              run_eagerly=False,)
            #   metrics=[denorm_MAPE])

ckpt_dir = './ckpt_dir_BiLSTM'
# latest = tf.train.latest_checkpoint(ckpt_dir)
latest = None

if latest is not None:
    print("Found a pretrained model, restoring...")
    model.load_weights(latest)
else:
    print("Starting training from scratch...")
    
filepath = os.path.join(ckpt_dir, "{epoch:02d}-{val_loss:.2f}")

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    verbose=1,
    mode="min",
    monitor='val_loss',
    save_best_only=False,
    save_weights_only=True,
    save_freq='epoch')

model.fit(ds_train,
          epochs=25,
          steps_per_epoch=2000,
          validation_data=ds_validation,
          validation_steps=200,
          callbacks=[cp_callback],
          use_multiprocessing=True)

ds_test = input_fn(TEST_PATH, shuffle=False)
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
# ds_test = ds_test.map(lambda x, y: transformation(x, y))
model.evaluate(ds_test)
