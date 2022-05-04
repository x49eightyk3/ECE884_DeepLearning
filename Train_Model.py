import os
import datetime
import tensorflow as tf
import glob
import numpy as np

Conv2D = tf.keras.layers.Conv2D
Input = tf.keras.layers.Input
LeakyReLU = tf.keras.layers.LeakyReLU
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
Reshape = tf.keras.layers.Reshape
Conv2DTranspose = tf.keras.layers.Conv2DTranspose
Activation = tf.keras.layers.Activation
BatchNormalization = tf.keras.layers.BatchNormalization
Model = tf.keras.Model
Sequential = tf.keras.Sequential

tf.random.set_seed(999)
np.random.seed(999)

## Prepare Input features
def tf_record_parser(record):
    keys_to_features = {
        "noise_stft_phase": tf.io.FixedLenFeature((), tf.string, default_value=""),
        'noise_stft_mag_features': tf.io.FixedLenFeature([], tf.string),
        "clean_stft_magnitude": tf.io.FixedLenFeature((), tf.string)
    }

    features = tf.io.parse_single_example(record, keys_to_features)

    noise_stft_mag_features = tf.io.decode_raw(features['noise_stft_mag_features'], tf.float32)
    clean_stft_magnitude = tf.io.decode_raw(features['clean_stft_magnitude'], tf.float32)
    noise_stft_phase = tf.io.decode_raw(features['noise_stft_phase'], tf.float32)

    # reshape input and annotation images
    noise_stft_mag_features = tf.reshape(noise_stft_mag_features, (129, 8, 1), name="noise_stft_mag_features")
    clean_stft_magnitude = tf.reshape(clean_stft_magnitude, (129, 1, 1), name="clean_stft_magnitude")
    noise_stft_phase = tf.reshape(noise_stft_phase, (129,), name="noise_stft_phase")

    return noise_stft_mag_features, clean_stft_magnitude

def build_model(l2_strength):
    inputs = Input(shape=[numFeatures, numSegments, 1])
    x = inputs

    # 1----
    x = tf.keras.layers.ZeroPadding2D(((4, 4), (0, 0)))(x)
    x = Conv2D(filters=18, kernel_size=[9, 8], strides=[1, 1], padding='valid', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    skip0 = Conv2D(filters=30, kernel_size=[5, 1], strides=[1, 1], padding='same', use_bias=False,
                   kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(skip0)
    x = BatchNormalization()(x)

    x = Conv2D(filters=8, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # 2----
    x = Conv2D(filters=18, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    skip1 = Conv2D(filters=30, kernel_size=[5, 1], strides=[1, 1], padding='same', use_bias=False,
                   kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(skip1)
    x = BatchNormalization()(x)

    x = Conv2D(filters=8, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # 3---
    x = Conv2D(filters=18, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=30, kernel_size=[5, 1], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=8, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # 4---
    x = Conv2D(filters=18, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=30, kernel_size=[5, 1], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = x + skip1
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=8, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # 5---
    x = Conv2D(filters=18, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=30, kernel_size=[5, 1], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = x + skip0
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=8, kernel_size=[9, 1], strides=[1, 1], padding='same', use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # 6---
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    x = Conv2D(filters=1, kernel_size=[129, 1], strides=[1, 1], padding='same')(x)

    model = Model(inputs=inputs, outputs=x)

    optimizer = tf.keras.optimizers.Adam(3e-4)

    model.compile(optimizer=optimizer, loss='mse',
                  metrics=[tf.keras.metrics.RootMeanSquaredError('rmse')])
    return model



path_to_dataset = "C:/Users/TruMoone/PycharmProjects/Records"
mozilla_basepath = 'C:\\Users\\TruMoone\\PycharmProjects\\CommonVoice\\Datasets\\en\\'
UrbanSound8K_basepath = 'C:\\Users\\TruMoone\\PycharmProjects\\UrbanSound8K\\'

# get training and validation tf record file names
train_tfrecords_filenames = glob.glob(os.path.join(path_to_dataset, 'train_*'))
val_tfrecords_filenames = glob.glob(os.path.join(path_to_dataset, 'val_*'))

# shuffle the file names for training
np.random.shuffle(train_tfrecords_filenames)
print("Training file names: ", train_tfrecords_filenames)
print("Validation file names: ", val_tfrecords_filenames)

windowLength = 256
overlap = round(0.25 * windowLength)  # overlap of 75%
ffTLength = windowLength
inputFs = 48e3
fs = 16e3
numFeatures = ffTLength // 2 + 1
numSegments = 8
print("windowLength:", windowLength)
print("overlap:", overlap)
print("ffTLength:", ffTLength)
print("inputFs:", inputFs)
print("fs:", fs)
print("numFeatures:", numFeatures)
print("numSegments:", numSegments)



## Create tf.Data.Dataset
train_dataset = tf.data.TFRecordDataset([train_tfrecords_filenames])
train_dataset = train_dataset.map(tf_record_parser)
train_dataset = train_dataset.shuffle(8192)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.batch(512)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

test_dataset = tf.data.TFRecordDataset([val_tfrecords_filenames])
test_dataset = test_dataset.map(tf_record_parser)
test_dataset = test_dataset.repeat(1)
test_dataset = test_dataset.batch(512)




model = build_model(l2_strength=0.0)
model.summary()

baseline_val_loss = model.evaluate(test_dataset)[0]
print(f"Baseline accuracy {baseline_val_loss}")

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True,
                                                           baseline=None)
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, update_freq='batch')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./denoiser_cnn_log_mel_generator.h5',
                                                         monitor='val_loss', save_best_only=True)
model.fit(train_dataset,
          steps_per_epoch=600,  # you might need to change this
          validation_data=test_dataset,
          epochs=400,
          callbacks=[early_stopping_callback, tensorboard_callback, checkpoint_callback]
          )

val_loss = model.evaluate(test_dataset)[0]

if val_loss < baseline_val_loss:
    print("New model saved.")
    model.save('C:/Users/TruMoone/PycharmProjects/CNNModel/model3.h5')

