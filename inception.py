import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions
import tensorflow_hub as hub
import torch
import torchvision.models as models
import numpy as np


def inception_score(images, n_split):
    model = InceptionV3()
    processed = images.astype('float32')




model = InceptionV3()
# model = models.inception_v3(pretrained=True, transform_input=True)



image = preprocess_input(np.zeros((1, 299, 299, 3)))

print(len(model.predict(image)[0]))
print(decode_predictions(model.predict(image)))


module = hub.Module("https://tfhub.dev/google/compare_gan/ssgan_128x128/1")

batch_size = 8
z_dim = 120

# Sample random noise (z) and ImageNet label (y) inputs.
z = tf.random.normal([batch_size, z_dim])  # noise sample
labels = tf.random.uniform([batch_size], maxval=1000, dtype=tf.int32)
inputs = dict(z=z, labels=labels)

samples = module(inputs)
print(samples)
