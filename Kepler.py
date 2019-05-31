import tensorflow as tf
import matplotlib.pyplot as plt
import pdb
import numpy as np
import pandas as pd
from tensorflow.keras import layers

sample_num = 500000
# coeff = tf.cast(4*np.pi*np.pi/(6.673*10**-11), dtype = tf.float32)
coeff = tf.cast(1, dtype = tf.float32)
#try the range of 10**5 ~10**7 for both mass and radius
# radius = tf.random.normal(shape = [sample_num,1], mean = 0, dtype = tf.float32)
# massinv = tf.random.normal(shape = [sample_num,1], mean = 0, dtype = tf.float32)
radius = tf.random.truncated_normal(shape = [sample_num,1], mean = 2, stddev = 0.5, dtype = tf.float32)
massinv = tf.random.truncated_normal(shape = [sample_num,1], mean = 2, stddev = 0.5, dtype = tf.float32)
period = radius ** 3 * massinv * coeff

def normalize(data):
  if isinstance(data, tf.Tensor):
    data = data.numpy()
  data = (data - np.mean(data)) / np.std(data)
  return tf.cast(data, dtype = tf.float64)

def denorm(data, denorm_factor):
  # denorm_factor is a tuple of (mean, std)
  return data * denorm_factor[1] + denorm_factor[0]

data = tf.stack([radius, massinv], axis = 1)
data = tf.squeeze(data)
normed_label = normalize(period)
denorm_factor = (np.mean(period.numpy()), np.std(period.numpy()))


def build_model():
  model = tf.keras.Sequential([layers.Dense(17),
                              layers.BatchNormalization(),
                              layers.Activation('sigmoid'),
                              layers.Dense(17),
                              layers.BatchNormalization(),
                              layers.Activation('sigmoid'),
                              layers.Dense(1)])
  model.compile(optimizer = tf.keras.optimizers.Adam(0.0001),
               loss = 'mse',
               metrics = ['mape', 'mae', 'mse'])
  return model
model = build_model()

history = model.fit(data, normed_label, epochs = 50, validation_split = 0.2, batch_size = 64, verbose =1)

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epochs'] = history.epoch
  plt.figure()
  plt.xlabel('epochs')
  plt.ylabel('mae')
  plt.plot(hist['epochs'], hist['mae'], label = 'train_mae')
  plt.plot(hist['epochs'], hist['val_mae'], label = 'val_mae')
  plt.legend()
  
  plt.figure()
  plt.xlabel('epochs')
  plt.ylabel('mse')
  plt.plot(hist['epochs'], hist['mse'], label = 'train_mse')
  plt.plot(hist['epochs'], hist['val_mse'], label = 'val_mse')
  plt.legend()
  
  plt.show()
  
plot_history(history)

sun_earth = {'radius': [2440*10**6, 3390*10**6, 6052*10**6],'mass':[(3.3*10**23)**-1, (6.4*10**23)**-1, (4.87*10**24)**-1]}
sun_earth_data = np.stack([sun_earth['radius'], sun_earth['mass']], axis = 1)
result1 = model.predict(sun_earth_data)
result = denorm(result1,denorm_factor)
print(result)
#수 화 금
# 수성 0.2409 
# 화성 1.8809
# 금성 0.6102 
# 지구 1.0000 
