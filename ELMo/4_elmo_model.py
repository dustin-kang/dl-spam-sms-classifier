# -*- coding: utf-8 -*-
"""4_ELMo_Model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14Y10TI0OgPKmIEarhBEbgR1N9fo25TV4

# ELMo를 활용한 스팸 문자 메세지 분류

[paper](https://aclweb.org/anthology/N18-1202)

## 4-1. 모듈 불러오기

- 모듈을 다운 받고 Tensorflow Hub에서 ELMo를 다운 받습니다.
- 텐서플로우 버전 2.0을 사용하려고 했지만 불행히도 아직 업데이트가 되지 않은 것 같습니다. [링크](https://github.com/tensorflow/hub/issues/412)
- 그래서 1.0버전으로 사용하기로 했습니다.

[ELMO Tensorflow Hub](https://tfhub.dev/google/elmo/3)
"""

# !pip uninstall tensorflow tensorflow_hub tensorflowjs
# !pip install tensorflow==2.0.0a0 tensorflow_hub==0.5.0 tensorflowjs==1.2.6

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x

!pip install tensorflow-hub

import tensorflow_hub as hub
import tensorflow as tf
from keras import backend as K
import urllib.request
import pandas as pd
import numpy as np

elmo = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)

session = tf.Session()
K.set_session(session)
session.run(tf.global_variables_initializer())
session.run(tf.tables_initializer())

"""## 4-2. 데이터 전처리 및 분리"""

from google.colab import drive
drive.mount('/content/drive')

# data = pd.read_csv('/content/spam.csv', encoding='latin-1')
data = pd.read_csv('/content/sms_data.csv')

data.head()

data['v1'] = data['v1'].replace(['ham','spam'],[0,1])

data = data.rename(columns={"v1": "category", "v2": "text"})

y_data = list(data['category'])
X_data = list(data['text'])

train_count = int(len(X_data) * 0.8)
test_count = int(len(X_data) - train_count)

print(f"{len(X_data)} = {train_count} + {test_count}")

X_train = np.asarray(X_data[:train_count])
X_test = np.asarray(X_data[test_count:])
y_train =  np.asarray(y_data[:train_count])
y_test = np.asarray(y_data[test_count:])

"""## 4-3. ELMo 모델링"""

def ELMo(x):
  return elmo(tf.squeeze(tf.cast(x, tf.string)), as_dict=True, signature="default")["default"]

from keras.models import Model
from keras.layers import Dense, Lambda, Input

input = Input(shape=(1,), dtype=tf.string)
embedding = Lambda(ELMo, output_shape=(1024, ))(input)

hidden = Dense(256, activation='relu')(embedding)

output = Dense(1, activation='sigmoid')(hidden)

model = Model(inputs=[input], outputs=output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping, ModelCheckpoint
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, verbose=1)
checkpoint = ModelCheckpoint('/content/drive/MyDrive/elmo_cp.1', monitor='val_acc', save_best_only=True, verbose=1)

history = model.fit(X_train, y_train, callbacks=[checkpoint, earlystopping], epochs=10, batch_size=64, validation_split=0.2)

print("\n Test_accuracy: %.3f" % (model.evaluate(X_test, y_test)[1]))

model.save('/content/drive/MyDrive/elmo_model')

import matplotlib.pyplot as plt


epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('ELMo Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()