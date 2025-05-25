# Нейронные сети:
# - свертчоные (конволюционные) нейронные сети (CNN) - компьютерное зрение, классификация изображений
# - рекурретные нейроные сети (RNN) - распознование рукописного текста, обработка естественного языка
# - генеративные состязательные сети (GAN) - создание художественных, музыкальных произведений
# - многослойный прецептрон - простейший тип НС

# глубина - количество слоев, ширина - количество нейронов
# средний слой - скрытый
# НС работают только с float
# Смещение у каждого нейрона, кроме входного слоя
# Вес у каждого нейрона
# Функция активации (RELU)

# алгоритм обратно распространения ошибки - оптимизатор
# оптимизатор включает вычисление частных производных
# градиентный спуск
# скорость обучения

# TensorFlow PyTorch

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions

import numpy as np

img_path = './dog.png'
img = image.load_img(img_path, target_size=(224,224))

# plt.imshow(img)
# plt.show()

img_array = image.img_to_array(img)
print(img_array.shape)

img_batch = np.expand_dims(img_array, axis=0)
print(img_batch.shape)

img_processed = preprocess_input(img_batch)

model = ResNet50()
prediction = model.predict(img_processed)
print(decode_predictions(prediction, top=5)[0])