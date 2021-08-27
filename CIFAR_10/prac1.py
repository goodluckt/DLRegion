from CIFAR_10.utils import *
from keras.models import load_model

model = load_model('./data/models/resnet.h5')
img_path = './seeds_50_1/8_3189.jpg'
img = cifar_preprocessing(img_path)
img1 = img.copy()
pred = model.predict(img1)
label_pred = np.argmax(pred[0])
print('pred')
print(pred)
print(type(label_pred))
print(label_pred)