from MNIST.utils import *
from keras.models import load_model
lenet1 = get_neuron_profile('lenet1')
print(lenet1)
model  = load_model('./data/models/lenet1.h5')
print(model.summary())
