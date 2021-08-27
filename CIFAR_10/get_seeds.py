import numpy as np
import os
# from keras.preprocessing import  image
# from keras.
import random
import shutil

import imageio
import numpy as np
from keras.models import load_model
from CIFAR_10.utils import *


def unpickle(file):
    import pickle
    with open(file, 'rb') as f:
        dict = pickle.load(f,encoding='bytes')
    return dict


def main(cifar10_data_dir):
    '''
    for i in range(1, 6):
        train_data_file = os.path.join(cifar10_data_dir, 'data_batch_' + str(i))
        print(train_data_file)
        data = unpickle(train_data_file)
        print('unpickle done')
        for j in range(10000):
            img = np.reshape(data[b'data'][j], (3, 32, 32))
            img = img.transpose(1, 2, 0)
            img_name = 'train/' + str(data[b'labels'][j]) + '_' + str(j + (i - 1) * 10000) + '.jpg'
            imageio.imwrite(os.path.join(cifar10_data_dir, img_name), img)
    '''
    test_data_file = os.path.join(cifar10_data_dir, 'test_batch')
    data = unpickle(test_data_file)
    for i in range(10000):
        img = np.reshape(data[b'data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        img_name = 'test/' + str(data[b'labels'][i]) + '_' + str(i) + '.jpg'
        imageio.imwrite(os.path.join(cifar10_data_dir, img_name), img)



'''def cifar_preprocessing(img_path):
    img = image.load_img(img_path, target_size=(32, 32 ))
    temp = image.img_to_array(img)
    input_img_data = temp.reshape(1, 32, 32, 3)
    input_img_data = input_img_data.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        input_img_data[ :, :,:, i] = (input_img_data[ :, :,:, i] - mean[i]) / std[i]
    return input_img_data
'''

def random_get_seeds(model,img_dir,save_dir,get_input_num):
    img_paths = os.listdir(img_dir)
    img_num = len(img_paths)
    num = 0
    while num < get_input_num:
        choose_img_idx = random.sample(range(img_num),get_input_num-num)
        for i in choose_img_idx:
            print(i)
            img_path = os.path.join(img_dir,img_paths[i])
            img_label = img_paths[i].split('_')[0]
            print('img_path'+img_path)
            print('img_label'+img_label)
            #img = preprocess_image(img_path)
            img = cifar_preprocessing(img_path)
            pred = model.predict(img)
            label_pred = np.argmax(pred[0])
            print(label_pred)
            if int(img_label) == int(label_pred):
                print('yes')
                if not os.path.exists(os.path.join(save_dir,img_paths[i])):
                    num += 1
                    print('num'+str(num))
                    shutil.copy(img_path,save_dir)


if __name__ == "__main__":
    #main('./data/cifar-10-batches-py')
    model = load_model('./data/models/resnet.h5')
    img_dir = './data/cifar-10-batches-py/test'
    save_dir = './seeds_50_1'
    creat_path(save_dir)
    random_get_seeds(model,img_dir,save_dir,50)



