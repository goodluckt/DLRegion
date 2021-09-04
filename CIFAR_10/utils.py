import random
from collections import defaultdict
import numpy as np
import os
from datetime import datetime
from keras import backend as K
#from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.resnet import preprocess_input
from keras.models import Model
from keras.preprocessing import image
import pickle
import shutil


model_layer_weights_top_k = []

model_profile_path = {
    'resnet20': "./data/profiling/resnet20/0_50000.pickle",
    #'vgg16': "./data/profiling/vgg16/0_50000.pickle",
    #'resnet50': './data/profiling/resnet50_merged.pickle',
}

def creat_path(path):
    if os.path.exists(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            if os.path.isfile(path_file):
                os.remove(path_file)
    else:
        os.makedirs(path)
'''
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    input_img_data = image.img_to_array(img)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)  # final input shape = (1,32,32,3)
    return input_img_data
'''
'''
def deprocess_image(x): #true
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, 32, 32))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((32, 32, 3))
        # (Remove zero-center by mean pixel)
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x
'''
def deprocess_image(x):
    x = x.reshape((32, 32, 3))
    # Remove zero-center by mean pixel
    #x[:, :, 0] += 103.939
    #x[:, :, 1] += 116.779
    #x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    #x = x[:, :, ::-1]
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x
'''
def cifar_preprocessing(img_path): #true
    img = image.load_img(img_path, target_size=(32, 32))
    temp = image.img_to_array(img)
    input_img_data = temp.reshape(1, 32, 32, 3)
    input_img_data = input_img_data.astype('float32')
    input_img_data = preprocess_input(input_img_data)
    #return img
    #mean = [125.307, 122.95, 113.865]
    #std = [62.9932, 62.0887, 66.7048]
    #for i in range(3):
        #input_img_data[ :, :,:, i] = (input_img_data[ :, :,:, i] - mean[i]) / std[i]
    return input_img_data
'''

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    input_img_data = image.img_to_array(img)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = input_img_data.astype(float)
    input_img_data /= 255
    #input_img_data = preprocess_input(input_img_data)  # final input shape = (1,224,224,3)
    return input_img_data

'''
def decode_label(pred):
    return decode_predictions(pred)[0][0][1]
'''
def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def get_neuron_profile(model_name):
    profile_dict = pickle.load(open(model_profile_path[model_name], 'rb'),encoding='iso-8859-1')
    return profile_dict

def init_times(model,model_layer_times):
    for layer in model.layers:
        if 'input' in layer.name or 'flatten' in layer.name:# or 'batch' in layer.name or 'activation' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_times[(layer.name,index)] = 0

def init_value(model,model_layer_neuron_value):
    for layer in model.layers:
        if 'input' in layer.name or 'flatten' in layer.name:# or 'batch' in layer.name or 'activation' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_neuron_value[(layer.name,index)] = 0

def init_coverage_times_NC(model):
    model_layer_times = defaultdict(int)
    init_times(model,model_layer_times)
    return model_layer_times

def init_coverage_times(model,criterion,criterion_para):
    if criterion == 'NC':
        model_layer_times = {}
        for layer in model.layers:
            if 'input' in layer.name or 'flatten' in layer.name:# or 'batch' in layer.name or 'activation' in layer.name:
                continue
            for index in range(layer.output_shape[-1]):
                model_layer_times[(layer.name, index)] = 0
        return model_layer_times
    elif criterion == 'KMNC':
        model_layer_times = {}
        for layer in model.layers:
            if 'input' in layer.name or 'flatten' in layer.name or 'batch' in layer.name or 'activation' in layer.name:
                continue
            for index in range(layer.output_shape[-1]):
                model_layer_times[(layer.name, index)] = np.full(int(criterion_para), 0, dtype=np.uint8)
        return model_layer_times
    elif criterion == 'NBC':
        model_layer_times = {}
        for layer in model.layers:
            if 'input' in layer.name or 'flatten' in layer.name or 'batch' in layer.name or 'activation' in layer.name:
                continue
            for index in range(layer.output_shape[-1]):
                model_layer_times[(layer.name, index)] = np.full(2, 0, dtype=np.uint8)
        return model_layer_times
    elif criterion == 'SNAC':
        model_layer_times = {}
        for layer in model.layers:
            if 'input' in layer.name or 'flatten' in layer.name or 'batch' in layer.name or 'activation' in layer.name:
                continue
            for index in range(layer.output_shape[-1]):
                model_layer_times[(layer.name, index)] = 0
        return model_layer_times
    elif criterion == 'TKNC':
        model_layer_times = {}
        for layer in model.layers:
            if 'input' in layer.name or 'flatten' in layer.name:# or 'batch' in layer.name or 'activation' in layer.name:
                continue
            for index in range(layer.output_shape[-1]):
                model_layer_times[(layer.name, index)] = 0
        return model_layer_times

def init_layer_neuron_value(model):
    model_layer_neuron_value = defaultdict(float)
    init_value(model,model_layer_neuron_value)
    return model_layer_neuron_value



def scale(intermediate_layer_output,rmax=1,rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min())/(
            intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax-rmin) + rmin
    return  X_scaled

def neuron_covered_num(criterion,model_layer_times):
    if criterion == 'NC':
        covered_num = len([v for v in model_layer_times.values() if v > 0])
        total_neurons = len(model_layer_times)
        return covered_num, total_neurons, covered_num / float(total_neurons)
    elif criterion == 'KMNC':
        total_neurons = len(model_layer_times)
        total_equal = 0
        for key, value in model_layer_times.items():
            neuron_equal_num = len([x for x in value if x > 0])
            equal_num = len(value)
            neuron_equal_ratio = neuron_equal_num / equal_num
            total_equal += neuron_equal_ratio
        return total_equal, total_neurons, total_equal / float(total_neurons)
    elif criterion == 'NBC':
        total_neurons = len(model_layer_times)
        corner_neuron_num = 0
        for key, value in model_layer_times.items():
            corner_neuron_num += len([x for x in value if x > 0])
        return corner_neuron_num, total_neurons, corner_neuron_num / float(2 * total_neurons)
    elif criterion == 'SNAC':
        covered_num = len([v for v in model_layer_times.values() if v > 0])
        total_neurons = len(model_layer_times)
        return covered_num, total_neurons, covered_num / float(total_neurons)
    elif criterion == 'TKNC':
        covered_num = len([v for v in model_layer_times.values() if v > 0])
        total_neurons = len(model_layer_times)
        return covered_num, total_neurons, covered_num / float(total_neurons)

def update_NC_coverage(input_data,model,model_layer_times,threshold=0):
    layer_names = [layer.name for layer in model.layers
                 if 'input' not in layer.name and 'flatten' not in layer.name]# and 'batch' not in layer.name and 'activation' not in layer.name ]
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)
    for layer_idx,intermediate_layer_output in enumerate(intermediate_layer_outputs):
        Scaled = scale(intermediate_layer_output[0])
        for neuron_idx in range(Scaled.shape[-1]):
            if np.mean(Scaled[...,neuron_idx]) > threshold:
                model_layer_times[(layer_names[layer_idx],neuron_idx)] += 1
    return intermediate_layer_outputs

def update_KMNC_coverage(input_data,model,model_name,model_layer_times,criterion_para):
    profile_dict = get_neuron_profile(model_name)
    layer_names = [layer.name for layer in model.layers
                   if 'input' not in layer.name and 'flatten' not in layer.name and 'batch' not in layer.name and  'activation' not in layer.name]
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)
    for layer_idx, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        for neuron_idx in range(intermediate_layer_output.shape[-1]):
            neuron_value = np.mean(intermediate_layer_output[0][...,neuron_idx])
            upper_bound = profile_dict[(layer_names[layer_idx],neuron_idx)][-1]
            lower_bound = profile_dict[(layer_names[layer_idx],neuron_idx)][-2]
            meta = (upper_bound-lower_bound)/criterion_para
            if meta== 0:
                continue
            if neuron_value < lower_bound or neuron_value >upper_bound:
                continue
            subrange_idx = int((neuron_value-lower_bound)/meta)
            if subrange_idx == criterion_para:
                subrange_idx -= 1
            model_layer_times[(layer_names[layer_idx],neuron_idx)][subrange_idx] += 1
    return  intermediate_layer_outputs

def update_NBC_coverage(input_data, model, model_name, model_layer_times,criterion_para):
    profile_dict = get_neuron_profile(model_name)
    layer_names = [layer.name for layer in model.layers
                   if 'input' not in layer.name and 'flatten' not in layer.name and 'batch' not in layer.name and 'activation' not in layer.name]
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)
    for layer_idx, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        for neuron_idx in range(intermediate_layer_output.shape[-1]):
            neuron_value = np.mean(intermediate_layer_output[0][..., neuron_idx])
            upper_bound = profile_dict[(layer_names[layer_idx], neuron_idx)][-1]
            lower_bound = profile_dict[(layer_names[layer_idx], neuron_idx)][-2]
            standard_dev = profile_dict[(layer_names[layer_idx],neuron_idx)][-3]
            if neuron_value < lower_bound - criterion_para * standard_dev:
                model_layer_times[(layer_names[layer_idx],neuron_idx)][0] += 1
            elif neuron_value > upper_bound + criterion_para* standard_dev:
                model_layer_times[(layer_names[layer_idx],neuron_idx)][1] += 1
    '''neuron_region_dict = get_neuron_output_region(input_data,model,model_name)
        lowerCornerNeuron_list = [(layer_name,neuron_idx) for (layer_name,neuron_idx), region in  neuron_region_dict.items() if region == 0 ]
        upperCornerNeuron_list = [(layer_name,neuron_idx) for (layer_name,neuron_idx), region in  neuron_region_dict.items() if region == 5 ]
        for i in range(len(lowerCornerNeuron_list)):
            (layer_name,neuron_idx) = lowerCornerNeuron_list[i]
            model_layer_times[(layer_name,neuron_idx)][0] += 1
        for j in range(len(upperCornerNeuron_list)):
            (layer_name, neuron_idx) = upperCornerNeuron_list[j]
            model_layer_times[(layer_name, neuron_idx)][1] += 1
        print('lower corner neuron num :' + str(len(lowerCornerNeuron_list)))
        print('upper corner neuron num :' + str(len(upperCornerNeuron_list)))'''

def update_SNAC_coverage(input_data,model,model_name,model_layer_times,criterion_para):
    profile_dict = get_neuron_profile(model_name)
    layer_names = [layer.name for layer in model.layers
                   if 'input' not in layer.name and 'flatten' not in layer.name and 'batch' not in layer.name and 'activation' not in layer.name]
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)
    for layer_idx, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        for neuron_idx in range(intermediate_layer_output.shape[-1]):
            neuron_value = np.mean(intermediate_layer_output[0][..., neuron_idx])
            upper_bound = profile_dict[(layer_names[layer_idx], neuron_idx)][-1]
            standard_dev = profile_dict[(layer_names[layer_idx], neuron_idx)][-3]
            if neuron_value > upper_bound + criterion_para* standard_dev:
                model_layer_times[(layer_names[layer_idx],neuron_idx)] += 1

def update_TKNC_coverage(input_data,model,model_layer_times,criterion_para):
    layer_names = [layer.name for layer in model.layers
                   if 'input' not in layer.name and 'flatten' not in layer.name]# and 'batch' not in layer.name and 'activation' not in layer.name]
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)
    for layer_idx, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        neuron_value_list = []
        for neuron_idx in range(intermediate_layer_output.shape[-1]):
            neuron_value = np.mean(intermediate_layer_output[0][..., neuron_idx])
            neuron_value_list.append(neuron_value)
        top_k_neuron_idx = np.argsort(neuron_value_list)[-int(criterion_para):]
        for i in top_k_neuron_idx:
            model_layer_times[(layer_names[layer_idx], i)] += 1

def update_coverage(input_data,model,model_name,model_layer_times,criterion,criterion_para=0):
    if criterion == 'NC' :
        update_NC_coverage(input_data,model,model_layer_times,criterion_para)
    elif criterion == 'KMNC':
        update_KMNC_coverage(input_data,model,model_name,model_layer_times,criterion_para)
    elif criterion == 'NBC' :
        update_NBC_coverage(input_data,model,model_name,model_layer_times,criterion_para)
    elif criterion == 'SNAC' :
        update_SNAC_coverage(input_data,model,model_name,model_layer_times,criterion_para)
    elif criterion == 'TKNC':
        update_TKNC_coverage(input_data, model, model_layer_times, criterion_para)

def update_neuron_value(input_data,model,model_layer_neuron_value):
    layer_names = [layer.name for layer in model.layers
                   if 'input' not in layer.name and 'flatten' not in layer.name]# and 'batch' not in layer.name and 'activation' not in layer.name]
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)
    for layer_idx,intermediate_layer_output in enumerate(intermediate_layer_outputs):
        Scaled = scale(intermediate_layer_output[0])
        for neuron_idx in range(Scaled.shape[-1]):
            model_layer_neuron_value[(layer_names[layer_idx],neuron_idx)] = np.mean(Scaled[...,neuron_idx])
    return intermediate_layer_outputs


def spilt_neuron_output(model_name):
    profile_dict = get_neuron_profile(model_name)
    equal = 1000
    low_up_equal = up_high_equal =  300
    neuron_spilt_bound_dict = {}
    for key,value in profile_dict.items():
        lower_bound = value[-2]
        higher_bound = value[-1]
        low_up_bound = ((higher_bound - lower_bound) * low_up_equal / equal) + lower_bound
        up_high_bound = higher_bound - ((higher_bound - lower_bound) * up_high_equal / equal)
        neuron_spilt_bound_dict[(key[0],key[1])] = [lower_bound,low_up_bound,up_high_bound,higher_bound]
    return neuron_spilt_bound_dict

def get_neuron_output_region(input_data,model,model_name):
    neuron_spilt_bound_dict =spilt_neuron_output(model_name)
    neuron_region_dict={}
    layer_names = [layer.name for layer in model.layers
                   if 'input' not in layer.name and 'flatten' not in layer.name and 'batch' not in layer.name and 'activation' not in layer.name]
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediat_layer_outputs = intermediate_layer_model.predict(input_data)
    for layer_idx,intermediat_layer_output in enumerate(intermediat_layer_outputs):
        #Scaled = scale(intermediate_layer_output[0])    #profile 文件好像没有
        for neuron_idx in range(intermediat_layer_output.shape[-1]):
            neuron_value = np.mean(intermediat_layer_output[0][...,neuron_idx])
            if neuron_value <= neuron_spilt_bound_dict[(layer_names[layer_idx],neuron_idx)][0]:
                neuron_region_dict[(layer_names[layer_idx], neuron_idx)] = 1
                if neuron_value < neuron_spilt_bound_dict[(layer_names[layer_idx],neuron_idx)][0]:
                    neuron_region_dict[(layer_names[layer_idx], neuron_idx)] = 0
            elif neuron_value <= neuron_spilt_bound_dict[(layer_names[layer_idx],neuron_idx)][1]:
                neuron_region_dict[(layer_names[layer_idx], neuron_idx)] = 2
            elif neuron_value <= neuron_spilt_bound_dict[(layer_names[layer_idx], neuron_idx)][2]:
                neuron_region_dict[(layer_names[layer_idx], neuron_idx)] = 3
            elif neuron_value <= neuron_spilt_bound_dict[(layer_names[layer_idx], neuron_idx)][3]:
                neuron_region_dict[(layer_names[layer_idx], neuron_idx)] = 4
            else:
                neuron_region_dict[(layer_names[layer_idx], neuron_idx)] = 5
    return neuron_region_dict

def neuron_to_cover_DeepXplore(model_layer_dict): #从model_layer_dict 随机选取未被激活的神经元
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]  #dict_items([(('a', 0), False), (('a', 1), False)])
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())  #dict_keys([('a', 0), ('a', 1)])
    return layer_name, index

def neuron_to_cover(not_covered,model_layer_times_NC):
    if not_covered:
        layer_name,neuron_idx = random.choice(not_covered)
        not_covered.remove((layer_name,neuron_idx))
    else:
        layer_name,neuron_idx = random.choice(model_layer_times_NC.keys())
    return layer_name,neuron_idx

def neuron_select_high_weight(model, layer_names, top_k):
    global model_layer_weights_top_k  #存放参数layer_names指定的层中权值top_k所对应的[layer_name,index]
    model_layer_weights_dict = {}
    for layer_name in layer_names:
        weights = model.get_layer(layer_name).get_weights() #得到该层(包括很多神经元）的权值组合
        if len(weights) <= 0:
            continue
        w = np.asarray(weights[0])  # 0 is weights, 1 is biases
        w = w.reshape(w.shape)
        for index in range(model.get_layer(layer_name).output_shape[-1]):  #遍历layer_name对应层的神经元
            index_w = np.mean(w[..., index])  #该神经元的权值
            if index_w <= 0:
                continue
            model_layer_weights_dict[(layer_name,index)]=index_w  #存储了每层有权值的神经元的权值的平均值
    # notice!  #dict_items([(('a', 0), 9), (('a', 1), 6)])
    model_layer_weights_list = sorted(model_layer_weights_dict.items(), key=lambda x: x[1], reverse=True)  #key=lambda x: x[1] 表示按照列表的第二个元素排序 reverse=True 表示降序排序

    k = 0
    for (layer_name, index),weight in model_layer_weights_list:
        if k >= top_k:
            break
        model_layer_weights_top_k.append([layer_name,index])  #选取权值topk的神经元
        k += 1


def random_strategy(model,model_layer_times_NC,neuron_to_cover_num): #随机选择神经元 不按照神经元选择策略
    loss_neuron = []
    not_covered = [(layer_name,neuron_idx) for (layer_name,neuron_idx),v in model_layer_times_NC.items() if v==0 ]
    for _ in range(neuron_to_cover_num):
        layer_name,neuron_idx = neuron_to_cover(not_covered,model_layer_times_NC)
        loss00_neuron = K.mean(model.get_layer(layer_name).output[...,neuron_idx])
        loss_neuron.append(loss00_neuron)
    return loss_neuron

def get_neuron_cover_time(layer_neuronidx_list,model_layer_times):
    model_layer_times_region = {}
    for value  in layer_neuronidx_list:
        times = model_layer_times[(value[0],value[1])]
        model_layer_times_region[(value[0],value[1])] = times
    return model_layer_times_region

def get_loss_neuron_astime(model,model_layer_times,neuron_to_choose_num):
    i = 0
    neuron_key_pos = {}
    neuron_covered_times = []
    loss_neuron_layer_name = []
    for (layer_name,neuron_idx),times in model_layer_times.items():
        neuron_covered_times.append(times)
        neuron_key_pos[i] = (layer_name,neuron_idx)
        i += 1
    neuron_covered_times_sort = np.argsort(neuron_covered_times)
    for pos in neuron_covered_times_sort[:neuron_to_choose_num]:
        loss_neuron_layer_name.append(neuron_key_pos[pos])
    return  loss_neuron_layer_name

def get_loss_neuron_as_mosttime(model,model_layer_times,neuron_to_choose_num):
    i = 0
    neuron_key_pos = {}
    neuron_covered_times = []
    loss_neuron_layer_name = []
    for (layer_name,neuron_idx),times in model_layer_times.items():
        neuron_covered_times.append(times)
        neuron_key_pos[i] = (layer_name,neuron_idx)
        i += 1
    neuron_covered_times_sort = np.argsort(neuron_covered_times)
    for pos in neuron_covered_times_sort[-neuron_to_choose_num:]:
        loss_neuron_layer_name.append(neuron_key_pos[pos])
    return  loss_neuron_layer_name

def neuron_selection(model,neuron_region_dict,neuron_selection_strategy,model_layer_times_NC,neuron_to_cover_num):
    region_neuron_num = 0
    if neuron_selection_strategy == None:
        return random_strategy(model,model_layer_times_NC,neuron_to_cover_num)
    num_strategy = len([x for x in neuron_selection_strategy if x in ['1','2','3','4','5']])
    neuron_to_cover_num_eachstrategy = int(neuron_to_cover_num / num_strategy)

    loss_neuron = []
    if '1' in neuron_selection_strategy:
        neuron_region1 = [(layer_name,neuron_idx) for (layer_name,neuron_idx), region in neuron_region_dict.items() if region == 1]
        region_neuron_num = len(neuron_region1)
        print(len(neuron_region1))
        #if len(neuron_region1) == 0:
            #return  random_strategy(model,model_layer_times_NC,1) , region_neuron_num
        if len(neuron_region1) < neuron_to_cover_num_eachstrategy:#从neuron_region1中随机选择
            choose_neuron_region1 = neuron_region1
        else:
            choose_neuron_region1 = random.sample(neuron_region1,neuron_to_cover_num_eachstrategy)
        for i in range(len(choose_neuron_region1)):
            layer_name,neuron_idx = choose_neuron_region1[i]
            loss0_neuron = K.mean(model.get_layer(layer_name).output[...,neuron_idx])
            loss_neuron.append(loss0_neuron)

    if '2' in neuron_selection_strategy:
        neuron_region2 = [(layer_name,neuron_idx) for (layer_name,neuron_idx),region in neuron_region_dict.items() if region == 2 ]
        region_neuron_num = len(neuron_region2)
        print(len(neuron_region2))
        #if len(neuron_region2) == 0:
            #return random_strategy(model,model_layer_times_NC,1) , region_neuron_num
        if len(neuron_region2) < neuron_to_cover_num_eachstrategy:
            choose_neuron_region2 = neuron_region2
        else:
            choose_neuron_region2 = random.sample(neuron_region2,neuron_to_cover_num_eachstrategy)
        for i in range(len(choose_neuron_region2)):
            layer_name,neuron_idx = choose_neuron_region2[i]
            loss0_neuron = K.mean(model.get_layer(layer_name).output[...,neuron_idx])
            loss_neuron.append(loss0_neuron)

    if '3' in neuron_selection_strategy:
        neuron_region3 = [(layer_name,neuron_idx) for (layer_name,neuron_idx),region in neuron_region_dict.items() if region == 3 ]
        region_neuron_num = len(neuron_region3)
        print(len(neuron_region3))
        #if len(neuron_region3) == 0:
            #return random_strategy(model,model_layer_times_NC,1) , region_neuron_num
        if len(neuron_region3) < neuron_to_cover_num_eachstrategy:
            choose_neuron_region3 = neuron_region3
        else:
            choose_neuron_region3 = random.sample(neuron_region3,neuron_to_cover_num_eachstrategy)
        for i in range(len(choose_neuron_region3)):
            layer_name,neuron_idx = choose_neuron_region3[i]
            loss0_neuron = K.mean(model.get_layer(layer_name).output[...,neuron_idx])
            loss_neuron.append(loss0_neuron)

    if '4' in neuron_selection_strategy:
        neuron_region4 = [(layer_name,neuron_idx) for (layer_name,neuron_idx),region in neuron_region_dict.items() if region == 4 ]
        region_neuron_num = len(neuron_region4)
        print(len(neuron_region4))
        #if len(neuron_region4) == 0:
            #return random_strategy(model,model_layer_times_NC,1) , region_neuron_num
        if len(neuron_region4) < neuron_to_cover_num_eachstrategy:
            choose_neuron_region4 = neuron_region4
        else:
            choose_neuron_region4 = random.sample(neuron_region4,neuron_to_cover_num_eachstrategy)
        for i in range(len(choose_neuron_region4)):
            layer_name,neuron_idx = choose_neuron_region4[i]
            loss0_neuron = K.mean(model.get_layer(layer_name).output[...,neuron_idx])
            loss_neuron.append(loss0_neuron)

    if '5' in neuron_selection_strategy:
        neuron_region5 = [(layer_name,neuron_idx) for (layer_name,neuron_idx),region in neuron_region_dict.items() if region == 5 ]
        region_neuron_num = len(neuron_region5)
        print(len(neuron_region5))
        #if len(neuron_region5) == 0:
            #return random_strategy(model,model_layer_times_NC,1) , region_neuron_num
        if len(neuron_region5) < neuron_to_cover_num_eachstrategy:
            choose_neuron_region5 = neuron_region5
        else:
            choose_neuron_region5 = random.sample(neuron_region5,neuron_to_cover_num_eachstrategy)
        for i in range(len(choose_neuron_region5)):
            layer_name,neuron_idx = choose_neuron_region5[i]
            loss0_neuron = K.mean(model.get_layer(layer_name).output[...,neuron_idx])
            loss_neuron.append(loss0_neuron)

    if len(loss_neuron) == 0:
        return  random_strategy(model,model_layer_times_NC,1) , region_neuron_num
    print(len(loss_neuron))
    return loss_neuron , region_neuron_num

def neuron_selection_addtime(model,neuron_region_dict,neuron_selection_area,model_layer_times_NC,neuron_to_cover_num):
    region_neuron_num = 0
    if neuron_selection_area == None:
        return random_strategy(model,model_layer_times_NC,neuron_to_cover_num)
    num_strategy = len([x for x in neuron_selection_area if x in ['1','2','3','4','5']])
    neuron_to_cover_num_eachstrtegy = int(neuron_to_cover_num / num_strategy)

    loss_neuron = []
    if '1' in neuron_selection_area:
        neuron_region1 = [(layer_name,neuron_idx) for (layer_name,neuron_idx), region in neuron_region_dict.items() if region == 1]
        region_neuron_num = len(neuron_region1)
        print(len(neuron_region1))
        if len(neuron_region1) < neuron_to_cover_num_eachstrtegy:
            choose_neuron_region1 = neuron_region1
            loss_neuron_layer_name = get_loss_neuron_astime(model,model_layer_times_NC,neuron_to_cover_num_eachstrtegy-len(neuron_region1))
            for key in loss_neuron_layer_name:
                choose_neuron_region1.append(key)
        else:
            choose_neuron_region1 = get_loss_neuron_astime(model,get_neuron_cover_time(neuron_region1,model_layer_times_NC),neuron_to_cover_num_eachstrtegy)
        for i in range(len(choose_neuron_region1)):
            layer_name,neuron_idx = choose_neuron_region1[i]
            loss0_neuron = K.mean(model.get_layer(layer_name).output[...,neuron_idx])
            loss_neuron.append(loss0_neuron)

    if '2' in neuron_selection_area:
        neuron_region2 = [(layer_name,neuron_idx) for (layer_name,neuron_idx),region in neuron_region_dict.items() if region == 2 ]
        region_neuron_num = len(neuron_region2)
        print(len(neuron_region2))
        if len(neuron_region2) < neuron_to_cover_num_eachstrtegy:
            choose_neuron_region2 = neuron_region2
            loss_neuron_layer_name = get_loss_neuron_astime(model, model_layer_times_NC,neuron_to_cover_num_eachstrtegy - len(neuron_region2))
            for key in loss_neuron_layer_name:
                choose_neuron_region2.append(key)
        else:
            choose_neuron_region2 = get_loss_neuron_astime(model,get_neuron_cover_time(neuron_region2, model_layer_times_NC),neuron_to_cover_num_eachstrtegy)
        for i in range(len(choose_neuron_region2)):
            layer_name, neuron_idx = choose_neuron_region2[i]
            loss0_neuron = K.mean(model.get_layer(layer_name).output[..., neuron_idx])
            loss_neuron.append(loss0_neuron)

    if '3' in neuron_selection_area:
        neuron_region3 = [(layer_name,neuron_idx) for (layer_name,neuron_idx),region in neuron_region_dict.items() if region == 3 ]
        region_neuron_num = len(neuron_region3)
        print(len(neuron_region3))
        if len(neuron_region3) < neuron_to_cover_num_eachstrtegy:
            choose_neuron_region3 = neuron_region3
            loss_neuron_layer_name = get_loss_neuron_astime(model, model_layer_times_NC,neuron_to_cover_num_eachstrtegy - len(neuron_region3))
            for key in loss_neuron_layer_name:
                choose_neuron_region3.append(key)
        else:
            choose_neuron_region3 = get_loss_neuron_astime(model,get_neuron_cover_time(neuron_region3, model_layer_times_NC),neuron_to_cover_num_eachstrtegy)
        for i in range(len(choose_neuron_region3)):
            layer_name, neuron_idx = choose_neuron_region3[i]
            loss0_neuron = K.mean(model.get_layer(layer_name).output[..., neuron_idx])
            loss_neuron.append(loss0_neuron)

    if '4' in neuron_selection_area:
        neuron_region4 = [(layer_name,neuron_idx) for (layer_name,neuron_idx),region in neuron_region_dict.items() if region == 4 ]
        region_neuron_num = len(neuron_region4)
        print(region_neuron_num)
        if len(neuron_region4) < neuron_to_cover_num_eachstrtegy:
            choose_neuron_region4 = neuron_region4
            loss_neuron_layer_name = get_loss_neuron_astime(model, model_layer_times_NC,neuron_to_cover_num_eachstrtegy - len(neuron_region4))
            for key in loss_neuron_layer_name:
                choose_neuron_region4.append(key)
        else:
            choose_neuron_region4 = get_loss_neuron_astime(model,get_neuron_cover_time(neuron_region4, model_layer_times_NC),neuron_to_cover_num_eachstrtegy)
        for i in range(len(choose_neuron_region4)):
            layer_name, neuron_idx = choose_neuron_region4[i]
            loss0_neuron = K.mean(model.get_layer(layer_name).output[..., neuron_idx])
            loss_neuron.append(loss0_neuron)

    if '5' in neuron_selection_area:
        neuron_region5 = [(layer_name,neuron_idx) for (layer_name,neuron_idx),region in neuron_region_dict.items() if region == 5 ]
        region_neuron_num = len(neuron_region5)
        print(region_neuron_num)
        if len(neuron_region5) < neuron_to_cover_num_eachstrtegy:
            choose_neuron_region5 = neuron_region5
            loss_neuron_layer_name = get_loss_neuron_astime(model, model_layer_times_NC,neuron_to_cover_num_eachstrtegy - len(neuron_region5))
            for key in loss_neuron_layer_name:
                choose_neuron_region5.append(key)
        else:
            choose_neuron_region5 = get_loss_neuron_astime(model,get_neuron_cover_time(neuron_region5, model_layer_times_NC),neuron_to_cover_num_eachstrtegy)
        for i in range(len(choose_neuron_region5)):
            layer_name, neuron_idx = choose_neuron_region5[i]
            loss0_neuron = K.mean(model.get_layer(layer_name).output[..., neuron_idx])
            loss_neuron.append(loss0_neuron)

    if len(loss_neuron) == 0:
        return  random_strategy(model,model_layer_times_NC,1) , region_neuron_num
    print(len(loss_neuron))
    return loss_neuron , region_neuron_num

def neuron_selection_addmosttime(model,neuron_region_dict,neuron_selection_area,model_layer_times_NC,neuron_to_cover_num):
    region_neuron_num = 0
    if neuron_selection_area == None:
        return random_strategy(model, model_layer_times_NC, neuron_to_cover_num)
    num_strategy = len([x for x in neuron_selection_area if x in ['1', '2', '3', '4', '5']])
    neuron_to_cover_num_eachstrtegy = int(neuron_to_cover_num / num_strategy)

    loss_neuron = []
    if '1' in neuron_selection_area:
        neuron_region1 = [(layer_name, neuron_idx) for (layer_name, neuron_idx), region in neuron_region_dict.items() if
                          region == 1]
        region_neuron_num = len(neuron_region1)
        print(len(neuron_region1))
        if len(neuron_region1) < neuron_to_cover_num_eachstrtegy:
            choose_neuron_region1 = neuron_region1
            loss_neuron_layer_name = get_loss_neuron_as_mosttime(model, model_layer_times_NC,neuron_to_cover_num_eachstrtegy - len(neuron_region1))
            for key in loss_neuron_layer_name:
                choose_neuron_region1.append(key)
        else:
            choose_neuron_region1 = get_loss_neuron_as_mosttime(model,get_neuron_cover_time(neuron_region1, model_layer_times_NC),neuron_to_cover_num_eachstrtegy)
        for i in range(len(choose_neuron_region1)):
            layer_name, neuron_idx = choose_neuron_region1[i]
            loss0_neuron = K.mean(model.get_layer(layer_name).output[..., neuron_idx])
            loss_neuron.append(loss0_neuron)
    if '2' in neuron_selection_area:
        neuron_region2 = [(layer_name, neuron_idx) for (layer_name, neuron_idx), region in neuron_region_dict.items() if
                          region == 2]
        region_neuron_num = len(neuron_region2)
        print(len(neuron_region2))
        if len(neuron_region2) < neuron_to_cover_num_eachstrtegy:
            choose_neuron_region2 = neuron_region2
            loss_neuron_layer_name = get_loss_neuron_as_mosttime(model, model_layer_times_NC,neuron_to_cover_num_eachstrtegy - len(neuron_region2))
            for key in loss_neuron_layer_name:
                choose_neuron_region2.append(key)
        else:
            choose_neuron_region2 = get_loss_neuron_as_mosttime(model,get_neuron_cover_time(neuron_region2, model_layer_times_NC),neuron_to_cover_num_eachstrtegy)
        for i in range(len(choose_neuron_region2)):
            layer_name, neuron_idx = choose_neuron_region2[i]
            loss0_neuron = K.mean(model.get_layer(layer_name).output[..., neuron_idx])
            loss_neuron.append(loss0_neuron)
    if '3' in neuron_selection_area:
        neuron_region3 = [(layer_name, neuron_idx) for (layer_name, neuron_idx), region in neuron_region_dict.items() if
                          region == 3]
        region_neuron_num = len(neuron_region3)
        print(len(neuron_region3))
        if len(neuron_region3) < neuron_to_cover_num_eachstrtegy:
            choose_neuron_region3 = neuron_region3
            loss_neuron_layer_name = get_loss_neuron_as_mosttime(model, model_layer_times_NC,neuron_to_cover_num_eachstrtegy - len(neuron_region3))
            for key in loss_neuron_layer_name:
                choose_neuron_region3.append(key)
        else:
            choose_neuron_region3 = get_loss_neuron_as_mosttime(model,get_neuron_cover_time(neuron_region3, model_layer_times_NC),neuron_to_cover_num_eachstrtegy)
        for i in range(len(choose_neuron_region3)):
            layer_name, neuron_idx = choose_neuron_region3[i]
            loss0_neuron = K.mean(model.get_layer(layer_name).output[..., neuron_idx])
            loss_neuron.append(loss0_neuron)

    if '4' in neuron_selection_area:
        neuron_region4 = [(layer_name, neuron_idx) for (layer_name, neuron_idx), region in neuron_region_dict.items() if
                          region == 4]
        region_neuron_num = len(neuron_region4)
        print(len(neuron_region4))
        if len(neuron_region4) < neuron_to_cover_num_eachstrtegy:
            choose_neuron_region4 = neuron_region4
            loss_neuron_layer_name = get_loss_neuron_as_mosttime(model, model_layer_times_NC,neuron_to_cover_num_eachstrtegy - len(neuron_region4))
            for key in loss_neuron_layer_name:
                choose_neuron_region4.append(key)
        else:
            choose_neuron_region4 = get_loss_neuron_as_mosttime(model,get_neuron_cover_time(neuron_region4, model_layer_times_NC),neuron_to_cover_num_eachstrtegy)
        for i in range(len(choose_neuron_region4)):
            layer_name, neuron_idx = choose_neuron_region4[i]
            loss0_neuron = K.mean(model.get_layer(layer_name).output[..., neuron_idx])
            loss_neuron.append(loss0_neuron)

    if '5' in neuron_selection_area:
        neuron_region5 = [(layer_name, neuron_idx) for (layer_name, neuron_idx), region in neuron_region_dict.items() if
                          region == 5]
        region_neuron_num = len(neuron_region5)
        print(len(neuron_region5))
        if len(neuron_region5) < neuron_to_cover_num_eachstrtegy:
            choose_neuron_region5 = neuron_region5
            loss_neuron_layer_name = get_loss_neuron_as_mosttime(model, model_layer_times_NC,neuron_to_cover_num_eachstrtegy - len(neuron_region5))
            for key in loss_neuron_layer_name:
                choose_neuron_region5.append(key)
        else:
            choose_neuron_region5 = get_loss_neuron_as_mosttime(model,get_neuron_cover_time(neuron_region5, model_layer_times_NC),neuron_to_cover_num_eachstrtegy)
        for i in range(len(choose_neuron_region5)):
            layer_name, neuron_idx = choose_neuron_region5[i]
            loss0_neuron = K.mean(model.get_layer(layer_name).output[..., neuron_idx])
            loss_neuron.append(loss0_neuron)
    print(len(loss_neuron))
    return loss_neuron, region_neuron_num

def neuron_selection_astime_random(model,neuron_region_dict,neuron_selection_area,model_layer_times_NC,neuron_to_cover_num):
    region_neuron_num = 0
    if neuron_selection_area == None:
        return random_strategy(model,model_layer_times_NC,neuron_to_cover_num)
    num_strategy = len([x for x in neuron_selection_area if x in ['1','2','3','4','5']])
    neuron_to_cover_num_eachstrtegy = int(neuron_to_cover_num / num_strategy)

    loss_neuron = []

    neuron_region = [(layer_name, neuron_idx) for (layer_name, neuron_idx), region in neuron_region_dict.items() if
                     region == int(neuron_selection_area[1])]
    #print('neuron_selection_area[1]' + str(neuron_selection_area[1]))
    region_neuron_num = len(neuron_region)
    print(region_neuron_num)
    i = 0
    neurons_covered_times = []
    neurons_key_pos = {}
    if region_neuron_num == 0:
        for (layer_name, neuron_idx), time in model_layer_times_NC.items():
            neurons_covered_times.append(time)
            neurons_key_pos[i] = (layer_name, neuron_idx)
            i += 1
    else:
        for (layer_name, neuron_idx) in neuron_region:
            time = model_layer_times_NC[(layer_name, neuron_idx)]
            neurons_covered_times.append(time)
            neurons_key_pos[i] = (layer_name, neuron_idx)
            i += 1
    neurons_covered_times = np.asarray(neurons_covered_times)  # 此时的 neurons_covered_times为[0 1 2 3]这种形式  这里是按层展开将一个个神经元曾经被覆盖次数列成一个数组
    times_total = sum(neurons_covered_times)
    if times_total == 0:
        #if region_neuron_num < neuron_to_cover_num_eachstrtegy:
            #num_neuron = np.random.choice(range(len(neurons_covered_times)), region_neuron_num, replace=False, )
        #else:
        num_neuron = np.random.choice(range(len(neurons_covered_times)), neuron_to_cover_num_eachstrtegy,replace=False, )
    else:
        neurons_covered_times_inverse = np.subtract(max(neurons_covered_times),neurons_covered_times)  # 按层展开每个神经元与最多被覆盖次数的神经元被覆盖次数差
        neurons_covered_percentage_inverse = neurons_covered_times_inverse / float(sum(neurons_covered_times_inverse))
        # num_neuron1 = np.random.choice(range(len(neurons_covered_times)), p=neurons_covered_percentage_inverse)
        num_neuron = np.random.choice(range(len(neurons_covered_times)), neuron_to_cover_num_eachstrtegy, replace=False,p=neurons_covered_percentage_inverse)
    for num in num_neuron:
        (layer_name1, index1) = neurons_key_pos[num]
        loss1_neuron = K.mean(model.get_layer(layer_name1).output[..., index1])
        loss_neuron.append(loss1_neuron)
    if len(loss_neuron) == 0:
        return  random_strategy(model,model_layer_times_NC,1) , region_neuron_num
    print(len(loss_neuron))
    return loss_neuron , region_neuron_num

def neuron_selection_asmosttime_random(model,neuron_region_dict,neuron_selection_area,model_layer_times_NC,neuron_to_cover_num):
    if neuron_selection_area == None:
        return random_strategy(model,model_layer_times_NC,neuron_to_cover_num)
    num_strategy = len([x for x in neuron_selection_area if x in ['1','2','3','4','5']])
    neuron_to_cover_num_eachstrtegy = int(neuron_to_cover_num / num_strategy)

    loss_neuron = []

    neuron_region = [(layer_name, neuron_idx) for (layer_name, neuron_idx), region in neuron_region_dict.items() if
                    region == int(neuron_selection_area[1])]
    region_neuron_num = len(neuron_region)
    print(region_neuron_num)
    i = 0
    neurons_covered_times = []
    neurons_key_pos = {}
    flag = False
    num = 0
    if region_neuron_num == 0:
        for (layer_name, neuron_idx), time in model_layer_times_NC.items():
            neurons_covered_times.append(time)
            neurons_key_pos[i] = (layer_name, neuron_idx)
            i += 1
    else:
        for (layer_name, neuron_idx) in neuron_region:
            time = model_layer_times_NC[(layer_name, neuron_idx)]
            neurons_covered_times.append(time)
            neurons_key_pos[i] = (layer_name, neuron_idx)
            i += 1
    neurons_covered_times = np.asarray(neurons_covered_times)  # 此时的 neurons_covered_times为[0 1 2 3]这种形式  这里是按层展开将一个个神经元曾经被覆盖次数列成一个数组
    times_total = sum(neurons_covered_times)
    if times_total == 0:
        # if region_neuron_num < neuron_to_cover_num_eachstrtegy:
        # num_neuron = np.random.choice(range(len(neurons_covered_times)), region_neuron_num, replace=False, )
        # else:
        num_neuron = np.random.choice(range(len(neurons_covered_times)), neuron_to_cover_num_eachstrtegy,replace=False, )
        #num_neuron = random_strategy(model,model_layer_times_NC,1)
    else:
        neurons_covered_percentage = neurons_covered_times / float(times_total)
        for i in neurons_covered_percentage:
            if i > 0:
                num += 1
        print(num)
        # num_neuron1 = np.random.choice(range(len(neurons_covered_times)), p=neurons_covered_percentage_inverse)
        if num < neuron_to_cover_num_eachstrtegy:
            num_neuron = np.random.choice(range(len(neurons_covered_times)), neuron_to_cover_num_eachstrtegy,
                                          replace=False,)
        else:
            num_neuron = np.random.choice(range(len(neurons_covered_times)), neuron_to_cover_num_eachstrtegy, replace=False, p=neurons_covered_percentage)
    for num in num_neuron:
        (layer_name1, index1) = neurons_key_pos[num]
        loss1_neuron = K.mean(model.get_layer(layer_name1).output[..., index1])
        loss_neuron.append(loss1_neuron)
    if len(loss_neuron) == 0:
        return random_strategy(model, model_layer_times_NC, 1), region_neuron_num
    print(len(loss_neuron))
    return loss_neuron, region_neuron_num

def neuron_selection_DLFuzz(model, model_layer_times, model_layer_value, neuron_select_strategy, neuron_to_cover_num, threshold):
    if neuron_select_strategy == 'None':
        return random_strategy(model, model_layer_times, neuron_to_cover_num)

    num_strategy = len([x for x in neuron_select_strategy if x in ['0', '1', '2', '3']])  #num_strategy存放选择神经元选择策略的策略数
    neuron_to_cover_num_each = neuron_to_cover_num / num_strategy  #平均每个策略应该去覆盖几个神经元

    loss_neuron = []
    # initialization for strategies
    if ('0' in list(neuron_select_strategy)) or ('1' in list(neuron_select_strategy)):
        i = 0
        neurons_covered_times = []
        neurons_key_pos = {}
        for (layer_name, index), time in model_layer_times.items():  #将model_layer_times.items中(layer_name, index) 和times分开存放
            if 'batch' in layer_name or 'activation' in layer_name:
                continue
            neurons_covered_times.append(time)  #此时的neurons_covered_times为[0,1,2,3]这种形式
            neurons_key_pos[i] = (layer_name, index)
            i += 1
        neurons_covered_times = np.asarray(neurons_covered_times) #此时的 neurons_covered_times为[0 1 2 3]这种形式  这里是按层展开将一个个神经元曾经被覆盖次数列成一个数组
        times_total = sum(neurons_covered_times) #总共覆盖了多少个神经元次数

    # select neurons covered often
    if '0' in list(neuron_select_strategy):  #选择经常或很少被覆盖的神经元
        if times_total == 0: #还未覆盖任何神经元 则就需要一开始随机选取一个神经元覆盖 否则首先计算神经元覆盖比例
            return random_strategy(model, model_layer_times, 1)#The beginning of no neurons covered
        neurons_covered_percentage = neurons_covered_times / float(times_total) #这是每个神经元对应的覆盖次数占总次数的百分比 是一个一位数组
        # num_neuron0 = np.random.choice(range(len(neurons_covered_times)), p=neurons_covered_percentage)
        num_neuron0 = np.random.choice(range(len(neurons_covered_times)), int(neuron_to_cover_num_each), replace=True, p=neurons_covered_percentage)  #采样的数量是neuron_to_cover_num_each  replace指定为True时，采样的元素会有重复；当replace指定为False时，采样不会重复。 p : 一个一维数组，制定了a中每个元素采样的概率，若为默认的None，则a中每个元素被采样的概率相同   所以覆盖次数高的被采样几率大
        for num in num_neuron0:
            layer_name0, index0 = neurons_key_pos[num]
            #print((layer_name0,index0))
            loss0_neuron = K.mean(model.get_layer(layer_name0).output[..., index0])
            loss_neuron.append(loss0_neuron)

    # select neurons covered rarely
    if '1' in list(neuron_select_strategy):
        if times_total == 0:
            return random_strategy(model, model_layer_times, 1)
        neurons_covered_times_inverse = np.subtract(max(neurons_covered_times), neurons_covered_times)  #按层展开每个神经元与最多被覆盖次数的神经元被覆盖次数差
        neurons_covered_percentage_inverse = neurons_covered_times_inverse / float(sum(neurons_covered_times_inverse))
        # num_neuron1 = np.random.choice(range(len(neurons_covered_times)), p=neurons_covered_percentage_inverse)
        num_neuron1 = np.random.choice(range(len(neurons_covered_times)), int(neuron_to_cover_num_each), replace=False,
                                       p=neurons_covered_percentage_inverse)  #与最多覆盖次数离的越远 越可能被采样到 则说明这里是选择相对来说较少被覆盖的神经元
        for num in num_neuron1:
            layer_name1, index1 = neurons_key_pos[num]
            loss1_neuron = K.mean(model.get_layer(layer_name1).output[..., index1])
            loss_neuron.append(loss1_neuron)

    # select neurons with largest weights (feature maps with largest filter weights)
    if '2' in list(neuron_select_strategy):
        layer_names = [layer.name for layer in model.layers if
                       'flatten' not in layer.name and 'input' not in layer.name ]# and 'batch' not in layer.name and 'activation' not in layer.name]
        k = 0.1
        top_k = k * len(model_layer_times)  # number of neurons to be selected within
        global model_layer_weights_top_k  #model_layer_weights_top_k中存放的是layer_name, index
        if len(model_layer_weights_top_k) == 0:
            neuron_select_high_weight(model, layer_names, top_k)  # Set the value

        num_neuron2 = np.random.choice(range(len(model_layer_weights_top_k)), int(neuron_to_cover_num_each),replace=True )#replace=False
        for i in num_neuron2:
            # i = np.random.choice(range(len(model_layer_weights_top_k)))
            layer_name2 = model_layer_weights_top_k[i][0]
            index2 = model_layer_weights_top_k[i][1]
            loss2_neuron = K.mean(model.get_layer(layer_name2).output[..., index2])
            loss_neuron.append(loss2_neuron)

    if '3' in list(neuron_select_strategy):
        above_threshold = []
        below_threshold = []
        above_num = neuron_to_cover_num_each / 2
        below_num = neuron_to_cover_num_each - above_num #奇怪这两个不是相等吗
        above_i = 0
        below_i = 0
        for (layer_name, index), value in model_layer_value.items():
            if threshold + 0.25 > value > threshold and layer_name != 'fc1' and layer_name != 'fc2' and \
                    layer_name != 'predictions' and layer_name != 'fc1000' and layer_name != 'before_softmax' \
                    and above_i < above_num:
                above_threshold.append([layer_name, index])
                above_i += 1
                # print(layer_name,index,value)
                # above_threshold_dict[(layer_name, index)]=value
            elif threshold > value > threshold - 0.2 and layer_name != 'fc1' and layer_name != 'fc2' and \
                    layer_name != 'predictions' and layer_name != 'fc1000' and layer_name != 'before_softmax' \
                    and below_i < below_num:
                below_threshold.append([layer_name, index])
                below_i += 1
        #
        # loss3_neuron_above = 0
        # loss3_neuron_below = 0
        loss_neuron = []
        if len(above_threshold) > 0:
            for above_item in range(len(above_threshold)):
                loss_neuron.append(K.mean(
                    model.get_layer(above_threshold[above_item][0]).output[..., above_threshold[above_item][1]]))

        if len(below_threshold) > 0: #低于激活值的loss_neuron是负的K.mean
            for below_item in range(len(below_threshold)):
                loss_neuron.append(-K.mean(
                    model.get_layer(below_threshold[below_item][0]).output[..., below_threshold[below_item][1]]))

        # loss_neuron += loss3_neuron_below - loss3_neuron_above

        # for (layer_name, index), value in model_layer_value.items():
        #     if 0.5 > value > 0.25:
        #         above_threshold.append([layer_name, index])
        #     elif 0.25 > value > 0.2:
        #         below_threshold.append([layer_name, index])
        # loss3_neuron_above = 0
        # loss3_neuron_below = 0
        # if len(above_threshold)>0:
        #     above_i = np.random.choice(range(len(above_threshold)))
        #     loss3_neuron_above = K.mean(model.get_layer(above_threshold[above_i][0]).output[..., above_threshold[above_i][1]])
        # if len(below_threshold)>0:
        #     below_i = np.random.choice(range(len(below_threshold)))
        #     loss3_neuron_below = K.mean(model.get_layer(below_threshold[below_i][0]).output[..., below_threshold[below_i][1]])
        # loss_neuron += loss3_neuron_below - loss3_neuron_above
        if loss_neuron == 0:
            return random_strategy(model, model_layer_times, 1)  # The beginning of no neurons covered

    return loss_neuron

def neuron_selection_DeepXplore(model,model_layer_times,neuron_to_cover_num):
    loss_neuron = []
    num = 0
    while num < neuron_to_cover_num:
        layer_name,neuron_idx = neuron_to_cover_DeepXplore(model_layer_times)
        if 'batch' not in layer_name and 'activation' not in layer_name:
            loss0_neuron = K.mean(model.get_layer(layer_name).output[...,neuron_idx])
            loss_neuron.append(loss0_neuron)
            num += 1
    print(len(loss_neuron))
    return loss_neuron

def neuron_selection_Random(model,model_layer_times,neuron_to_cover_num):
    loss_neuron = []
    num = 0
    model_layer_times_key = [(layer_name,neuron_idx) for (layer_name,neuron_idx),v in model_layer_times.items() if 'batch' not in layer_name and 'activation' not in layer_name]
    while num < neuron_to_cover_num:
        layer_name, neuron_idx = random.choice(model_layer_times_key)
        loss0_neuron = K.mean(model.get_layer(layer_name).output[...,neuron_idx])
        loss_neuron.append(loss0_neuron)
        num += 1
    print(len(loss_neuron))
    return loss_neuron

def sort_img(img_list,model):
    pred_list = []
    img_list_sort = []
    for i in range(len(img_list)):
        img = img_list[i]
        img_pred = model.predict(img)
        img_label = np.argmax(img_pred[0])
        pred = img_pred[0][img_label]
        pred_list.append(pred)
    pred_list_sort = np.argsort(pred_list)
    for j in range(len(img_list)):
        img_list_sort.append(img_list[pred_list_sort[j]])
    return  img_list_sort

def choose_seed(img_dir,save_dir,model,choose_seeds_num):
    creat_path(save_dir)
    pred_list = []
    img_path_sort_list = []
    img_paths = os.listdir(img_dir)
    img_num = len(img_paths)
    for i in range(img_num):
        img_path = os.path.join(img_dir, img_paths[i])
        img_name = img_paths[i].split('.')[0]
        img = preprocess_image(img_path)
        img_pred = model.predict(img)
        img_label = np.argmax(img_pred[0])
        pred = img_pred[0][img_label]
        pred_list.append(pred)
        pred_list_sort = np.argsort(pred_list)
    for j in range(img_num):
        img_path_sort_list.append(img_paths[pred_list_sort[j]])
    for k in range(choose_seeds_num):
        img_choose = img_path_sort_list[k]
        img_choose_path = os.path.join(img_dir,img_choose)
        #img_choose_name = img_choose.split('.')[0]
        shutil.copy(img_choose_path,save_dir)

def choose_seed_random(img_dir,save_dir,model,choose_seeds_num):
    creat_path(save_dir)
    img_paths = os.listdir(img_dir)
    list = random.sample(img_paths,choose_seeds_num)
    for i in range(choose_seeds_num):
        img__path = os.path.join(img_dir, list[i])
        shutil.copy(img__path,save_dir)