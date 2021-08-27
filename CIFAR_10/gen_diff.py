# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as  np
from keras.layers import Input
from keras.models import load_model
import  imageio
imsave = imageio.imsave
import sys
import os
import time
from keras import backend as K
from CIFAR_10.utils import *



def gen_diff_region_effect(model_name,criterion,criterion_para,neuron_selection_area,subdir,iteration_times,neuron_to_cover_num,seed_selection_strategy,loss_neuron_strategy):
    if model_name == 'resnet20':
        model = load_model('./data/models/resnet.h5')
        before_softmax = 'dense_1'
    else:
        print('please specify model name')
        os._exit(0)
    print(model_name)
    neuron_to_cover_weight = 0.7
    predict_weight = 0.3
    learning_step = 0.02
    model_layer_neuron_value = init_layer_neuron_value(model)
    profile_dict = {}
    region_neuron_num_min = 100000
    region_neuron_num_max = 0
    img_rows, img_cols = 32, 32
    input_shape = (img_rows, img_cols, 3)
    input_tensor = Input(shape=input_shape)
    choose_save_dir = './choose_seeds_20'
    if seed_selection_strategy == 0:
        choose_seed_random('./seeds_50_1', choose_save_dir, model, 20)
    elif seed_selection_strategy == 1:
        choose_seed('./seeds_50_1', choose_save_dir, model, 20)
    img_dir = choose_save_dir
    img_paths = os.listdir(img_dir)
    img_num = len(img_paths)
    model_layer_times2 = init_coverage_times(model,criterion,criterion_para)
    model_layer_times1 = init_coverage_times(model,criterion,criterion_para)
    model_layer_times_NC1 = init_coverage_times_NC(model)
    save_dir = './generated_inputs/' + subdir + '/'
    creat_path(save_dir)

    total_time = 0
    total_norm = 0
    adversarial_num = 0

    total_perturb_adversial = 0

    for i in range(img_num):
        start_time = time.clock()
        img_list = []
        img_path = os.path.join(img_dir,img_paths[i])
        img_name = img_paths[i].split('.')[0]
        mannual_label = int(img_name.split('_')[0])
        print(img_path)
        tmp_img = preprocess_image(img_path)
        ori_img = tmp_img.copy()
        img_list.append(tmp_img)
        update_coverage(tmp_img,model,model_name,model_layer_times2,criterion,criterion_para)  #return intermediate_layer_outputs  intermediate_layer_outputs[0].shape=(1,28,28,4)
        while len(img_list) > 0:
            #img_list = sort_img(img_list,model)
            gen_img = img_list[0]
            img_list.remove(gen_img)

            ori_pred = model.predict(gen_img)    #ori_pred.shape=(1,10)
            #print('ori_pred')
            #print(ori_pred)
            ori_label = np.argmax(ori_pred[0])
            print('ori_label:  '+ str(ori_label))

            label_top5 = np.argsort(ori_pred[0])[-5:]   #argsort is ascending

            #update_neuron_value(gen_img,model,model_layer_neuron_value)
            #if criterion != 'NC':
            update_NC_coverage(gen_img,model,model_layer_times_NC1,0.65)

            #print('每层属于选择区域的神经元个数：')
            #print(get_layer_region_neuron_num(gen_img, model, model_name, neuron_select_area).items())
            update_coverage(gen_img,model,model_name,model_layer_times1,criterion,criterion_para)


            loss_1 = K.mean(model.get_layer(before_softmax).output[...,ori_label])
            loss_2 = K.mean(model.get_layer(before_softmax).output[...,label_top5[-2]])
            loss_3 = K.mean(model.get_layer(before_softmax).output[...,label_top5[-3]])
            loss_4 = K.mean(model.get_layer(before_softmax).output[...,label_top5[-4]])
            loss_5 = K.mean(model.get_layer(before_softmax).output[..., label_top5[-5]])

            layer_output = (predict_weight * (loss_2 + loss_3 + loss_4 + loss_5) - loss_1)
            if loss_neuron_strategy == 0:
                loss_neuron = neuron_selection(model,get_neuron_output_region(gen_img,model,model_name),neuron_selection_area,model_layer_times_NC1,neuron_to_cover_num)[0]
            elif loss_neuron_strategy == 1:
                loss_neuron = neuron_selection_addtime(model,get_neuron_output_region(gen_img,model,model_name),neuron_selection_area,model_layer_times_NC1,neuron_to_cover_num)[0]
            elif loss_neuron_strategy == 2:
                loss_neuron = neuron_selection_addmosttime(model, get_neuron_output_region(gen_img, model, model_name), neuron_selection_area,model_layer_times_NC1, neuron_to_cover_num)[0]
            elif loss_neuron_strategy == 3:
                loss_neuron = neuron_selection_astime_random(model, get_neuron_output_region(gen_img, model, model_name),neuron_selection_area, model_layer_times_NC1,neuron_to_cover_num)[0]
            elif loss_neuron_strategy == 4:
                loss_neuron = neuron_selection_asmosttime_random(model, get_neuron_output_region(gen_img, model, model_name),neuron_selection_area, model_layer_times_NC1, neuron_to_cover_num)[0]

            #loss_neuron = neuron_selection_layer(gen_img,model,model_name,neuron_select_area,neuron_to_cover_num)
            #loss_neuron = neuron_selection_upper(model,model_name,neuron_to_cover_num)
            #loss_neuron = neuron_selection_maxmin(gen_img,model)
            #region_neuron_num = neuron_selection(model,get_neuron_output_region(gen_img,model,model_name),neuron_select_area,model_layer_times_NC1,neuron_to_cover_num)[1]

            #if region_neuron_num < region_neuron_num_min:
                #region_neuron_num_min = region_neuron_num
            #if region_neuron_num > region_neuron_num_max:
                #region_neuron_num_max = region_neuron_num
            ## extreme value means the activation value for a neuron can be as high as possible ...
            EXTREME_VALUE = False
            if EXTREME_VALUE:
                neuron_to_cover_weight = 2

            layer_output += neuron_to_cover_weight * K.sum(loss_neuron)

            final_loss = K.mean(layer_output)
            input_tensor = model.input
            #print(type(K.gradients(final_loss,input_tensor)[0]))
            grads = normalize(K.gradients(final_loss,input_tensor)[0])

            grads_tensor_list = [loss_1 , loss_2 , loss_3 , loss_4 , loss_5]
            grads_tensor_list.extend(loss_neuron)
            grads_tensor_list.append(grads)

            iterate = K.function([input_tensor] , grads_tensor_list)

            for iter in range(iteration_times):

                loss_neuron_list = iterate([gen_img])

                perturb = loss_neuron_list[-1] * learning_step

                gen_img += perturb

                previous_coverage = neuron_covered_num(criterion,model_layer_times1)[2]

                pred1 = model.predict(gen_img)
                label1 = np.argmax(pred1[0])
                print('label1 : ' +str(label1))

                #if criterion != 'NC':
                update_NC_coverage(gen_img,model,model_layer_times_NC1,0.65)
                #print('iter:'+str(iter)+ ' 每层属于所选区域神经元个数')
                #print(get_layer_region_neuron_num(gen_img, model, model_name, neuron_select_area).items())
                update_coverage(gen_img,model,model_name,model_layer_times1,criterion,criterion_para)

                current_coverage = neuron_covered_num(criterion,model_layer_times1)[2]

                diff_img = gen_img - ori_img

                diff_img_L2 = np.linalg.norm(diff_img)

                ori_img_L2 = np.linalg.norm(ori_img)

                perturb_adversial = diff_img_L2 / ori_img_L2
                #这里和DLFuzz不同
                #print('迭代次数：'+str(iter)+'label1:'+str(label1))
                if label1 == ori_label:
                    if criterion == 'NC':
                        if current_coverage - previous_coverage >  0.01 / (i + 1) and perturb_adversial < 0.02:
                            img_list.append(gen_img)
                            print('img_list+1')
                    elif criterion == 'KMNC':
                        if current_coverage - previous_coverage > 0.01 / (i + 1) and perturb_adversial < 0.02:
                            img_list.append(gen_img)
                            print('img_list+1')
                    elif criterion == 'NBC':
                        if current_coverage - previous_coverage > 0 and perturb_adversial < 0.02:
                            img_list.append(gen_img)
                            print('img_list+1')
                    elif criterion == 'SNAC':
                        if current_coverage - previous_coverage > 0 and perturb_adversial < 0.02:
                            img_list.append(gen_img)
                            print('img_list+1')
                    elif criterion == 'TKNC':
                        if current_coverage - previous_coverage > 0 and perturb_adversial < 0.02:
                            img_list.append(gen_img)
                            print('img_list+1')

                else:
                    #print('ori_img coverage:'+ str(neuron_covered_num(criterion,model_layer_times2)[2]))
                    update_coverage(gen_img,model,model_name,model_layer_times2,criterion,criterion_para)

                    total_norm += diff_img_L2
                    total_perturb_adversial += perturb_adversial

                    gen_adv_img = gen_img.copy()

                    gen_adv_img_deprocessed = deprocess_image(gen_adv_img)

                    save_img = save_dir + img_name + '_' + str(label1) + '_' + str(adversarial_num+1) +'.png'

                    imsave(save_img,gen_adv_img_deprocessed)

                    adversarial_num += 1
        end_time = time.clock()

        print('Total neuron : %d ,  Neuron coverage : %.3f' %(len(model_layer_times2) , neuron_covered_num(criterion,model_layer_times2)[2]))

        duration = end_time - start_time

        print('Used time : '+ str(duration))

        total_time += duration

    neuron_coverage = neuron_covered_num(criterion, model_layer_times2)[2]
    adversarial_num_all = adversarial_num
    return neuron_coverage, adversarial_num_all

if __name__ == '__main__':
    neuron_selection_area_list = ['[4]','[2]','[4]','[4]','[2]']
    criterion_list = ['NC','KMNC','NBC','SNAC','TKNC']
    model_name = 'resnet20'
    #criterion = 'KMNC'
    criterion_para_list = [0.65,1000,0,0,10]
    #for neuron_selection_area1 in neuron_selection_area_list:
    #neuron_selection_area = neuron_selection_area1
    #neuron_selection_area = '[2]'
    iteration_times = 5
    neuron_to_cover_num = 10
    for i in range(1):
        criterion = criterion_list[1]
        criterion_para = criterion_para_list[1]
        neuron_selection_area = neuron_selection_area_list[1]
        for j in range(1):
            subdir = criterion + '_' + str(4)
            neuron_coverage,adversarial_nums = gen_diff_region_effect(model_name,criterion,criterion_para,neuron_selection_area,subdir,iteration_times,neuron_to_cover_num,1,4)
            print(criterion + '_' + str(4) + '_' + 'Neuron Coverage :  ' + str(neuron_coverage))
            print(criterion + '_' + str(4) + '_' + 'Adversarial nums :  ' + str(adversarial_nums))
