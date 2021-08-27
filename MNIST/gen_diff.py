from __future__ import print_function
import numpy as  np
from keras.layers import Input
from keras import backend as K
from keras.models import load_model
import imageio
imsave = imageio.imsave
import sys
import os
import time
from datetime import  datetime
#from MNIST import Model1
from MNIST.utils import *

model_name_list = ['lenet1','lenet4','lenet5']
criterion_list = ['NC','KMNC','NBC','SNAC','TKNC']
criterion_para_list = [0.5,1000,0,0,3]

class gen_diff_region_effect():
    def __init__(self,model_name,criterion,criterion_para,neuron_select_area,subdir,iteration_times,neuron_to_cover_num,seed_selection_strategy):
        self.model_name = model_name
        self.crierion = criterion
        self.crierion_para = criterion_para
        self.neuron_selection_area = neuron_select_area
        self.subdir = subdir
        self.iteration_times = iteration_times
        self.neuron_to_cover_num = neuron_to_cover_num
        if model_name == 'lenet1':
            model = load_model('./data/models/lenet1.h5')
            before_softmax = 'dense_1'
        elif model_name == 'lenet4':
            model = load_model('./data/models/lenet4.h5')
            before_softmax = 'dense_2'
        elif model_name == 'lenet5':
            model = load_model('./data/models/lenet5.h5')
            before_softmax = 'dense_3'
        else:
            print('please specify model name')
            os._exit(0)
        print(model_name)
        neuron_to_cover_weight = 0.7
        predict_weight = 0.3
        learning_step = 0.02
        model_layer_neuron_value = init_layer_neuron_value(model)
        profile_dict = {}
        region_neuron_num_min = 1000
        region_neuron_num_max = 0
        model_layer_times2 = init_coverage_times(model, criterion, criterion_para)
        model_layer_times1 = init_coverage_times(model, criterion, criterion_para)
        model_layer_times_NC1 = init_coverage_times_NC(model)
        img_rows, img_cols = 28, 28
        input_shape = (img_rows, img_cols, 1)
        input_tensor = Input(shape=input_shape)
        K.set_learning_phase(0)
        choose_save_dir = './choose_seeds_20'
        if seed_selection_strategy == 0:
            choose_seed_random('./seeds_50', choose_save_dir, model, 20)
        elif seed_selection_strategy ==1:
            choose_seed('./seeds_50', choose_save_dir, model, 20)
        img_dir = choose_save_dir
        img_paths = os.listdir(img_dir)
        img_num = len(img_paths)
        save_dir = './generated_inputs/' + subdir + '/'
        creat_path(save_dir)
        total_time = 0
        total_norm = 0
        adversarial_num = 0
        total_perturb_adversial = 0
        for i in range(img_num):
            start_time = time.clock()
            img_list = []
            img_path = os.path.join(img_dir, img_paths[i])
            img_name = img_paths[i].split('.')[0]
            mannual_label = int(img_name.split('_')[1])
            print(img_path)
            tmp_img = preprocess_image(img_path)
            ori_img = tmp_img.copy()
            img_list.append(tmp_img)
            update_coverage(tmp_img, model, model_name, model_layer_times2, criterion,criterion_para)  # return intermediate_layer_outputs  intermediate_layer_outputs[0].shape=(1,28,28,4)
            while len(img_list) > 0:
                # img_list = sort_img(img_list,model)
                gen_img = img_list[0]  # 这里之后要加种子排序
                img_list.remove(gen_img)

                ori_pred = model.predict(gen_img)  # ori_pred.shape=(1,10)
                ori_label = np.argmax(ori_pred[0])

                label_top5 = np.argsort(ori_pred[0])[-5:]  # argsort is ascending

                # update_neuron_value(gen_img,model,model_layer_neuron_value)

                update_NC_coverage(gen_img, model, model_layer_times_NC1, 0.5)


                update_coverage(gen_img, model, model_name, model_layer_times1, criterion, criterion_para)

                loss_1 = K.mean(model.get_layer(before_softmax).output[..., ori_label])
                loss_2 = K.mean(model.get_layer(before_softmax).output[..., label_top5[-2]])
                loss_3 = K.mean(model.get_layer(before_softmax).output[..., label_top5[-3]])
                loss_4 = K.mean(model.get_layer(before_softmax).output[..., label_top5[-4]])
                loss_5 = K.mean(model.get_layer(before_softmax).output[..., label_top5[-5]])

                layer_output = (predict_weight * (loss_2 + loss_3 + loss_4 + loss_5) - loss_1)

                loss_neuron = neuron_selection(model,get_neuron_output_region(gen_img,model,model_name),neuron_select_area,model_layer_times_NC1,neuron_to_cover_num)[0]
                region_neuron_num = neuron_selection(model,get_neuron_output_region(gen_img,model,model_name),neuron_select_area,model_layer_times_NC1,neuron_to_cover_num)[1]

                if region_neuron_num < region_neuron_num_min:
                    region_neuron_num_min = region_neuron_num
                if region_neuron_num > region_neuron_num_max:
                    region_neuron_num_max = region_neuron_num
                ## extreme value means the activation value for a neuron can be as high as possible ...
                EXTREME_VALUE = False
                if EXTREME_VALUE:
                    neuron_to_cover_weight = 2

                layer_output += neuron_to_cover_weight * K.sum(loss_neuron)

                final_loss = K.mean(layer_output)
                input_tensor = model.input
                # print(type(K.gradients(final_loss,input_tensor)[0]))
                grads = normalize(K.gradients(final_loss, input_tensor)[0])

                grads_tensor_list = [loss_1, loss_2, loss_3, loss_4, loss_5]
                grads_tensor_list.extend(loss_neuron)
                grads_tensor_list.append(grads)

                iterate = K.function([input_tensor], grads_tensor_list)

                for iter in range(iteration_times):

                    loss_neuron_list = iterate([gen_img])

                    perturb = loss_neuron_list[-1] * learning_step

                    gen_img += perturb

                    previous_coverage = neuron_covered_num(criterion, model_layer_times1)[2]

                    pred1 = model.predict(gen_img)
                    label1 = np.argmax(pred1[0])

                    #if criterion != 'NC':
                    update_NC_coverage(gen_img, model, model_layer_times_NC1, 0.5)

                    update_coverage(gen_img, model, model_name, model_layer_times1, criterion, criterion_para)

                    current_coverage = neuron_covered_num(criterion, model_layer_times1)[2]

                    diff_img = gen_img - ori_img

                    diff_img_L2 = np.linalg.norm(diff_img)

                    ori_img_L2 = np.linalg.norm(ori_img)

                    perturb_adversial = diff_img_L2 / ori_img_L2
                    # 这里和DLFuzz不同
                    #print('迭代次数：' + str(iter) + 'label1:' + str(label1))
                    if label1 == ori_label:
                        if criterion == 'NC':
                            if current_coverage - previous_coverage > 0.01 / (i + 1) and perturb_adversial < 0.02:
                                img_list.append(gen_img)
                                print('img_list+1')
                        elif criterion == 'KMNC':
                            if current_coverage - previous_coverage > 0 and perturb_adversial < 0.02:
                                img_list.append(gen_img)
                        elif criterion == 'NBC':
                            if current_coverage - previous_coverage > 0 and perturb_adversial < 0.02:
                                img_list.append(gen_img)
                        elif criterion == 'SNAC':
                            if current_coverage - previous_coverage > 0 and perturb_adversial < 0.02:
                                img_list.append(gen_img)
                        elif criterion == 'TKNC':
                            if current_coverage - previous_coverage > 0 and perturb_adversial < 0.02:
                                img_list.append(gen_img)

                    else:
                        #print('ori_img coverage:' + str(neuron_covered_num(criterion, model_layer_times2)[2]))
                        update_coverage(gen_img, model, model_name, model_layer_times2, criterion, criterion_para)

                        total_norm += diff_img_L2
                        total_perturb_adversial += perturb_adversial

                        gen_adv_img = gen_img.copy()

                        gen_adv_img_deprocessed = deprocess_image(gen_adv_img)

                        save_img = save_dir + img_name + '_' + str(label1) + '_' + str(adversarial_num + 1) + '.png'

                        imsave(save_img, gen_adv_img_deprocessed)

                        adversarial_num += 1
            end_time = time.clock()

            print('Total neuron : %d ,  Neuron coverage : %.3f' % (
            len(model_layer_times2), neuron_covered_num(criterion, model_layer_times2)[2]))

            duration = end_time - start_time

            print('Used time : ' + str(duration))

            total_time += duration


        neuron_coverage = neuron_covered_num(criterion , model_layer_times2)[2]
        adversarial_num_all = adversarial_num
        self.result = [neuron_coverage , adversarial_num_all]


class gen_adv_DLRegion():
    def __init__(self,model_name,criterion,criterion_para,subdir,iteration_times,neuron_to_cover_num,seed_selection_strategy):
        self.mode_name = model_name
        self.model_name = model_name
        self.criterion = criterion
        self.criterion_para = criterion_para
        self.subdir = subdir
        self.iteration_times = iteration_times
        self.neuron_to_cover_num = neuron_to_cover_num
        img_rows, img_cols = 28, 28
        input_shape = (img_rows, img_cols, 1)
        input_tensor = Input(shape=input_shape)
        K.set_learning_phase(0)
        if model_name == 'lenet1':
            model = load_model('./data/models/lenet1.h5')
            before_softmax = 'dense_1'
        elif model_name == 'lenet4':
            model = load_model('./data/models/lenet4.h5')
            before_softmax = 'dense_2'
        elif model_name == 'lenet5':
            model = load_model('./data/models/lenet5.h5')
            before_softmax = 'dense_3'
        else:
            print('please specify model name')
            os._exit(0)
        print(model_name)
        profile_dict = {}
        region_neuron_num_min = 1000
        region_neuron_num_max = 0
        choose_save_dir = './choose_seeds_20'
        if seed_selection_strategy == 0:
            choose_seed_random('./seeds_50', choose_save_dir, model, 20)
        elif seed_selection_strategy == 1:
            choose_seed('./seeds_50', choose_save_dir, model, 20)
        img_dir = choose_save_dir
        img_paths = os.listdir(img_dir)
        img_num = len(img_paths)
        save_dir = './generated_inputs/' + subdir + '/'
        neuron_to_cover_weight = 0.7
        predict_weight = 0.3
        learning_step = 0.02
        neuron_selection_area = criterion_to_neuron_selection_area(model_name,criterion)
        model_layer_times2 = init_coverage_times(model, criterion, criterion_para)
        model_layer_times1 = init_coverage_times(model, criterion, criterion_para)
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
            img_path = os.path.join(img_dir, img_paths[i])
            img_name = img_paths[i].split('.')[0]
            mannual_label = int(img_name.split('_')[1])
            print(img_path)
            tmp_img = preprocess_image(img_path)
            ori_img = tmp_img.copy()
            img_list.append(tmp_img)
            update_coverage(tmp_img, model, model_name, model_layer_times2, criterion,criterion_para)  # return intermediate_layer_outputs  intermediate_layer_outputs[0].shape=(1,28,28,4)
            while len(img_list) > 0:
                gen_img = img_list[0]
                img_list.remove(gen_img)
                ori_pred = model.predict(gen_img)  # ori_pred.shape=(1,10)
                ori_label = np.argmax(ori_pred[0])

                label_top5 = np.argsort(ori_pred[0])[-5:]
                update_NC_coverage(gen_img, model, model_layer_times_NC1, 0.5)
                update_coverage(gen_img, model, model_name, model_layer_times1, criterion, criterion_para)

                loss_1 = K.mean(model.get_layer(before_softmax).output[..., ori_label])
                loss_2 = K.mean(model.get_layer(before_softmax).output[..., label_top5[-2]])
                loss_3 = K.mean(model.get_layer(before_softmax).output[..., label_top5[-3]])
                loss_4 = K.mean(model.get_layer(before_softmax).output[..., label_top5[-4]])
                loss_5 = K.mean(model.get_layer(before_softmax).output[..., label_top5[-5]])

                layer_output = (predict_weight * (loss_2 + loss_3 + loss_4 + loss_5) - loss_1)

                loss_neuron = criterion_to_neuron_selection_strategy(criterion,model,get_neuron_output_region(gen_img,model,model_name),neuron_selection_area,model_layer_times_NC1,neuron_to_cover_num)

                EXTREME_VALUE = False
                if EXTREME_VALUE:
                    neuron_to_cover_weight = 2

                layer_output += neuron_to_cover_weight * K.sum(loss_neuron)

                final_loss = K.mean(layer_output)
                input_tensor = model.input

                grads = normalize(K.gradients(final_loss, input_tensor)[0])

                grads_tensor_list = [loss_1, loss_2, loss_3, loss_4, loss_5]
                grads_tensor_list.extend(loss_neuron)
                grads_tensor_list.append(grads)

                iterate = K.function([input_tensor], grads_tensor_list)

                for iter in range(iteration_times):
                    loss_neuron_list = iterate([gen_img])

                    perturb = loss_neuron_list[-1] * learning_step * (random.random() + 0.5)

                    gen_img += perturb

                    previous_coverage = neuron_covered_num(criterion, model_layer_times1)[2]

                    pred1 = model.predict(gen_img)
                    label1 = np.argmax(pred1[0])

                    update_NC_coverage(gen_img, model, model_layer_times_NC1, 0.5)

                    update_coverage(gen_img, model, model_name, model_layer_times1, criterion, criterion_para)

                    current_coverage = neuron_covered_num(criterion, model_layer_times1)[2]

                    diff_img = gen_img - ori_img

                    diff_img_L2 = np.linalg.norm(diff_img)

                    ori_img_L2 = np.linalg.norm(ori_img)

                    perturb_adversial = diff_img_L2 / ori_img_L2

                    if label1 == ori_label:
                        if criterion == 'NC':
                            if current_coverage - previous_coverage > 0.01 / (i + 1) and perturb_adversial < 0.02:
                                img_list.append(gen_img)
                                print('img_list+1')
                        elif criterion == 'KMNC':
                            if current_coverage - previous_coverage > 0 and perturb_adversial < 0.02:
                                print('img_list+1')
                                img_list.append(gen_img)
                        elif criterion == 'NBC':
                            if current_coverage - previous_coverage > 0 and perturb_adversial < 0.02:
                                img_list.append(gen_img)
                        elif criterion == 'SNAC':
                            if current_coverage - previous_coverage > 0 and perturb_adversial < 0.02:
                                img_list.append(gen_img)
                        elif criterion == 'TKNC':
                            if current_coverage - previous_coverage > 0 and perturb_adversial < 0.02:
                                img_list.append(gen_img)
                    else:
                        update_coverage(gen_img, model, model_name, model_layer_times2, criterion, criterion_para)

                        total_norm += diff_img_L2
                        total_perturb_adversial += perturb_adversial

                        gen_adv_img = gen_img.copy()

                        gen_adv_img_deprocessed = deprocess_image(gen_adv_img)

                        save_img = save_dir + img_name + '_' + str(label1) + '_' + str(adversarial_num + 1) + '.png'

                        imsave(save_img, gen_adv_img_deprocessed)

                        adversarial_num += 1
            end_time = time.clock()
            print('Total neuron : %d ,  Neuron coverage : %.3f' % (len(model_layer_times2), neuron_covered_num(criterion, model_layer_times2)[2]))
            duration = end_time - start_time
            print('Used time : ' + str(duration))
            total_time += duration
        neuron_coverage = neuron_covered_num(criterion, model_layer_times2)[2]
        adversarial_num_all = adversarial_num
        used_time = total_time
        self.result =[neuron_coverage, adversarial_num_all,used_time]


if __name__ == '__main__':
    technique = sys.argv[1]
    if technique == 'DLRegion':
        run_times  = 3
        model_name = model_name_list[2]
        for criterion_idx in range(len(criterion_list)):
            if criterion_idx == 0:
                continue
            elif criterion_idx == 1:
                continue
            criterion = criterion_list[criterion_idx]
            criterion_para = criterion_para_list[criterion_idx]
            neuron_coverage_list = []
            adversarial_nums_list = []
            neuron_coverage_all = 0
            adversarial_nums_all = 0
            used_time_all = 0
            for i in range(run_times):
                neuron_coverage ,adversarial_nums , used_time = gen_adv_DLRegion(model_name,criterion,criterion_para,technique+'_'+model_name+'_'+criterion+'_'+str(i),5,5,0).result
                neuron_coverage_list.append(neuron_coverage)
                adversarial_nums_list.append(adversarial_nums)
                neuron_coverage_all += neuron_coverage
                adversarial_nums_all += adversarial_nums
                used_time_all += used_time
            neuron_coverage_average = neuron_coverage_all / run_times
            adversarial_nums_average = adversarial_nums_all / run_times
            used_time_average = used_time_all / run_times
            print(model_name+'_'+ criterion + '_'+ 'Seed_selection0_Neuron coverage list: ')
            print(neuron_coverage_list)
            print(model_name+'_'+ criterion + '_'+ 'Seed_selection0_Adversarial nums list:')
            print(adversarial_nums_list)
            print(model_name+'_'+ criterion + '_'+ 'Seed_selection0_Average neuron coverage by 3 times : ' + str(neuron_coverage_average))
            print(model_name+'_'+ criterion + '_'+ 'Seed_selection0_Average adversarial nums by 3 times : ' + str(adversarial_nums_average))
            print(model_name+'_'+ criterion + '_'+ 'Seed_selection0_Average used time by 3 times :' + str(used_time_average))
