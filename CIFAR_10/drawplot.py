from CIFAR_10.gen_diff import *
import matplotlib.pyplot as plt


model_name_list = ['resnet20']
criterion_list = ['NC','KMNC','NBC','SNAC','TKNC']
criterion_para_list = [0.65,1000,0,0,10]
neuron_selection_area_list = ['[1]','[2]','[3]','[4]','[5]']
seed_selection_strategy = [0,1]
neuron_coverage_model_dict = {}
adversarial_example_nums_model_dict = {}

def get_data(model_name,criterion,criterion_para,subdir,iteration_times,neuron_to_cover_num,seed_selection_strategy):
    run_times = 5
    adversarial_example_nums_list = []
    neuron_coverage_list = []
    for area in neuron_selection_area_list:
        if criterion == 'TKNC' and area == '[1]':
            neuron_coverage_list.append(0.571)
            adversarial_example_nums_list.append(99.0)
            continue
        if criterion == 'TKNC' and area == '[2]':
            neuron_coverage_list.append(0.607)
            adversarial_example_nums_list.append(95.4)
            continue
        if criterion == 'TKNC' and area == '[3]':
            neuron_coverage_list.append(0.590)
            adversarial_example_nums_list.append(96.0)
            continue
        print('choose area:'+' '+area)
        neuron_coverage_all = 0
        adversarial_example_nums_all = 0
        for i in range(run_times):
            neuron_coverage, adversarial_example_nums = gen_diff_region_effect(model_name,criterion,criterion_para,area,subdir+'_'+area+'_'+str(i),iteration_times,neuron_to_cover_num,seed_selection_strategy)
            neuron_coverage_all += neuron_coverage
            adversarial_example_nums_all += adversarial_example_nums
        neuron_coverage_average = neuron_coverage_all/run_times
        adversarial_example_nums_average = adversarial_example_nums_all/run_times
        print('Average'+ criterion + 'neuron coverage at ' + area + 'by 5 times :  ' + str(neuron_coverage_average))
        print('Average'+ criterion + ' adversarial example nums at ' + area + 'by 5 times :  ' + str(adversarial_example_nums_average))
        neuron_coverage_list.append(neuron_coverage_average)
        adversarial_example_nums_list.append(adversarial_example_nums_average)
    neuron_coverage_model_dict[(model_name,criterion)] = neuron_coverage_list
    adversarial_example_nums_model_dict[(model_name,criterion)] =  adversarial_example_nums_list

def coverage_plot(criterion):
    plt.plot(neuron_selection_area_list,neuron_coverage_model_dict[('resnet20',criterion)],label='resnet20',linestyle = '--',color = 'green')
    plt.legend()
    plt.xlabel('choose region')
    plt.ylabel(criterion)
    plt.title(criterion+' '+'coverage plot')
    plt.show()

def adversarial_num_plot(criterion):
    plt.plot(neuron_selection_area_list,adversarial_example_nums_model_dict[('resnet20',criterion)],label = 'resnet20',linestyle = ':',color = 'blue')
    plt.legend()
    plt.xlabel('choose region')
    plt.ylabel(criterion)
    plt.title(criterion+' '+'adversarial nums plot')
    plt.show()

if __name__ == '__main__':
    for model_name  in model_name_list:
        print('开始计算'+model_name)
        for criterion_idx in range(len(criterion_list)):
            if criterion_idx == 0:
                continue
            if criterion_idx == 1:
                continue
            if criterion_idx == 2:
                continue
            if criterion_idx == 3:
                continue
            get_data(model_name,criterion_list[criterion_idx],criterion_para_list[criterion_idx],model_name+'_'+criterion_list[criterion_idx],5,10,seed_selection_strategy[1])
            coverage_plot(criterion_list[criterion_idx])
            adversarial_num_plot(criterion_list[criterion_idx])