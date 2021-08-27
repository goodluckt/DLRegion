from MNIST.gen_diff import *
import matplotlib.pyplot as plt


model_name_list = ['lenet1','lenet4','lenet5']
criterion_list = ['NC','KMNC','NBC','SNAC','TKNC']
criterion_para_list = [0.5,1000,0,0,3]
neuron_selection_area_list = ['[1]','[2]','[3]','[4]','[5]']
seed_selection_strategy = [0,1]
run_times = 10
neuron_coverage_model_dict = {}
adversarial_example_nums_model_dict = {}

def get_data(model_name,criterion,criterion_para,subdir,iteration_times,neuron_to_cover_num,seed_selection_strategy):
    adversarial_example_nums_list = []
    neuron_coverage_list = []
    for area in neuron_selection_area_list:
        print('choose area:'+' '+area)
        neuron_coverage_all = 0
        adversarial_example_nums_all = 0
        for i in range(run_times):
            [neuron_coverage, adversarial_example_nums] = gen_diff_region_effect(model_name,criterion,criterion_para,area,subdir+'_'+area+'_'+str(i),iteration_times,neuron_to_cover_num,seed_selection_strategy).result
            neuron_coverage_all += neuron_coverage
            adversarial_example_nums_all += adversarial_example_nums
        neuron_coverage_average = neuron_coverage_all/run_times
        adversarial_example_nums_average = adversarial_example_nums_all/run_times
        neuron_coverage_list.append(neuron_coverage_average)
        adversarial_example_nums_list.append(adversarial_example_nums_average)
    neuron_coverage_model_dict[(model_name,criterion)] = neuron_coverage_list
    adversarial_example_nums_model_dict[(model_name,criterion)] =  adversarial_example_nums_list

def coverage_plot(criterion):
    plt.plot(neuron_selection_area_list,neuron_coverage_model_dict[('lenet1',criterion)],label='lenet1',linestyle = '--',color = 'green')
    plt.plot(neuron_selection_area_list, neuron_coverage_model_dict[('lenet4', criterion)], label='lenet4',linestyle=':', color='red')
    plt.plot(neuron_selection_area_list, neuron_coverage_model_dict[('lenet5', criterion)], label='lenet5',linestyle='-', color='blue')
    plt.legend()
    plt.xlabel('choose region')
    plt.ylabel(criterion)
    plt.title(criterion+' '+'coverage plot')
    plt.show()

def adversarial_num_plot(criterion):
    plt.plot(neuron_selection_area_list,adversarial_example_nums_model_dict[('lenet1',criterion)],label = 'lenet1',linestyle = '--',color = 'green')
    plt.plot(neuron_selection_area_list,adversarial_example_nums_model_dict[('lenet4',criterion)],label = 'lenet4',linestyle = ':',color = 'red')
    plt.plot(neuron_selection_area_list,adversarial_example_nums_model_dict[('lenet5',criterion)],label = 'lenet5',linestyle = '-',color = 'blue')
    plt.legend()
    plt.xlabel('choose region')
    plt.ylabel(criterion)
    plt.title(criterion+' '+'adversarial nums plot')
    plt.show()

if __name__ == '__main__':
    for model_name  in model_name_list:
        print('开始计算'+model_name)
        get_data(model_name,criterion_list[4],criterion_para_list[4],model_name+'_'+criterion_list[4],5,5,seed_selection_strategy[1])
    coverage_plot(criterion_list[4])
    adversarial_num_plot(criterion_list[4])


