from MNIST.gen_diff import *
import matplotlib.pyplot as plt

'''这个种子选择策略后面没用了 只用在了DLFuzz方法上证明了种子选择策略的有效性 '''
model_name_list = ['lenet1','lenet4','lenet5']
criterion_list = ['NC','KMNC','NBC','SNAC','TKNC']
criterion_para_list = [0.5,1000,0,0,3]
neuron_selection_area_list = ['[1]','[2]','[3]','[4]','[5]']
seed_selection_strategy = [0,1]
run_times = 3
seed_selection_coverage_dict = {}
seed_selection_adversarial_nums_dict = {}


def get_data(model_name,criterion,criterion_para,neuron_selection_area,subdir,iteration_times,neuron_to_cover_num,seed_selection_strategy):
    neuron_coverage_sum = 0
    adversarial_example_nums_sum = 0
    for i in range(run_times):
        [neuron_coverage,adversarial_example_nums] = gen_diff_region_effect(model_name,criterion,criterion_para,neuron_selection_area,subdir,iteration_times,neuron_to_cover_num,seed_selection_strategy).result
        neuron_coverage_sum += neuron_coverage
        adversarial_example_nums_sum +=  adversarial_example_nums
    neuron_coverage_average = neuron_coverage_sum/run_times
    adversarial_example_nums_average = adversarial_example_nums_sum/run_times
    return neuron_coverage_average,adversarial_example_nums_average


if __name__ == '__main__':
    for model_name_idx in range(len(model_name_list)):
        for critrion_idx in range(len(criterion_list)):
            for seed_selection_idx in range(len(seed_selection_strategy)):
                neuron_coverage_average,adversarial_example_nums_average = get_data(model_name_list[model_name_idx],criterion_list[critrion_idx],criterion_para_list[critrion_idx],neuron_selection_area_list[0],'seed',5,5,seed_selection_strategy[seed_selection_idx])
                #adversarial_example_nums_average = get_data(model_name_list[model_name_idx],criterion_list[critrion_idx],criterion_para_list[critrion_idx],neuron_selection_area_list[0],'seed',5,5,seed_selection_strategy[seed_selection_idx])[1]
                seed_selection_coverage_dict[(model_name_list[model_name_idx],criterion_list[critrion_idx],seed_selection_strategy[seed_selection_idx])] = neuron_coverage_average
                seed_selection_adversarial_nums_dict[(model_name_list[model_name_idx],criterion_list[critrion_idx],seed_selection_strategy[seed_selection_idx])] =adversarial_example_nums_average
        print(model_name_list[model_name_idx])
        print(seed_selection_coverage_dict)
        print(seed_selection_adversarial_nums_dict)



