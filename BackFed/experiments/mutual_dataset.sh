############## Multishot CIFAR10 ############## 
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg \
    atk_config=cifar10_multishot \
    atk_config.secret_dataset=True \
    atk_config.model_poison_method=base,neurotoxin,chameleon \
    atk_config.data_poison_method=pattern,distributed,edge_case,a3fl,iba \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=cifar10_secret_dataset_multishot \
    cuda_visible_devices=\"0,1,2,3,4\"


############## Singleshot CIFAR10 ############## 
python main.py -m -cn cifar10 \
    aggregator=unweighted_fedavg \
    atk_config=cifar10_singleshot \
    atk_config.secret_dataset=True \
    atk_config.model_poison_method=base,neurotoxin,chameleon \
    atk_config.data_poison_method=pattern,distributed,edge_case,a3fl,iba \
    checkpoint=2000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=cifar10_secret_dataset_singleshot \
    cuda_visible_devices=\"0,1,2,3,4\"


############## Multishot EMNIST ############## 
python main.py -m -cn emnist \
    aggregator=unweighted_fedavg \
    atk_config=emnist_multishot \
    atk_config.secret_dataset=True \
    atk_config.model_poison_method=base,neurotoxin,chameleon \
    atk_config.data_poison_method=pattern,distributed,edge_case,a3fl,iba \
    checkpoint=1000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=emnist_secret_dataset_multishot \
    cuda_visible_devices=\"0,1,2,3,4\"


############## Singleshot EMNIST ############## 
python main.py -m -cn emnist \
    aggregator=unweighted_fedavg \
    atk_config=emnist_singleshot \
    atk_config.secret_dataset=True \
    atk_config.model_poison_method=base,neurotoxin,chameleon \
    atk_config.data_poison_method=pattern,distributed,edge_case,a3fl,iba \
    checkpoint=1000 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=0.5 \
    num_cpus=1 \
    dir_tag=enist_secret_dataset_singleshot \
    cuda_visible_devices=\"0,1,2,3,4\"


