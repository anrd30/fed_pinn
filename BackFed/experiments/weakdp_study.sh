python main.py -m \
    aggregator=weakdp \
    aggregator_config.weakdp.std_dev=0.0002,0.001 \
    aggregator_config.weakdp.clipping_norm=7.0,5.0,3.0,1.0 \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2600 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=1 \
    num_cpus=1 \
    cuda_visible_devices=\"5,4,3,0\" \
    dir_tag=weakdp_study_weak

python main.py -m \
    aggregator=weakdp \
    aggregator_config.weakdp.std_dev=0.005,0.025 \
    aggregator_config.weakdp.clipping_norm=7.0,5.0,3.0,1.0 \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=a3fl \
    atk_config.poison_start_round=2301 \
    atk_config.poison_end_round=2600 \
    save_logging=csv \
    num_rounds=600 \
    num_gpus=1 \
    num_cpus=1 \
    cuda_visible_devices=\"5,4,3,0\" \
    dir_tag=weakdp_study_strong
