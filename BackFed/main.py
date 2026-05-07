"""
Main entry point.
"""
import hydra
import omegaconf
import torch
import os
import ray
import traceback

from hydra.core.hydra_config import HydraConfig
from rich.traceback import install
from backfed.servers.base_server import BaseServer
from backfed.utils import system_startup, log
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import instantiate
from logging import ERROR

# Use a decorator that allows specifying the config file via command line
# To change main config file: 
#   python main.py --config-name (or -cn) sentiment140 (corresponds to config/sentiment140.yaml) 
@hydra.main(config_path="config", config_name="cifar10", version_base=None)
def main(config: DictConfig):
    # Add output_dir to config
    hydra_cfg = HydraConfig.get()
    with open_dict(config):
        config.output_dir = hydra_cfg.runtime.output_dir
    # Set seeds and do some setups
    system_startup(config)
    aggregator = config["aggregator"]
    try:
        server : BaseServer = instantiate(config.aggregator_config[aggregator], server_config=config, _recursive_=False)
        server.run_experiment()
    except Exception as e:
        error_traceback = traceback.format_exc()
        log(ERROR, f"Error: {e}\n{error_traceback}") # Log traceback
        exit(1)

if __name__ == "__main__":
    # Rich traceback and suppress traceback from hydra, omegaconf, and torch
    OmegaConf.register_new_resolver("eval", eval) # For conditional config on dir_tag
    install(show_locals=False, suppress=[hydra, omegaconf, torch, ray])
    os.environ["HYDRA_FULL_ERROR"] = "1" # For detailed error messages
    os.environ["RAY_memory_monitor_refresh_ms"] = '0' # Disable worker killing
    main()
