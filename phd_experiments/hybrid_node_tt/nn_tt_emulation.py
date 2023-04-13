import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import random_split, DataLoader

from phd_experiments.hybrid_node_tt.models import ProjectionModel, OdeSolverModel, OutputModel, LearnableOde
from phd_experiments.hybrid_node_tt.ttnode_driver import PANDAS_MAX_DISPLAY_ROW
from phd_experiments.hybrid_node_tt.utils import get_activation, get_logger, get_dataset, get_solver, get_ode_func, \
    assert_models_learnability, get_tensor_dtype, get_loss_function

EXPERIMENTS_LOG_DIR = "./experiments_log"
EXPERIMENTS_COUNTER_FILE = "./experiment_counter.txt"
PANDAS_MAX_DISPLAY_ROW = 1000
LOG_FORMAT = "[%(filename)s:%(lineno)s - %(funcName)10s()] %(asctime)s %(levelname)s %(message)s"
DATE_TIME_FORMAT = "%Y-%m-%d:%H:%M:%S"

if __name__ == '__main__':
    tstamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    # set pandas config
    pd.set_option('display.max_rows', PANDAS_MAX_DISPLAY_ROW)
    # load configs
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # get logger
    logger, experiment_number = get_logger(level=config['train']['log-level'],
                                           date_time_format=DATE_TIME_FORMAT, log_format=LOG_FORMAT,
                                           experiments_counter_file_path=EXPERIMENTS_COUNTER_FILE,
                                           experiments_log_dir=EXPERIMENTS_LOG_DIR)
    logger.info(f"This is NN-TT Emulation training script")
    # set seed
    seed = config['train']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # get dataset and loader
    overall_dataset = get_dataset(config)
    input_dim = overall_dataset.get_input_dim()
    output_dim = overall_dataset.get_output_dim()
    splits = random_split(dataset=overall_dataset,
                          lengths=[config["train"]["ratio"], 1 - config["train"]["ratio"]])
    train_dataset = splits[0]
    test_dataset = splits[1]
    train_loader = DataLoader(dataset=train_dataset, batch_size=config["train"]["batch_size"],
                              shuffle=config["train"]["shuffle"])
    # test_loader = DataLoader(dataset=test_dataset, batch_size=config["train"]["batch_size"],
    #                          shuffle=config["train"]["shuffle"])

    # get ode-solver-model
    solver = get_solver(config=config)
    latent_dim = config["container"]["latent-dim"]
    # FIXME remove and test Dz>Dx
    assert latent_dim == input_dim, "latent-dim must == input-dim, for now"
    # ode_func = get_ode_func(config=config)
    ode_func = torch.load("ode_func_models/ode_func_nn_experiment_no_271_2023-04-13-20:38:00.model")
    ode_solver_model = OdeSolverModel(solver=solver, ode_func=ode_func, t_span=config['ode']['solver']['t-span'])
    # get projection-model
    projection_model_activation = get_activation(activation_name=config['projection']['activation'])
    projection_model = ProjectionModel(Dx=input_dim, Dz=latent_dim,
                                       activation_module=projection_model_activation,
                                       unif_low=config['init']['uniform']['low'],
                                       unif_high=config['init']['uniform']['high'],
                                       learnable=config['projection']['learnable'])
    # get output-model
    output_activation_model = get_activation(activation_name=config['output']['activation'])
    output_model = OutputModel(Dz=latent_dim, Dy=output_dim, activation_module=output_activation_model,
                               learnable=config['output']['learnable'],
                               linear_weight_full_value=config['output']['full'])
    # assert learnability to be as configured
    assert_models_learnability(config=config, projection_model=projection_model, output_model=output_model)
    # get-model
    tensor_dtype = get_tensor_dtype(config=config)
    model = LearnableOde(projection_model=projection_model, ode_solver_model=ode_solver_model,
                         output_model=output_model)
    loss_fn = get_loss_function(loss_name=config['train']['loss'])
    optimizer = torch.optim.SGD(params=model.parameters(), lr=config['train']['lr'])
    logger.info(f"ODE-FUNC Model = {type(ode_func)} , learnable-numel = {ode_func.num_learnable_scalars()}")
    logger.info(f"Running with config : \n "
                f"{config}"
                f"\n"
                f"============="
                f"\n")
