import os.path

import pandas as pd
import yaml
import logging
import random
import numpy as np
import torch.nn
from torch.utils.data import random_split, DataLoader
from phd_experiments.hybrid_node_tt.models import LearnableOde
from phd_experiments.hybrid_node_tt.utils import get_dataset, get_solver, get_ode_func, get_tensor_dtype, \
    get_activation, get_loss_function, get_logger
from datetime import datetime
from tqdm import tqdm

"""
some material

stability of linear ode
https://physiology.med.cornell.edu/people/banfelder/qbio/resources_2010/2010_4.2%20Stability%20and%20Linearization%20of%20ODEs.pdf
https://www.cs.princeton.edu/courses/archive/fall11/cos323/notes/cos323_f11_lecture19_pde2.pdf
https://faculty.ksu.edu.sa/sites/default/files/stability.pdf
"""

"""
Objective of this script is to verify on research hypothesis ( Tensor-Neural ODE expressive power , no focus on
memory or speed 
dzdt = A.phi([z,t]) that works with complex problems
"""

EXPERIMENTS_LOG_DIR = "./experiments_log"
EXPERIMENTS_COUNTER_FILE = "./experiment_counter.txt"
PANDAS_MAX_DISPLAY_ROW = 1000
LOG_FORMAT = "[%(filename)s:%(lineno)s - %(funcName)10s()] %(asctime)s %(levelname)s %(message)s"
DATE_TIME_FORMAT = "%Y-%m-%d:%H:%M:%S"
if __name__ == '__main__':
    # set pandas config
    pd.set_option('display.max_rows', PANDAS_MAX_DISPLAY_ROW)
    # load configs
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        # get logger
    logger = get_logger(level=config['train']['log-level'], date_time_format=DATE_TIME_FORMAT, log_format=LOG_FORMAT,
                        experiments_counter_file_path=EXPERIMENTS_COUNTER_FILE, experiments_log_dir=EXPERIMENTS_LOG_DIR)
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
    test_loader = DataLoader(dataset=test_dataset, batch_size=config["train"]["batch_size"],
                             shuffle=config["train"]["shuffle"])

    # create model
    # get ode-func

    latent_dim = config["model"]["latent-dim"]
    assert latent_dim == input_dim, "latent-dim must == input-dim, for now"

    solver = get_solver(config=config)
    # get ode-func
    ode_func = get_ode_func(config=config)
    # get-model
    tensor_dtype = get_tensor_dtype(config=config)
    output_activation = get_activation(activation_name=config['model']['output-layer']['activation'])
    model = LearnableOde(Dx=input_dim, Dz=latent_dim, Dy=output_dim, solver=solver, t_span=config['solver']['t-span'],
                         tensor_dtype=tensor_dtype,
                         unif_low=config['model']['init']['low'], unif_high=config['model']['init']['high'],
                         ode_func=ode_func,
                         output_activation=output_activation,
                         output_linear_learnable=config['model']['output-layer']['learnable'],
                         projection_learnable=config['model']['projection']['learnable'])
    loss_fn = get_loss_function(loss_name=config['train']['loss'])
    optimizer = torch.optim.SGD(params=model.parameters(), lr=config['train']['lr'])
    logger.info(f"Running with config : \n "
                f"{config}"
                f"\n"
                f"============="
                f"\n")
    # TODO
    start_time = datetime.now()
    epoch_no_list = []
    epoch_avg_loss = []
    for epoch in tqdm(range(config['train']['epochs']), desc="epochs"):
        batches_losses = []
        for i, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_hat = model(X)
            residual = loss_fn(y_hat, y)
            loss = residual
            batches_losses.append(residual.item())
            loss.backward()
            optimizer.step()
        if epoch % config['train']['epochs_block'] == 0:
            epoch_no_list.append(epoch)
            epoch_avg_loss.append(np.nanmean(batches_losses))
            logger.debug(f"\t epoch # {epoch} : loss = {np.nanmean(batches_losses)}")
    end_time = datetime.now()
    epochs_losses_df = pd.DataFrame({'epoch': epoch_no_list, 'loss': epoch_avg_loss})
    logger.info(f'\n{epochs_losses_df}\n')
    logger.info(f'Training-Time = {(end_time - start_time).seconds} seconds')
