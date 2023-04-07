import yaml

import logging
import random
import numpy as np
import torch.nn
from torch.nn import MSELoss
from torch.utils.data import random_split, DataLoader

from phd_experiments.hybrid_node_tt.models import LearnableOde
from phd_experiments.hybrid_node_tt.utils import get_dataset, get_solver, get_ode_func, get_tensor_dtype, get_activation
from datetime import datetime

# TODO
#   * Document 2x3 experiments (tt/nn-odefunc) x ( toy_ode,toy_relu,boston datasets)
#       - save each experiment dump in a test file under experiments log
#       - document details in the gdoc
#   * experiment different weight init. schemes and document them
#       Steps
#       - https://github.com/rtqichen/torchdiffeq/blob/master/examples/ode_demo.py#L122
#       - https://pytorch.org/docs/stable/nn.init.html
#       - add mechanism to experiment different initializers
#       - document results in the gdoc
#   gdoc for experiments
#   https://docs.google.com/document/d/11-13S54BK4fdqMls0yja26wtveMBZQ3G0g9krLXeyG8/edit?usp=share_link

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

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    # load configs
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
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
    loss_fn = MSELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=config['train']['lr'])
    logger.info(f"Running with config : \n "
                f"{config}"
                f"\n"
                f"============="
                f"\n")
    # TODO
    start_time = datetime.now()
    for epoch in range(1, config['train']['epochs'] + 1):
        batches_loss = []
        for i, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_hat = model(X)
            residual = loss_fn(y_hat, y)
            loss = residual
            batches_loss.append(residual.item())
            loss.backward()
            optimizer.step()
        if epoch % 1 == 0:
            logger.info(f'epoch = {epoch} -> avg-batches-loss = {np.nanmean(batches_loss)}')
    end_time = datetime.now()
    logger.info(f'Training-Time = {(end_time - start_time).seconds} seconds')
