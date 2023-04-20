import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import random_split, DataLoader
from torchviz import make_dot
from tqdm import tqdm
from phd_experiments.hybrid_node_tt.models import ProjectionModel, OdeSolverModel, OutputModel, LearnableOde, NNodeFunc, \
    TensorTrainOdeFunc
from phd_experiments.hybrid_node_tt.utils import get_activation, get_logger, get_dataset, get_solver, get_ode_func, \
    assert_models_learnability, get_tensor_dtype, get_loss_function

EXPERIMENTS_LOG_DIR = "./experiments_log"
EXPERIMENTS_COUNTER_FILE = "./experiment_counter.txt"
LOG_FORMAT = "[%(filename)s:%(lineno)s - %(funcName)10s()] %(asctime)s %(levelname)s %(message)s"
DATE_TIME_FORMAT = "%Y-%m-%d:%H:%M:%S"
PANDAS_MAX_DISPLAY_ROW = 1000


class TrajectorySynthDataSet(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.size()[0]


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
    ode_func = torch.load("ode_func_models/ode_func_nn_experiment_no_297_2023-04-13-22:23:33.model")
    assert isinstance(ode_func, NNodeFunc)
    ode_func.emulation = True
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
    logger.info("Generating trajectory dzdt=NN([z,t]) data with trained nn-ode-func")
    for epoch in tqdm(range(config['train']['epochs']), desc="epochs"):
        batches_losses = []
        for i, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_hat = model(X)
            loss_val = loss_fn(y_hat, y)
            logger.info(f'Epoch # {epoch} - Batch # {i} = {loss_val}')

    logger.info("============================================")
    logger.info("Emulation mode : training tt-ode-func over generated trajectory")
    epochs_emu = 500
    N = len(ode_func.dzdt_emu)
    Dz_aug = 14
    ode_model_emu = TensorTrainOdeFunc(Dz=13, basis_model="poly", unif_low=-0.1, unif_high=0.1, tt_rank=4,
                                       poly_deg=5)
    # ode_model_emu = NNodeFunc(latent_dim=13,nn_hidden_dim=100,emulation=False)
    emu_optimizer = torch.optim.SGD(params=ode_model_emu.parameters(), lr=0.1)
    for epoch_idx in tqdm(range(epochs_emu), desc="epochs"):
        batches_losses = []
        for i in range(N):  # len batches
            emu_optimizer.zero_grad()
            z_aug = ode_func.z_aug_emu[i].detach()
            dzdt_true = ode_func.dzdt_emu[i].detach()[:, 0]
            z_batch = z_aug[:, :(Dz_aug - 1)]
            t = z_aug[:, -1].detach().numpy()[0]
            dzdt_hat = None
            for core_idx in range(ode_model_emu.A_TT.order):
                # core_idx = 0
                ode_model_emu.set_learnable_core(core_idx, True)
                dzdt_hat = ode_model_emu.forward2(t, z_batch)

                # FIXME for debugging
                make_dot(dzdt_hat, params=dict(ode_model_emu.A_TT.named_parameters())).render(f"ttxr_traj_{i}",
                                                                                              format="png")
                loss = loss_fn(dzdt_hat, dzdt_true)
                batches_losses.append(loss.item()) # fixme this is per core_dim, batch and epoch !!!
                loss.backward()
                # fixme for debugging
                core_grad = ode_model_emu.A_TT.core_tensors[f"G{core_idx}"].grad
                emu_optimizer.step()
                logger.info(f'Emulation : epoch # {epoch_idx} - batch # {i} - core # {core_idx}=> loss = {loss}')
                ode_model_emu.set_learnable_core(core_idx, False)
        # if epoch_idx % 1 == 0:
        #     logger.info(f'Emulation - Epoch # {epoch_idx} - loss => {np.nanmean(batches_losses)}')

    # TODO notes, review
    # 1. tt-ode-func for this training also has the problem of very small gradients
    # TODO
    # 1. experiment different basis functions (Non-linear regression)
    # http://www.cs.toronto.edu/~mbrubake/teaching/C11/Handouts/NonlinearRegression.pdf
    # https://cedar.buffalo.edu/~srihari/CSE574/Chap6/Chap6.2-RadialBasisFunctions.pdf
    # http://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/
    # https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer
    # https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer/blob/master/Torch%20RBF/torch_rbf.py
