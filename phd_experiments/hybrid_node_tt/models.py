from typing import List, Tuple
import torch
from phd_experiments.hybrid_node_tt.basis import Basis
from phd_experiments.torch_ode_solvers.torch_ode_solver import TorchOdeSolver

ACTIVATIONS = (torch.nn.Identity, torch.nn.Sigmoid, torch.nn.Tanh, torch.nn.ReLU)


# hooks

def tensor_train_ode_func_hook(module, grad_input, grad_output):
    # https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_full_backward_hook.html#torch.nn.modules.module.register_module_full_backward_hook
    pass


def tensor_train_hook(module, grad_input, grad_output):
    pass


class OdeSolverModel(torch.nn.Module):
    def __init__(self, solver: TorchOdeSolver, ode_func: torch.nn.Module,
                 t_span: Tuple):
        super().__init__()
        self.t_span = t_span
        self.ode_func = ode_func
        self.solver = solver

    def forward(self, z0):
        soln = self.solver.solve_ivp(func=self.ode_func, t_span=self.t_span, z0=z0, args=None)
        zT = soln.z_trajectory[-1]
        zT.register_hook(OdeSolverModel.zT_hook)
        return zT

    @staticmethod
    def zT_hook(grad):
        pass


class TensorTrainFixedRank(torch.nn.Module):
    def __init__(self, dims: List[int], fixed_rank: int, requires_grad: bool, unif_low: float, unif_high: float,
                 poly_deg: int):
        """
        dims : dims[0]-> output dim
        dims[1:order] -> input dims
        """
        # register hooks

        super().__init__()
        self.poly_deg = poly_deg
        order = len(dims)
        tensor_uniform = torch.distributions.Uniform(low=unif_low, high=unif_high)
        # https://pytorch.org/docs/stable/generated/torch.nn.ParameterDict.html
        self.core_tensors = torch.nn.ParameterDict(
            {'G0': torch.nn.Parameter(tensor_uniform.sample(torch.Size([dims[1], fixed_rank])))})
        self.out_dim = dims[0]
        for i in range(2, order - 1):
            self.core_tensors[f"G{i - 1}"] = torch.nn.Parameter(
                tensor_uniform.sample(sample_shape=torch.Size([fixed_rank, dims[i], fixed_rank])))

        self.core_tensors[f"G{order - 2}"] = torch.nn.Parameter(
            tensor_uniform.sample(sample_shape=torch.Size([fixed_rank, dims[order - 1], self.out_dim])),
            requires_grad=requires_grad)
        assert len(self.core_tensors) == order - 1, \
            f"# core tensors should == order-1 : {len(self.core_tensors)} != {order - 1}"

        # FIXME register hooks
        for core in self.core_tensors.values():
            core.register_hook(TensorTrainFixedRank.core_hook)

        self.register_backward_hook(tensor_train_hook)

    def norm(self):
        tot_norm = 0

        for tensor in self.core_tensors.values():
            tot_norm += torch.norm(tensor)
        return tot_norm

    def display_sizes(self):
        sizes = []
        for tensor in self.core_tensors:
            sizes.append(tensor.size())
        self.logger.info(f'TensorTrain sizes = {sizes}')

    def is_learnable(self):
        res = all(list(map(lambda x: x.requires_grad, self.core_tensors)))
        return res

    def forward(self, t: float, z: torch.Tensor) -> torch.Tensor:
        core_activation = torch.nn.Tanh()
        # https://www.sciencedirect.com/science/article/abs/pii/S0893608021003415
        Phi = Basis.poly(z=z, t=t, poly_deg=self.poly_deg)
        assert len(self.core_tensors) == len(
            Phi), f"# of core-tensors must == number of basis tensors " \
                  f", {len(self.core_tensors)} != {len(Phi)}"
        n_cores = len(self.core_tensors)
        # first core
        core = self.core_tensors[f"G{0}"]
        res_tensor = torch.einsum("ij,bi->bj", core_activation(core), Phi[0])
        # middle cores
        for i in range(1, len(self.core_tensors) - 1):
            core = self.core_tensors[f"G{i}"]
            core_basis = torch.einsum("ijk,bj->bik", core_activation(core), Phi[i])
            res_tensor = torch.einsum("bi,bik->bk", res_tensor, core_basis)
        # last core
        core = self.core_tensors[f"G{n_cores - 1}"]
        core_basis = torch.einsum("ijl,bj->bil", core_activation(core), Phi[n_cores - 1])
        res_tensor = torch.einsum("bi,bil->bl", res_tensor, core_basis)
        assert res_tensor.size()[1] == self.out_dim, f"output tensor size must = " \
                                                     f"out_dim : {res_tensor.size()}!={self.out_dim}"
        return res_tensor

    @staticmethod
    def core_hook(grad):
        pass

    def gradients(self):
        gradient_list = list(map(lambda core: core.grad, self.core_tensors.values()))
        return gradient_list

    def flat_gradients(self):
        # TODO debug
        grad_vec = torch.cat(list(map(lambda core: torch.flatten(core.grad), self.core_tensors.values())))
        return grad_vec

    def gradients_sum_norm(self):
        gradient_list = self.gradients()
        norms_sum = sum(list(map(lambda g: torch.norm(g).item(), gradient_list)))
        return norms_sum

    def num_learnable_scalars(self):
        learnable_scalars_numel = 0
        for core in self.core_tensors.values():
            counter = core.numel() if core.requires_grad else 0
            learnable_scalars_numel += counter
        return learnable_scalars_numel

    def display_cores(self):
        self.logger.info(f'Cores : \n{self.core_tensors}\n')

    def flatten(self):
        params_vec = torch.cat(list(map(lambda core: torch.flatten(core), self.core_tensors.values())))
        return params_vec


class NNodeFunc(torch.nn.Module):
    # https://github.com/rtqichen/torchdiffeq/blob/master/examples/ode_demo.py#L111
    def __init__(self, latent_dim, nn_hidden_dim, emulation):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + 1, nn_hidden_dim),  # +1 for time
            torch.nn.Tanh(),
            torch.nn.Linear(nn_hidden_dim, latent_dim),
        )
        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.1)
                torch.nn.init.constant_(m.bias, val=0)
        self.emulation = emulation
        self.dzdt_emu = []
        self.z_aug_emu = []

    def forward(self, t, z, *args):
        b = z.size()[0]
        t_tensor = torch.tensor([t]).repeat([b, 1])
        z_aug = torch.cat([z, t_tensor], dim=1)
        dzdt = self.net(z_aug)
        if self.emulation:
            self.dzdt_emu.append(dzdt)
            self.z_aug_emu.append(z_aug)
        return dzdt

    def gradients(self):
        gradient_list = []
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                gradient_list.append(layer.weight.grad)
                gradient_list.append(layer.bias.grad)
        return gradient_list

    def flat_gradients(self):
        gradients = self.gradients()
        gradients_vec = torch.cat(list(map(lambda g: torch.flatten(g), gradients)))
        return gradients_vec

    def gradients_sum_norm(self):
        gradient_list = self.gradients()
        norms_sum = sum(list(map(lambda g: torch.norm(g).item(), gradient_list)))
        return norms_sum

    def num_learnable_scalars(self):
        learnable_numel = 0
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                numel_val = layer.weight.numel() if layer.weight.requires_grad else 0
                learnable_numel += numel_val
                numel_val = layer.bias.numel() if layer.bias.requires_grad else 0
                learnable_numel += numel_val
        return learnable_numel

    def flatten(self):
        params_ = []
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                if layer.weight.requires_grad:
                    params_.append(layer.weight)
                if layer.bias.requires_grad:
                    params_.append(layer.bias)
        vec = torch.cat(list(map(lambda param: torch.flatten(param), params_)))
        return vec


class TensorTrainOdeFunc(torch.nn.Module):

    def __init__(self, Dz: int, basis_model: str, unif_low: float, unif_high: float,
                 tt_rank: str | int | List[int], poly_deg: int):
        super().__init__()
        # register hooks for debugging
        self.register_full_backward_hook(tensor_train_ode_func_hook)
        # check params
        assert isinstance(poly_deg, int), f"deg must be int, got {type(self.deg)}"
        assert isinstance(tt_rank, int), f"Supporting fixed ranks only"
        assert basis_model == "poly", f"Supporting only poly basis"
        dims_A = [Dz] + [poly_deg + 1] * (Dz + 1)  # deg+1 as polynomial start from z^0 , z^1 , .. z^deg
        # Dz+1 to append time
        self.order = len(dims_A)
        self.A_TT = TensorTrainFixedRank(dims=dims_A, fixed_rank=tt_rank, unif_low=unif_low, unif_high=unif_high,
                                         requires_grad=True, poly_deg=poly_deg)

    def forward(self, t: float, z: torch.Tensor):
        """
        dzdt = A.Phi([z,t])
        """
        dzdt = self.A_TT(t, z)
        dzdt.register_hook(TensorTrainOdeFunc.dzdt_hook)
        return torch.nn.Tanh()(dzdt)

    @staticmethod
    def dzdt_hook(grad):
        pass

    def is_learnable(self):
        return self.A_TT.is_learnable()

    def gradients(self):
        return self.A_TT.gradients()

    def gradients_sum_norm(self):
        return self.A_TT.gradients_sum_norm()

    def num_learnable_scalars(self):
        return self.A_TT.num_learnable_scalars()

    def flatten(self):
        return self.A_TT.flatten()

    def flat_gradients(self):
        return self.A_TT.flat_gradients()


class ProjectionModel(torch.nn.Module):
    def __init__(self, Dx: int, Dz: int, activation_module: torch.nn.Module, unif_low: float, unif_high: float,
                 learnable: bool):
        super().__init__()
        self.activation_module = activation_module
        assert isinstance(self.activation_module,
                          ACTIVATIONS), f"activation module {self.activation_module} " \
                                        f"is not supported, must be one of {ACTIVATIONS}"
        assert Dz >= Dx, f"Dz must be >= Dx , got Dx={Dx} and Dz = {Dz}"
        if learnable:
            self.P = torch.nn.Parameter(torch.distributions.Uniform(unif_low, unif_high).sample(torch.Size([Dz, Dx])),
                                        requires_grad=True)
        else:
            assert Dx == Dz, f"If projection model is not learnable : Dx must ==Dz , got Dx = {Dx} and Dz = {Dz}"
            self.P = torch.eye(Dz)

    def forward(self, x):
        linear_out = torch.einsum(f"bi,ij->bj", x, self.P)
        activated_out = self.activation_module(linear_out)
        return activated_out

    def is_learnable(self):
        return self.P.requires_grad


class OutputModel(torch.nn.Module):
    def __init__(self, Dz: int, Dy: int, activation_module: torch.nn.Module, learnable: bool,
                 linear_weight_full_value: float):
        super().__init__()
        activation_module = activation_module
        assert isinstance(activation_module, ACTIVATIONS), f"activation-module must be one of {ACTIVATIONS}"
        assert Dz >= Dy, f"Dz must be > Dy; got Dz = {Dz} and Dy = {Dy}"
        self.linear_part = torch.nn.Linear(Dz, Dy)
        if not learnable:
            # set an arbitrary, fixed not-trainable weight
            weight_size = self.linear_part.weight.size()
            self.linear_part.weight = torch.nn.Parameter(torch.full(size=weight_size,
                                                                    fill_value=linear_weight_full_value))
            self.linear_part.weight.requires_grad = False
            # set zeros, not trainable bias
            bias_size = self.linear_part.bias.size()
            self.linear_part.bias = torch.nn.Parameter(torch.zeros(bias_size))
            self.linear_part.bias.requires_grad = False
        # activation module should be before the linear part (from experiment and neural-ode code
        # https://github.com/rtqichen/torchdiffeq/blob/master/examples/ode_demo.py#L116
        self.output_layer_model = torch.nn.Sequential(activation_module, self.linear_part)

    def forward(self, x):
        return self.output_layer_model(x)

    def is_learnable(self):
        return any([self.linear_part.weight.requires_grad, self.linear_part.weight.requires_grad])


class LearnableOde(torch.nn.Module):

    def __init__(self, projection_model: torch.nn.Module,
                 ode_solver_model: torch.nn.Module,
                 output_model: torch.nn.Module):
        super().__init__()
        self.complete_model = torch.nn.Sequential(projection_model, ode_solver_model, output_model)

    def forward(self, x: torch.Tensor):
        return self.complete_model(x)
