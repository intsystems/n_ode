from copy import deepcopy
from collections import defaultdict

from pipe import where, select

import torch

from torchdiffeq import odeint_adjoint

import pyro
import pyro.distributions
import pyro.nn
import pyro.nn.module
import pyro.infer
import pyro.infer.autoguide


class BayesNormDiagVectorField(pyro.nn.PyroModule):
    """Samples vector field's params from learnable prior
        and returns vector field as torch.nn.Module with these parameters
    """
    def __init__(
            self,
            vf: torch.nn.Module     # must not be used outside
        ):
        """_summary_

        Args:
            vf (torch.nn.Module): insance of vf class
        """
        super().__init__()

        self._vf_param_names = []

        for name, param in vf.named_parameters():
            # save names of the vf params
            self._vf_param_names.append(name)

            # ordinary parameters
            if name.find("batch_norm") != -1:
                setattr(
                    self,
                    name.replace(".", "_"),
                    pyro.nn.PyroParam(param)
                )
                continue

            # prior params
            setattr(
                self,
                name.replace(".", "_") + "_loc",
                pyro.nn.PyroParam(param)
            )
            sigma_init = 1
            setattr(
                self,
                name.replace(".", "_") + "_sigma",
                pyro.nn.PyroParam(
                    sigma_init * torch.ones_like(param),
                    constraint=pyro.distributions.constraints.positive
                )
            )

        # do not register vf as submodule
        self._vf = (vf, )

    def forward(self) -> torch.nn.Module:
        vf = self._vf[0]

        sampled_vf_state_dict = vf.state_dict()
        for name in self._vf_param_names:
            if name.find("batch_norm") != -1:
                sampled_vf_state_dict[name] = self.get_parameter(name.replace(".", "_"))
                continue

            loc = getattr(self, name.replace(".", "_") + "_loc")
            sigma = getattr(self, name.replace(".", "_") + "_sigma")
            sampled_vf_state_dict[name] = pyro.sample(name, pyro.distributions.Normal(loc, sigma).to_event(len(loc.shape)))
        vf.load_state_dict(sampled_vf_state_dict)

        return vf


class BayessianNODE(pyro.nn.PyroModule):
    def __init__(
            self,
            vf_sampler: pyro.nn.PyroModule,
            trajectory_len: int,
            odeint_kwargs: dict = {}
    ):
        """_summary_

        Args:
            vf_sampler (pyro.nn.PyroModule): bayessian vector field sampler
        """
        super().__init__()

        self.vf_sampler = vf_sampler
        self.traj_len = trajectory_len
        self.odeint_kwargs = odeint_kwargs
        # trajectory-uniform noise level
        self.sigma = pyro.nn.PyroParam(torch.tensor(1.0), constraint=pyro.distributions.constraints.positive)

    def forward(
            self,
            dataset_size: int,          # needed for correct elbo computation
            duration: torch.Tensor,
            z_0: torch.Tensor,
            traj: torch.Tensor = None   # <=> obs
    ) -> torch.Tensor:
        device = duration.device
        batch_size = duration.shape[0]

        # sample vf for batch
        vf = self.vf_sampler()

        with pyro.plate("traj_dataset", dataset_size, batch_size, dim=-3, device=device):
            pred = odeint_adjoint(
                vf,
                z_0,
                torch.arange(self.traj_len).to(z_0),
                **self.odeint_kwargs
            )
            # make dims = (batch, duration, state_dim)
            pred = torch.swapdims(pred, 0, 1)
            # build obs mask with durations
            if traj is not None:
                obs_mask = torch.full_like(traj, False).to(device)
                for i in range(batch_size):
                    obs_mask[i, :duration[i]] = True
            else:
                obs_mask = None

            # TODO: если obs_mask не работает, то сэмплить каждую траекторию своей длины
            pred_with_noise = pyro.sample(
                    "traj",
                    pyro.distributions.Normal(pred, self.sigma * torch.ones_like(pred).to(device)),
                    obs=traj,
                    obs_mask=obs_mask
                )

        return pred_with_noise