"""
This module contains generic trainer functionalities, added as a Mixin to
the Trainer module.
"""
import inspect
import os
import dill
import pickle
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import torch
import scanpy as sc
from anndata import AnnData, concat


class BaseMixin:
    """Adapted from
    Title: scvi-tools
    Authors: Romain Lopez <romain_lopez@gmail.com>,
             Adam Gayoso <adamgayoso@berkeley.edu>,
             Galen Xing <gx2113@columbia.edu>
    Date: 14.12.2020
    Code version: 0.8.0-beta.0
    Availability: https://github.com/YosefLab/scvi-tools
    Link to the used code:
    https://github.com/YosefLab/scvi-tools/blob/0.8.0-beta.0/scvi/core/models/base.py
    """

    def _get_user_attributes(self):
        # returns all the self attributes defined in a model class, eg, self.is_trained_
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        attributes = [
            a for a in attributes if not (a[0].startswith("__") and a[0].endswith("__"))
        ]
        attributes = [a for a in attributes if not a[0].startswith("_abc_")]
        return attributes

    def _get_public_attributes(self):
        public_attributes = self._get_user_attributes()
        public_attributes = {a[0]: a[1] for a in public_attributes if a[0][-1] == "_"}
        return public_attributes

    def save(
        self,
        dir_path: str,
        overwrite: bool = False,
        save_anndata: bool = True,
        **anndata_write_kwargs,
    ):
        """Save the state of the model.
        Neither the trainer optimizer state nor the trainer history are saved.
        Parameters
        ----------
        dir_path
             Path to a directory.
        overwrite
             Overwrite existing data or not. If `False` and directory
             already exists at `dir_path`, error will be raised.
        save_anndata
             If True, also saves the anndata
        anndata_write_kwargs
             Kwargs for anndata write function
        """
        # get all the public attributes
        public_attributes = self._get_public_attributes()
        # save the model state dict and the trainer state dict only
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                "{} already exists. Please provide an unexisting directory for saving.".format(
                    dir_path
                )
            )

        if save_anndata:
            self.adata.write(
                os.path.join(dir_path, "adata_ref.h5ad"), **anndata_write_kwargs
            )

        model_save_path = os.path.join(dir_path, "model_params.pt")
        attr_save_path = os.path.join(dir_path, "attr.pkl")
        varnames_save_path = os.path.join(dir_path, "var_names.csv")

        var_names = self.adata.var_names.astype(str)
        var_names = var_names.to_numpy()
        np.savetxt(varnames_save_path, var_names, fmt="%s")

        torch.save(self.model.state_dict(), model_save_path)

        # 现在使用 pickle.dump 来序列化，pickle 库已被替换为 dill
        with open(attr_save_path, "wb") as f:
            dill.dump(public_attributes, f)

    def _load_expand_params_from_dict(self, state_dict):
        load_state_dict = state_dict.copy()

        new_state_dict = self.model.state_dict()

        # 创建一个新的状态字典，仅包括在 new_state_dict 中也存在的键
        load_state_dict = {
            k: v for k, v in load_state_dict.items() if k in new_state_dict
        }

        for key, ten in new_state_dict.items():
            if key not in load_state_dict:
                load_state_dict[key] = ten

        self.model.load_state_dict(load_state_dict)

    @classmethod
    def _load_params(cls, dir_path: str, map_location: Optional[str] = None):
        setup_dict_path = os.path.join(dir_path, "attr.pkl")
        model_path = os.path.join(dir_path, "model_params.pt")
        varnames_path = os.path.join(dir_path, "var_names.csv")

        try:
            with open(setup_dict_path, "rb") as handle:
                attr_dict = dill.load(handle)
        # This catches the following error:
        # RuntimeError: Attempting to deserialize object on a CUDA device
        # but torch.cuda.is_available() is False.
        # If you are running on a CPU-only machine, please use torch.load with
        # map_location=torch.device('cpu') to map your storages to the CPU.
        except RuntimeError:
            with open(setup_dict_path, "rb") as handle:
                attr_dict = UnpicklerCpu(handle).load()

        model_state_dict = torch.load(model_path, map_location=map_location)

        var_names = np.genfromtxt(varnames_path, delimiter=",", dtype=str)

        return attr_dict, model_state_dict, var_names


class UnpicklerCpu(pickle.Unpickler):
    """Helps to pickle.load a model trained on GPU to CPU.

    See also https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219.
    """

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)
