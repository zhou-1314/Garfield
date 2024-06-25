import inspect
import os
import dill
# import pickle
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import torch
import scanpy as sc
from anndata import AnnData, concat
from scipy.sparse import issparse

from ._utils import UnpicklerCpu, _validate_var_names


class BaseMixin:
    """ Adapted from
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
            self.merged_adata.write(
                os.path.join(dir_path, "adata_ref.h5ad"), **anndata_write_kwargs
            )

        model_save_path = os.path.join(dir_path, "model_params.pt")
        attr_save_path = os.path.join(dir_path, "attr.pkl")
        varnames_save_path = os.path.join(dir_path, "var_names.csv")

        var_names = self.merged_adata.var_names.astype(str)
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
        load_state_dict = {k: v for k, v in load_state_dict.items() if k in new_state_dict}

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


class SurgeryMixin:
    @classmethod
    def load_query_data(
        cls,
        adata: AnnData,
        reference_model: Union[str, 'Model'],
        freeze: bool = True,
        remove_dropout: bool = True,
        map_location = None,
        **kwargs
    ):
        """Transfer Learning function for new data. Uses old trained model and expands it for new conditions.

           Parameters
           ----------
           adata
                Query anndata object.
           reference_model
                A model to expand or a path to a model folder.
           freeze: Boolean
                If 'True' freezes every part of the network except the first layers of encoder/decoder.
           remove_dropout: Boolean
                If 'True' remove Dropout for Transfer Learning.
           map_location
                map_location to remap storage locations (as in '.load') of 'reference_model'.
                Only taken into account if 'reference_model' is a path to a model on disk.
           kwargs
                kwargs for the initialization of the query model.

           Returns
           -------
           new_model
                New model to train on query data.
        """
        if isinstance(reference_model, str):
            attr_dict, model_state_dict, var_names = cls._load_params(reference_model, map_location)
            adata_ref = sc.read_h5ad(os.path.join(reference_model, "adata_ref.h5ad"))
            adata_ref.X = adata_ref.layers["counts"] if "counts" in adata_ref.layers.keys() else adata_ref.X
            # adata = _validate_var_names(adata, var_names)
        else:
            attr_dict = reference_model._get_public_attributes()
            model_state_dict = reference_model.model.state_dict()
            var_names = reference_model.merged_adata.var_names
            adata_ref = reference_model.merged_adata
            adata_ref.X = adata_ref.layers["counts"] if "counts" in adata_ref.layers.keys() else adata_ref.X
            # adata = _validate_var_names(adata, reference_model.merged_adata.var_names)

        init_params = deepcopy(cls._get_init_params_from_dict(attr_dict))

        if remove_dropout:
            init_params['dropout'] = 0.0

        # don't preprocess the data
        init_params['min_features'] = 0
        init_params['min_cells'] = 0

        # load the data
        adata_concat = concat(
            [adata_ref, adata],
            label='projection',
            keys=['reference', 'query'],
            index_unique=None,
        )
        init_params['adata_list'] = adata_concat
        init_params['rna_n_top_features'] = var_names

        init_params.update(kwargs)

        new_model = cls(init_params)
        new_model.model.to(next(iter(model_state_dict.values())).device)
        new_model._load_expand_params_from_dict(model_state_dict)

        if freeze:
            new_model.model.freeze = True
            for name, p in new_model.model.named_parameters():
                p.requires_grad = False
                if 'eps_weight' in name or 'eps_bias' in name:
                    p.requires_grad = True
                if "layers.0" in name or "norm.0" in name:
                    p.requires_grad = True

        return new_model