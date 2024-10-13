"""
This module contains generic base model functionalities, added as a Mixin to the
NicheCompass model.
"""

import inspect
import os
from copy import deepcopy
import warnings
from typing import Optional

import numpy as np

# import pickle
import scipy.sparse as sp
import torch
from anndata import AnnData

from .utils import load_saved_files, save_model_with_fallback


class BaseModelMixin:
    """
    Base model mix in class for universal model functionalities.

    Parts of the implementation are adapted from
    https://github.com/theislab/scarches/blob/master/scarches/models/base/_base.py#L15
    (01.10.2022) and
    https://github.com/scverse/scvi-tools/blob/master/scvi/model/base/_base_model.py#L63
    (01.10.2022).
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
        save_adata: bool = False,
        adata_file_name: str = "adata_ref.h5ad",
        **anndata_write_kwargs,
    ):
        """
        Save model to disk (the Trainer optimizer state is not saved).

        Parameters
        ----------
        dir_path:
            Path of the directory where the model will be saved.
        overwrite:
            If `True`, overwrite existing data. If `False` and directory
            already exists at `dir_path`, error will be raised.
        save_adata:
            If `True`, also saves the AnnData object.
        adata_file_name:
            File name under which the AnnData object will be saved.
        adata_write_kwargs:
            Kwargs for adata write function.
        """
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                f"Directory '{dir_path}' already exists."
                "Please provide another directory for saving."
            )

        model_save_path = os.path.join(dir_path, "model_params.pt")
        attr_save_path = os.path.join(dir_path, "attr.pkl")
        var_names_save_path = os.path.join(dir_path, "var_names.csv")

        if save_adata:
            # Convert storage format of adjacency matrix to be writable by
            # adata.write()
            if (
                "spatial_connectivities" in self.adata.obsp.keys()
                and self.adata.obsp["spatial_connectivities"] is not None
            ):
                self.adata.obsp["spatial_connectivities"] = sp.csr_matrix(
                    self.adata.obsp["spatial_connectivities"]
                )
            if (
                "connectivities" in self.adata.obsp.keys()
                and self.adata.obsp["connectivities"] is not None
            ):
                self.adata.obsp["connectivities"] = sp.csr_matrix(
                    self.adata.obsp["connectivities"]
                )
            self.adata.write(
                os.path.join(dir_path, adata_file_name), **anndata_write_kwargs
            )

        var_names = self.adata.var_names.astype(str).to_numpy()
        public_attributes = self._get_public_attributes()

        torch.save(self.model.state_dict(), model_save_path)
        np.savetxt(var_names_save_path, var_names, fmt="%s")
        save_model_with_fallback(public_attributes, attr_save_path)
        # with open(attr_save_path, "wb") as f:
        #     pickle.dump(public_attributes, f)

    @classmethod
    def load_query_data(
        cls,
        dir_path: str,
        query_adata: Optional[AnnData] = None,
        ref_adata_name: str = "adata_ref.h5ad",
        use_cuda: bool = True,
        unfreeze_all_weights: bool = False,
        unfreeze_eps_weight: bool = False,
        unfreeze_layer0: bool = False,
        **kwargs,
    ) -> torch.nn.Module:
        """
        Instantiate a model from saved output. Can be used for transfer learning
        scenarios and to learn de-novo gene programs by adding add-on gene
        programs and freezing non add-on weights.

        Parameters
        ----------
        dir_path:
            Path to saved outputs.
        adata:
            AnnData organized in the same way as data used to train the model.
            If ´None´, will check for and load adata saved with the model.
        adata_file_name:
            File name of the AnnData object to be loaded.
        use_cuda:
            If `True`, load model on GPU.
        unfreeze_all_weights:
            If `True`, unfreeze all weights.

        Returns
        -------
        model:
            Model with loaded state dictionaries and, if specified, frozen non
            add-on weights.
        """
        use_cuda = use_cuda and torch.cuda.is_available()
        map_location = torch.device("cpu") if use_cuda is False else None

        model_state_dict, var_names, attr_dict, adata_concat = load_saved_files(
            dir_path=dir_path,
            query_adata=query_adata,
            ref_adata_name=ref_adata_name,
            map_location=map_location,
        )
        # print('model_state_dict.keys() is', model_state_dict.keys())

        init_params = deepcopy(cls._get_init_params_from_dict(attr_dict))
        # don't preprocess the data
        init_params["min_features"] = 0
        init_params["adata_list"] = adata_concat
        init_params.update(kwargs)

        # model = initialize_model(cls, init_params)
        model = cls(init_params)

        # set saved attrs for loaded model
        for attr, val in init_params.items():
            setattr(model, attr, val)

        # TODO 可能缺少了 batch normalization的层参数，当 used_DSBN=False 时
        model.model.load_state_dict(model_state_dict, strict=False)

        if use_cuda:
            model.model.cuda()
        model.model.eval()

        # First freeze all parameters and then subsequently unfreeze based on
        # load settings
        for param_name, param in model.model.named_parameters():
            param.requires_grad = False
            model.freeze_ = True
        if unfreeze_all_weights:
            for param_name, param in model.model.named_parameters():
                param.requires_grad = True
            model.freeze_ = False
        if unfreeze_eps_weight:
            # allow updates of eps_weight
            for param_name, param in model.model.named_parameters():
                if "eps_weight" in param_name or "eps_bias" in param_name:
                    param.requires_grad = True
        if unfreeze_layer0:
            # Allow updates of the first embedder weights
            for param_name, param in model.model.named_parameters():
                if ("layers.0" in param_name) or ("norm.0" in param_name):
                    param.requires_grad = True

        return model

    def _check_if_trained(self, warn: bool = True):
        """
        Check if the model is trained.

        Parameters
        -------
        warn:
             If not trained and `warn` is True, raise a warning, else raise a
             RuntimeError.
        """
        message = (
            "Trying to query inferred values from an untrained model. "
            "Please train the model first."
        )
        if not self.is_trained_:
            if warn:
                warnings.warn(message)
            else:
                raise RuntimeError(message)
