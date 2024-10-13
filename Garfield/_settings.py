"""Configuration for Garfield"""
import os
import seaborn as sns
import matplotlib as mpl


class GarfieldConfig:
    """configuration class for Garfield"""

    def __init__(self, workdir="./result_Garfield", n_jobs=1):
        self.workdir = workdir
        self.n_jobs = n_jobs
        self.set_gf_params()

    def set_figure_params(
        self,
        context="notebook",
        style="white",
        palette="deep",
        font="sans-serif",
        font_scale=1.1,
        color_codes=True,
        dpi=80,
        dpi_save=150,
        fig_size=[5.4, 4.8],
        rc=None,
    ):
        """Set global parameters for figures. Modified from sns.set()

        Parameters
        ----------
        context : string or dict
            Plotting context parameters, see `seaborn.plotting_context`
        style: `string`,optional (default: 'white')
            Axes style parameters, see `seaborn.axes_style`
        palette : string or sequence
            Color palette, see `seaborn.color_palette`
        font_scale: `float`, optional (default: 1.3)
            Separate scaling factor to independently
            scale the size of the font elements.
        color_codes : `bool`, optional (default: True)
            If ``True`` and ``palette`` is a seaborn palette,
            remap the shorthand color codes (e.g. "b", "g", "r", etc.)
            to the colors from this palette.
        dpi: `int`,optional (default: 80)
            Resolution of rendered figures.
        dpi_save: `int`,optional (default: 150)
            Resolution of saved figures.
        rc: `dict`,optional (default: None)
            rc settings properties.
            Parameter mappings to override the values in the preset style.
            Please see "`matplotlibrc file
            <https://matplotlib.org/tutorials/introductory/customizing.html#a-sample-matplotlibrc-file>`__"
        """
        sns.set(
            context=context,
            style=style,
            palette=palette,
            font=font,
            font_scale=font_scale,
            color_codes=color_codes,
            rc={
                "figure.dpi": dpi,
                "savefig.dpi": dpi_save,
                "figure.figsize": fig_size,
                "image.cmap": "viridis",
                "lines.markersize": 6,
                "legend.columnspacing": 0.1,
                "legend.borderaxespad": 0.1,
                "legend.handletextpad": 0.1,
                "pdf.fonttype": 42,
            },
        )
        if rc is not None:
            assert isinstance(rc, dict), "rc must be dict"
            for key, value in rc.items():
                if key in mpl.rcParams.keys():
                    mpl.rcParams[key] = value
                else:
                    raise Exception("unrecognized property '%s'" % key)

    def set_workdir(self, workdir=None):
        """Set working directory.

        Parameters
        ----------
        workdir: `str`, optional (default: None)
            Working directory.

        Returns
        -------
        """
        if workdir is None:
            workdir = self.workdir
            print("Using default working directory.")
        if not os.path.exists(workdir):
            os.makedirs(workdir)
        self.workdir = workdir
        self.set_gf_params()
        print("Saving results in: %s" % workdir)

    def set_gf_params(self, config=None):
        """Set Garfield parameters

        Parameters
        ----------
        config : `dict`, optional
            Garfield training configuration parameters.

        Returns
        -------
        """
        default_config = {
            # Preprocessing options
            "adata_list": ["test.h5ad"],
            "profile": "RNA",
            "data_type": None,
            "sub_data_type": None,
            "sample_col": None,
            "weight": 0.8,
            "graph_const_method": "mu_std",
            "genome": None,
            "use_gene_weight": True,
            "user_cache_path": self.workdir,
            "use_top_pcs": False,
            "used_hvg": True,
            "min_features": 100,
            "min_cells": 3,
            "keep_mt": False,
            "target_sum": 1e4,
            "rna_n_top_features": 2000,
            "atac_n_top_features": 10000,
            "n_components": 50,
            "n_neighbors": 15,
            "metric": "euclidean",
            "svd_solver": "arpack",
            # datasets
            "used_pca_feat": False,
            "adj_key": "spatial_connectivities",
            # data split parameters
            "edge_val_ratio": 0.1,
            "edge_test_ratio": 0.0,
            "node_val_ratio": 0.1,
            "node_test_ratio": 0.0,
            # model parameters
            "augment_type": "svd",  # svd or dropout
            "svd_q": 5,  # if augment_type == 'svd'
            "use_FCencoder": True,
            "gnn_layer": 2,
            "conv_type": "GAT",
            "hidden_dims": [128, 128],
            "bottle_neck_neurons": 20,
            "cluster_num": 20,
            "num_heads": 3,
            "dropout": 0.2,
            "concat": True,
            "drop_feature_rate": 0.2,
            "drop_edge_rate": 0.2,
            "used_edge_weight": True,
            "used_DSBN": False,
            "used_mmd": False,
            # data loader parameters
            "num_neighbors": 3,
            "loaders_n_hops": 3,
            "edge_batch_size": 256,
            "node_batch_size": None,
            # loss parameters
            "include_edge_recon_loss": True,
            "include_gene_expr_recon_loss": True,
            "lambda_latent_contrastive_instanceloss": 1.0,
            "lambda_latent_contrastive_clusterloss": 0.5,
            "lambda_gene_expr_recon": 300.0,
            "lambda_latent_adj_recon_loss": 1.0,
            "lambda_edge_recon": 500.0,
            "lambda_omics_recon_mmd_loss": 0.2,
            # train parameters
            "n_epochs": 100,
            "n_epochs_no_edge_recon": 0,
            "learning_rate": 0.001,
            "weight_decay": 1e-05,
            "gradient_clipping": 5,
            # other parameters
            "latent_key": "garfield_latent",
            "reload_best_model": True,
            "use_early_stopping": True,
            "early_stopping_kwargs": None,
            "monitor": True,
            "device_id": 0,
            "seed": 2024,
            "verbose": False,
        }

        ## user_config
        if config is None:
            config = {}  # 如果 `config` 为空则设为空字典

        user_config = config
        config = {**default_config, **user_config}

        assert (
            config["adata_list"] is not None
        ), "The input `adata_list` must not be empty."
        assert config["profile"] in [
            "RNA",
            "ATAC",
            "ADT",
            "multi-modal",
            "spatial",
        ], "The `profile` should be set as one of the `RNA`, `ATAC`, `ADT`, `multi-modal`, `spatial`."
        if config["profile"] == "multi-modal":
            assert config["data_type"] in [
                "Paired",
                "UnPaired",
            ], "The `data_type` should be set as one of the `Paired`, `UnPaired`."
            assert config["sub_data_type"] in [
                ["rna", "atac"],
                ["rna", "adt"],
            ], 'The `sub_data_type` should be set as `["rna", "atac"]` or `["rna", "adt"]`.'
            if config["genome"] is not None:
                assert config["genome"] in [
                    "hg19",
                    "hg38",
                    "mm9",
                    "mm10",
                ], "`genome` must be one of ['hg19','hg38','mm9','mm10']"
        if config["profile"] == "spatial":
            assert config["data_type"] in [
                "single-modal",
                "multi-modal",
            ], "The `data_type` should be set as one of the `single-modal`, `multi-modal`."
        assert config["conv_type"] in [
            "GAT",
            "GATv2Conv",
            "GCN",
        ], "The current model only supports `GAT`, `GATv2Conv` or `GCN` layers."

        self.gf_params = config


settings = GarfieldConfig()
