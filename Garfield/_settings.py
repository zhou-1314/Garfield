"""Configuration for Garfield"""
import os
import seaborn as sns
import matplotlib as mpl

class GarfieldConfig:
    """configuration class for Garfield"""
    def __init__(self,
                 workdir='./result_Garfield',
                 save_fig=False,
                 n_jobs=1):
        self.workdir = workdir
        self.save_fig = save_fig
        self.n_jobs = n_jobs
        self.set_gf_params()

    def set_figure_params(self,
                          context='notebook',
                          style='white',
                          palette='deep',
                          font='sans-serif',
                          font_scale=1.1,
                          color_codes=True,
                          dpi=80,
                          dpi_save=150,
                          fig_size=[5.4, 4.8],
                          rc=None):
        """ Set global parameters for figures. Modified from sns.set()

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
        sns.set(context=context,
                style=style,
                palette=palette,
                font=font,
                font_scale=font_scale,
                color_codes=color_codes,
                rc={'figure.dpi': dpi,
                    'savefig.dpi': dpi_save,
                    'figure.figsize': fig_size,
                    'image.cmap': 'viridis',
                    'lines.markersize': 6,
                    'legend.columnspacing': 0.1,
                    'legend.borderaxespad': 0.1,
                    'legend.handletextpad': 0.1,
                    'pdf.fonttype': 42,
                    })
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
        if(workdir is None):
            workdir = self.workdir
            print("Using default working directory.")
        if(not os.path.exists(workdir)):
            os.makedirs(workdir)
        self.workdir = workdir
        self.set_gf_params()
        print('Saving results in: %s' % workdir)

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
            # Input options
            'data_dir': 'data',
            'project_name': 'test',
            'adata_list': ['test.h5ad'],
            'profile': 'RNA',
            'data_type': None,
            'weight': 0.5,
            'genome': None,
            'sample_col': None,

            ## whether to use metacell mode
            'metacell': False,
            'metacell_size': 1,
            'single_n_top_genes': 2000,
            'n_pcs': 20,

            # Preprocessing options
            'filter_cells_rna': True,
            'min_features': 100,
            'min_cells': 3,
            'keep_mt': False,
            'normalize': True,
            'target_sum': 1e4,
            'used_hvg': True,
            'used_scale': True,
            'rna_n_top_features': 2000,
            'atac_n_top_features': 10000,
            'n_neighbors': 15,
            'svd_solver': 'arpack',
            'method': 'umap',
            'metric': 'euclidean',
            'resolution_tol': 0.1,
            'leiden_runs': 1,
            'leiden_seed': None,
            'verbose': True,

            # Model options
            'gnn_layer': 2,
            'conv_type': 'GAT',
            'hidden_dims': [128, 128],
            'bottle_neck_neurons': 20,
            'svd_q': 5,
            'cluster_num': 20,
            'num_heads': 3,
            'concat': True,
            'used_edge_weight': True,
            'used_recon_exp': True,
            'used_DSBN': False,
            'used_mmd': False,
            'test_split': 0.1,
            'val_split': 0.1,
            'batch_size': 128,
            'num_neighbors': [3, 3],
            'epochs': 50,
            'dropout': 0.2,
            'mod_temperature': 0.6,  ## RNA
            'mmd_temperature': 0.2,
            'instance_temperature': 1.0,
            'cluster_temperature': 0.5,
            'l2_reg': 1e-03,
            'patience': 5,
            'monitor_only_val_losses': True,
            'gradient_clipping': 5,
            'learning_rate': 0.001,
            'weight_decay': 1e-05,

            # Other options
            'outdir': None,
            'load': False
        }

        ## user_config
        if config is None:
            config = {}  # 如果 `config` 为空则设为空字典

        user_config = config
        config = {**default_config, **user_config}

        assert config['adata_list'] is not None, 'The input `adata_list` must not be empty.'
        assert config['profile'] in ['RNA', 'ATAC', 'multi-modal'], \
            'The `profile` should be set as one of the `RNA`, `ATAC`, `multi-modal`.'
        if config['profile'] == 'multi-modal':
            assert config['data_type'] in ['Paired', 'UnPaired'], \
                'The `data_type` should be set as one of the `Paired`, `UnPaired`.'
            if config['data_type'] == 'UnPaired' and config['genome'] is not None:
                assert (config['genome'] in ['hg19', 'hg38', 'mm9', 'mm10']), \
                    "`genome` must be one of ['hg19','hg38','mm9','mm10']"
        assert config['conv_type'] in ['GAT', 'GCN'], \
            'The current model only supports `GAT` or `GCN` layers.'

        self.gf_params = config


settings = GarfieldConfig()

