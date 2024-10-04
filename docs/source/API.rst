.. automodule:: Garfield

API
===

Import Garfield as::

   import Garfield as gf

Configuration for Garfield
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   settings.set_figure_params
   settings.set_gf_params
   settings.set_workdir


Reading
~~~~~~~
.. module:: Garfield.data
.. currentmodule:: Garfield

.. autosummary::
   :toctree: generated

   data.initialize_dataloaders
   data.edge_level_split
   data.node_level_split_mask
   data.prepare_data
   data.GraphAnnTorchDataset

See more at `anndata <https://anndata.readthedocs.io/en/latest/api.html#reading>`_


Preprocessing
~~~~~~~~~~~~~
.. module:: Garfield.preprocessing
.. currentmodule:: Garfield

.. autosummary::
   :toctree: generated

   preprocessing.gene_scores
   preprocessing.get_nearest_neighbors
   preprocessing.preprocessing_rna
   preprocessing.preprocessing_atac
   preprocessing.preprocessing_adt
   preprocessing.preprocessing
   preprocessing.DataProcess


Model
~~~~~
.. module:: Garfield.model
.. currentmodule:: Garfield

.. autosummary::
   :toctree: generated

   model.utils.weighted_knn_trainer
   model.utils.weighted_knn_transfer
   model.Garfield.Garfield


Loss
~~~~
.. module:: Garfield.modules
.. currentmodule:: Garfield

.. autosummary::
   :toctree: generated

   modules.compute_omics_recon_mse_loss
   modules.compute_edge_recon_loss
   modules.compute_kl_reg_loss
   modules.compute_contrastive_instanceloss
   modules.compute_contrastive_clusterloss
   modules.compute_omics_recon_mmd_loss


Modules
~~~~~~~
.. module:: Garfield.modules
.. currentmodule:: Garfield

.. autosummary::
   :toctree: generated

   modules.GNNModelVAE


NN
~~
.. module:: Garfield.nn
.. currentmodule:: Garfield

.. autosummary::
   :toctree: generated

   nn.GATEncoder
   nn.GCNEncoder
   nn.GATDecoder
   nn.GCNDecoder
   nn.DSBatchNorm


Trainer
~~~~~~~
.. module:: Garfield.trainer
.. currentmodule:: Garfield

.. autosummary::
   :toctree: generated

   trainer.GarfieldTrainer
   trainer.eval_metrics
   trainer.plot_eval_metrics


Tools
~~~~~
.. module:: Garfield.trainer
.. currentmodule:: Garfield

.. autosummary::
   :toctree: generated

   trainer.EarlyStopping
   trainer.print_progress

