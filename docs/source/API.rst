.. automodule:: Garfield

API
===

Import Garfield as::

   import Garfield as gf

Configuration for Garfield
~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary

   settings.set_figure_params
   settings.set_gf_params
   settings.set_workdir


Reading
~~~~~~~

.. autosummary::
   :toctree: _autosummary

   preprocessing.read_mtx
   preprocessing.read_scData
   preprocessing.read_multi_scData
   preprocessing.concat_data

See more at `anndata <https://anndata.readthedocs.io/en/latest/api.html#reading>`_


Preprocessing
~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   preprocessing.filter_genes
   preprocessing.cal_qc_rna
   preprocessing.filter_cells_rna
   preprocessing.preprocessing_rna
   preprocessing.preprocessing_atac
   preprocessing.preprocessing
   preprocessing.get_centroids
   preprocessing.summarize_clustering
   preprocessing.GeneScores
   preprocessing.gene_scores
   preprocessing.construct_graph_rna
   preprocessing.graph_clustering


DataLoader
~~~~~

.. autosummary::
   :toctree: _autosummary

   model.UserDataset


Model
~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   Model.GarfieldTrainer
   Model.Garfield
   Model.GCNModelVAE
   Model.DSBatchNorm
   Model.GATEncoder
   Model.GATDecoder
   Model.GCNEncoder
   Model.GCNDecoder


Loss
~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   Model.VGAE_loss
   Model.InstanceLoss
   Model.ClusterLoss
   Model.mmd_loss_calc


Evaluation
~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   Model.batch_entropy_mixing_score
   Model.silhouette


Tools
~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   Model.label_transfer
   Model.EarlyStopping
   Model.Transfer_scData

