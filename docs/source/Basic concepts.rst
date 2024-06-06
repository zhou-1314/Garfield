================
Basic parameter
================


+----------------------+------------------+--------------------------------------------------------+-------------+
| Parameter type       | Name             | Explanation                                            | Input Type  |
+======================+==================+========================================================+=============+
| Input options        | data_dir         | The directory where the graph dataset should be stored.| str         |
|                      | project_name     | Name of the project for the dataset.                    | str         |
|                      | adata_list       | List of AnnData objects representing different batches of data. | list |
|                      | profile          | Type of data profile, e.g., 'RNA', 'ATAC', 'multi-modal'. | str     |
|                      | data_type        | Type of multi-modal data, e.g., 'Paired', 'UnPaired'. Default is None. | str |
|                      | genome           | Genome reference name, e.g., 'hg19', 'hg38', 'mm9', 'mm10'. Default is None, but if data_type is set as 'UnPaired', genome must be set. | str |
|                      | sample_col       | Column name that defines different samples. Default is 'batch'. | str |
+----------------------+------------------+--------------------------------------------------------+-------------+
| Preprocessing options| metacell         | Whether to construct metacells. Default is True.       | bool        |
|                      | metacell_size    | Size of metacell clusters, which will determine the number of metacell clusters. Default is 2. | int  |
|                      | n_pcs            | Number of principal components to use for each sample processing. Default is 20. | int |
|                      | filter_cells_rna | Whether to filter cells based on RNA expression. Default is False. | bool |
|                      | min_features     | Minimum number of features required for a cell. Default is 100. | int |
|                      | min_cells        | Minimum number of cells required for a feature. Default is 3. | int |
|                      | keep_mt          | Whether to keep mitochondrial genes. Default is False. | bool       |
|                      | normalize        | Whether to normalize the data. Default is True.        | bool        |
|                      | target_sum       | Target sum for total count normalization. Default is 1e4. | float    |
|                      | used_hvg         | Whether to use highly variable genes. Default is True. | bool        |
|                      | used_scale       | Whether to scale the data. Default is True.            | bool        |
|                      | single_n_top_genes | Number of top genes to select for single-sample analysis. Default is 2000. | int |
|                      | rna_n_top_features | Number of top features to select for RNA profile. Default is 3000. | int |
|                      | atac_n_top_features | Number of top features to select for ATAC profile. Default is 10000. | int |
|                      | n_neighbors      | Number of neighbors for graph construction. Default is 15. | int |
|                      | svd_solver       | SVD solver to use for PCA. Default is 'arpack'.        | str         |
|                      | method           | Dimensionality reduction method. Default is 'umap'.    | str         |
|                      | metric           | Distance metric for nearest neighbor search. Default is 'correlation'. | str |
|                      | resolution_tol   | Tolerance level for resolution clustering. Default is 0.1. | float    |
|                      | leiden_runs      | Number of times to run Leiden clustering. Default is 1. | int |
|                      | leiden_seed      | Seed for the Leiden algorithm. Default is None.        | int         |
|                      | verbose          | Whether to print progress information. Default is True.| bool        |
+----------------------+------------------+--------------------------------------------------------+-------------+
| Model options        | gnn_layer        | The number of repeat times for GNN layers forward propagation | int |
|                      | conv_type        | The convolution module used in the variational graph autoencoder. 'GAT' or 'GCN' | str |
|                      | hidden_dims      | List of output dimensions for each hidden layer in the GAT or GCN. | list[int] |
|                      | bottle_neck_neurons | Dimension of the latent feature representation produced by the encoder. | int |
|                      | svd_q            | Rank for the low-rank SVD approximation. Default is 5. | int         |
|                      | cluster_num      | The decoder outputs the number of predicted categories. Default is 20. | int |
|                      | used_edge_weight | Whether to use edge weights in the GAT or GCN layers (default is True). | bool |
|                      | used_recon_exp   | Whether to use expression profile reconstruction in the decoder layer. Default is True. | bool |
|                      | used_DSBN        | Whether to use domain-specific batch normalization (DSBN) (default is False). But for batch correction or multi-omics integration, it needs to be set to True. | bool |
|                      | used_mmd         | Whether to use mmd for data alignment. Default is False. But for batch correction or multi-omics integration, it needs to be set to True | bool |
|                      | num_heads        | Number of attention heads for each GAT layer.          | int         |
|                      | concat           | Whether to concatenate outputs of all attention heads or not. | bool |
|                      | test_split       | The number of test edges. If set to a floating-point value in 0-1, it represents the ratio of edges to include in the testing set. (default: 0.1). | float |
|                      | val_split        | The number of validation edges. If set to a floating-point value in 0-1, it represents the ratio of edges to include in the validation set. (default: 0.1). | float |
|                      | batch_size       | Number of samples per batch to load when model training. Default: 128. | int |
|                      | num_neighbors    | The number of neighbors to sample for each node in each iteration. | bool |
|                      | epochs           | The maximum number of model training.                   | int         |
|                      | dropout          | Dropout rate for model.                                | float       |
|                      | mmd_temperature  | Temperature coefficient of mmd loss. Default is 0.2.   | float       |
|                      | instance_temperature | Temperature coefficient of instance loss. Default is 1.0. | float |
|                      | cluster_temperature | Temperature coefficient of cluster loss. Default is 0.5. | float |
|                      | l2_reg           | Temperature coefficient of L2 regularization loss. Default is 1e-03. | float |
|                      | patience         | How long to wait after last time loss improved. Default: 10 | int |
|                      | monitor_only_val_losses | Whether only to monitor the loss changes in the validation set. Default is True. | bool |
|                      | gradient_clipping| Gradient clipping to prevent gradient explosion.      | float       |
|                      | learning_rate    | Learning rate of model. Default is 0.001.             | float       |
|                      | weight_decay     | weight decay. Default is 1e-05.                        | float       |
+----------------------+------------------+--------------------------------------------------------+-------------+
| Other options        | outdir           | Model output directory, if None, it will be consistent with the data_dir parameter. | str |
|                      | load             | Whether to load the existing model.                    | bool        |
+----------------------+------------------+--------------------------------------------------------+-------------+





