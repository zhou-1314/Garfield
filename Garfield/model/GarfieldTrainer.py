###### Trainer ##########
import warnings
from typing import Optional, Union
from types import SimpleNamespace
from collections import defaultdict
import os
import time
import matplotlib.pyplot as plt

import numpy as np
import random
import scanpy as sc
from anndata import concat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.sampler import NeighborSampler
from torch_geometric.loader import NeighborLoader, GraphSAINTEdgeSampler, NodeLoader
import torch_geometric.transforms as T

from .._settings import settings
# from ._utils import extract_subgraph
from ._utils import split_data
from ._tools import EarlyStopping, print_progress
from ._loss import VGAE_loss, InstanceLoss, ClusterLoss, mmd_loss_calc, unbalanced_ot, cosine_sim
from .prepare_Data import UserDataset
from .Garfield_net import Garfield
from .transfer_base import BaseMixin, SurgeryMixin
from .transfer_anno import weighted_knn_trainer, weighted_knn_transfer

# from torch.optim.lr_scheduler import StepLR
# from scheduler import CosineAnnealingWarmRestarts
# GAMMA=0.9
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")

class GarfieldTrainer(BaseMixin, SurgeryMixin):
    """
    Garfield model trainer.
    """
    def __init__(self, gf_params):
        """
        :param args: Arguments object.
        """
        if gf_params is None:
            gf_params = settings.gf_params.copy()
        else:
            assert isinstance(gf_params, dict), \
                "`gf_params` must be dict"

        self.args = SimpleNamespace(**gf_params)
        ## data preprocess parameter
        self.data_dir_ = self.args.data_dir
        self.project_name_ = self.args.project_name
        self.adata_list_ = self.args.adata_list
        self.profile_ = self.args.profile
        self.projection_ = self.args.projection
        self.data_type_ = self.args.data_type
        self.weight_ = self.args.weight
        self.genome_ = self.args.genome
        self.sample_col_ = self.args.sample_col
        self.filter_cells_rna_ = self.args.filter_cells_rna
        self.min_features_ = self.args.min_features
        self.min_cells_ = self.args.min_cells
        self.keep_mt_ = self.args.keep_mt
        self.normalize_ = self.args.normalize
        self.target_sum_ = self.args.target_sum
        self.used_hvg_ = self.args.used_hvg
        self.used_scale_ = self.args.used_scale
        self.single_n_top_genes_ = self.args.single_n_top_genes
        self.rna_n_top_features_ = self.args.rna_n_top_features
        self.atac_n_top_features_ = self.args.atac_n_top_features
        self.metacell_size_ = self.args.metacell_size
        self.metacell_ = self.args.metacell
        self.n_pcs_ = self.args.n_pcs
        self.n_neighbors_ = self.args.n_neighbors
        self.metric_ = self.args.metric
        self.svd_solver_ = self.args.svd_solver
        self.method_ = self.args.method
        self.resolution_tol_ = self.args.resolution_tol
        self.leiden_runs_ = self.args.leiden_runs
        self.leiden_seed_ = self.args.leiden_seed
        self.verbose_ = self.args.verbose

        ## model parameter
        self.epochs_ = self.args.epochs
        self.hidden_dims_ = self.args.hidden_dims
        self.bottle_neck_neurons_ = self.args.bottle_neck_neurons
        self.num_heads_ = self.args.num_heads
        self.dropout_ = self.args.dropout
        self.concat_ = self.args.concat
        self.svd_q_ = self.args.svd_q
        self.used_edge_weight_ = self.args.used_edge_weight
        self.used_mmd_ = self.args.used_mmd
        self.used_DSBN_ = self.args.used_DSBN
        self.used_recon_exp_ = self.args.used_recon_exp
        self.conv_type_ = self.args.conv_type
        self.cluster_num_ = self.args.cluster_num
        self.gnn_layer_ = self.args.gnn_layer
        self.instance_temperature_ = self.args.instance_temperature
        self.cluster_temperature_ = self.args.cluster_temperature
        self.mmd_temperature_ = self.args.mmd_temperature
        self.l2_reg_ = self.args.l2_reg
        self.patience_ = self.args.patience
        self.gradient_clipping_ = self.args.gradient_clipping
        self.learning_rate_ = self.args.learning_rate
        self.weight_decay_ = self.args.weight_decay

        ## data loader
        self.loader_type_ = self.args.loader_type
        self.num_neighbors_ = self.args.num_neighbors
        self.batch_size_ = self.args.batch_size
        self.test_split_ = self.args.test_split
        self.val_split_ = self.args.val_split

        self.monitor_only_val_losses_ = self.args.monitor_only_val_losses
        self.outdir_ = self.args.outdir
        self.impute_ = self.args.impute

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        seed = 2024
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.load_dataset()
        self.setup_model()

    # Load dataset
    def load_dataset(self):
        """
        Loading and processing dataset.
        """
        print("\nPreparing dataset...\n")
        self.training_graphs = UserDataset(root="{}/".format(self.data_dir_),
                                           project_name=self.project_name_,
                                           adata_list=self.adata_list_,
                                           profile=self.profile_,
                                           projection=self.projection_,
                                           data_type=self.data_type_,
                                           weight=self.weight_,
                                           genome=self.genome_,
                                           sample_col=self.sample_col_,
                                           filter_cells_rna=self.filter_cells_rna_,
                                           min_features=self.min_features_,
                                           min_cells=self.min_cells_,
                                           keep_mt=self.keep_mt_,
                                           normalize=self.normalize_,
                                           target_sum=self.target_sum_,
                                           used_hvg=self.used_hvg_,
                                           used_scale=self.used_scale_,
                                           single_n_top_genes=self.single_n_top_genes_,
                                           rna_n_top_features=self.rna_n_top_features_,
                                           atac_n_top_features=self.atac_n_top_features_,
                                           metacell_size=self.metacell_size_,
                                           metacell=self.metacell_,
                                           n_pcs=self.n_pcs_,
                                           n_neighbors=self.n_neighbors_,
                                           metric=self.metric_,
                                           svd_solver=self.svd_solver_,
                                           method=self.method_,
                                           resolution_tol=self.resolution_tol_,
                                           leiden_runs=self.leiden_runs_,
                                           leiden_seed=self.leiden_seed_,
                                           verbose=self.verbose_
                                           )
        self.merged_adata = self.training_graphs.merged_adata
        self.num_classes = self.training_graphs.num_classes ## batch number
        self.number_of_genes = self.training_graphs.num_features

        ## 划分 train_test_split_edges 构造Data对象
        self.training_graphs = self.testing_graphs = self.training_graphs[0]

    def setup_model(self):
        """
        Creating a Garfield.
        """
        self.model = Garfield(
            self.number_of_genes,
            self.hidden_dims_,
            self.bottle_neck_neurons_,
            self.num_classes,
            self.num_heads_,
            self.dropout_,
            self.concat_,
            self.svd_q_,
            self.used_edge_weight_,
            self.used_DSBN_,
            self.used_recon_exp_,
            self.conv_type_,
            self.cluster_num_,
            self.gnn_layer_
        )
        self.model = self.model.to(self.device)
        self.is_trained_ = False
        self.epoch = -1
        self.n_epochs = None
        self.training_time = 0
        self.logs = defaultdict(list) ## logs

        if self.outdir_ is not None:
            outdir = self.outdir_
        else:
            outdir = self.data_dir_
        os.makedirs(os.path.join(outdir, 'checkpoint'), exist_ok=True)

        self.criterion_instance = InstanceLoss(self.instance_temperature_, self.device).to(
            self.device)
        self.criterion_cluster = ClusterLoss(self.cluster_num_, self.cluster_temperature_, self.device).to(
            self.device)
        self.early_stopping = EarlyStopping(patience=self.patience_, checkpoint_file=os.path.join(outdir, 'checkpoint/model.pt'))

    # def save(self):
    #     """
    #     Saving model.
    #     """
    #     if self.args.outdir is not None:
    #         outdir = self.args.outdir
    #     else:
    #         outdir = self.args.data_dir
    #     torch.save(self.model.state_dict(), os.path.join(outdir, 'checkpoint/model.pt'))  # self.args.save
    #     print(f"Model is saved under {os.path.join(outdir, 'checkpoint/model.pt')}.")

    def load(self):
        """
        Loading model.
        """
        if self.outdir_ is not None:
            outdir = self.outdir_
        else:
            outdir = self.data_dir_

        self.model.load_state_dict(torch.load(os.path.join(outdir, 'checkpoint/model.pt')))
        print(f"Model is loaded from {os.path.join(outdir, 'checkpoint/model.pt')}.")

    def create_batches(self, is_train=True):
        """
        Creating batches from the training graph list.
        :return batches: training or testing loaders.
        """
        # Split into train/val/test
        # self.training_graphs.train_mask, self.training_graphs.val_mask, self.training_graphs.test_mask = split_data(self.training_graphs.y.size(0), self.val_split_, self.test_split_)

        if is_train:
            if self.loader_type_ == "neighbor":
                if self.num_neighbors_ is None:
                    self.num_neighbors_ = [3, 3]
                loader = NeighborLoader(self.training_graphs,
                                        num_neighbors=self.num_neighbors_,
                                        batch_size=self.batch_size_,
                                        input_nodes=torch.arange(self.training_graphs.num_nodes),
                                        shuffle=True)
            elif self.loader_type_ == "graphsaint":
                # loader = GraphSAINTRandomWalkSampler(data, batch_size = batch_size, walk_length = num_layers)
                loader = GraphSAINTEdgeSampler(self.training_graphs, batch_size=self.batch_size_, num_steps=16)
            else:
                raise NotImplementedError
        else:
            node_sampler = NeighborSampler(self.testing_graphs, num_neighbors=[0])  # num_neighbors=[0]表示不采样任何邻居
            # 设置 DataLoader
            loader = NodeLoader(
                self.testing_graphs,
                batch_size=self.batch_size_,
                shuffle=False,
                node_sampler=node_sampler
            )
        return loader

    def process_batch(self, batch_data, device):
        """
        Forward pass with a data.
        :param data: Data that is essentially pair of batches, for source and target graphs.
        :return loss: Loss on the data.
        """
        # Dont update any weight on first layers
        if self.model.freeze:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.BatchNorm1d):
                    if not module.weight.requires_grad:
                        module.affine = False
                        module.track_running_stats = False

        self.optimizer.zero_grad()
        ## calculate loss
        loss = self.calculate_losses(batch_data, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping_)
        self.optimizer.step()

    def validate_batch(self, val_data, test_data, device):
        # Get Train Epoch Logs
        for key in self.iter_logs:
            self.logs["epoch_" + key].append(np.array(self.iter_logs[key]).mean())

        # Validate Model
        vali_loss = self.validate(val_data, device)

        # AUC & precision
        self.test_metrics(test_data)

        # Monitor Logs
        if self.monitor_only_val_losses_:
            print_progress(self.epoch, self.logs, self.n_epochs, only_val_losses=True)
        else:
            print_progress(self.epoch, self.logs, self.n_epochs, only_val_losses=False)

        return vali_loss

    def calculate_losses(self, batch_data, device):
        """
        Calculate various losses for the batch data using the model.

        Parameters:
        - batch_data: The batch of data.
        - device: The device tensors are moved to.

        Returns:
        - all computed losses.
        """
        batch_mat = batch_data.x
        cell_batch = batch_data.y

        if self.used_recon_exp_:
            z, z_1, z_2, c_1, c_2, mu, logstd, recon_features = self.model(batch_data)
        else:
            z, z_1, z_2, c_1, c_2, mu, logstd = self.model(batch_data)

        # loss calculations
        # 1.VGAE ELBO loss
        vgae_loss = VGAE_loss(self.model.VGAE, z, mu, logstd, batch_data.pos_edge_label_index).to(device)

        # 2.Reconstruction loss
        if self.used_recon_exp_:
            ## total expr loss
            recon_loss = F.mse_loss(batch_mat, recon_features) * batch_mat.size(-1)

        # 3.Contrastive losses
        loss_instance = self.criterion_instance(z_1, z_2)
        loss_cluster = self.criterion_cluster(c_1, c_2)
        cl_loss = loss_instance + loss_cluster

        # 4.MMD loss
        mmd_loss = torch.tensor(0.0).to(device)
        if self.used_mmd_:
            cell_batch = cell_batch.detach().cpu()
            unique_groups, group_indices = np.unique(cell_batch, return_inverse=True)
            grouped_z_cell = {group: z[group_indices == i] for i, group in enumerate(unique_groups)}
            group_labels = list(unique_groups)
            num_groups = len(group_labels)

            for i in range(num_groups):
                for j in range(i + 1, num_groups):
                    z_i = grouped_z_cell[group_labels[i]]
                    z_j = grouped_z_cell[group_labels[j]]
                    mmd_loss_tmp = mmd_loss_calc(z_i, z_j)
                    mmd_loss += mmd_loss_tmp * z.size(0)

        # 5.Regularization loss
        regu_loss = torch.tensor(0.0).to(device)
        for param in self.model.parameters():
            regu_loss += param.norm(2).square()

        ### total loss
        loss = vgae_loss + recon_loss + cl_loss + regu_loss * self.l2_reg_ + mmd_loss * self.mmd_temperature_

        self.iter_logs["total_loss"].append(loss.item())
        self.iter_logs["vgae_loss"].append(vgae_loss.item())
        self.iter_logs["instance_loss"].append(loss_instance.item())
        self.iter_logs["cluster_loss"].append(loss_cluster.item())
        if self.used_recon_exp_:
            self.iter_logs["recon_loss"].append(recon_loss.item())
        if self.used_mmd_:
            self.iter_logs["mmd_loss"].append(mmd_loss.item())
        self.iter_logs["regu_loss"].append(regu_loss.item())

        return loss

    def fit(self):
        """
        Training a model.
        """
        print("\nModel training.\n")
        begin = time.time()
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(  # AdamW
            params,
            lr=self.learning_rate_,
            weight_decay=self.weight_decay_,
        )
        # self.scheduler = StepLR(self.optimizer, step_size=1, gamma=GAMMA)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        # )
        self.model.train()

        self.n_epochs = num_epochs = self.epochs_
        for self.epoch in range(num_epochs):
            self.iter_logs = defaultdict(list)
            loader = self.create_batches(is_train=True)
            for index, batch in enumerate(loader):
                if self.test_split_ > 0 and self.val_split_ > 0:
                    test_split = self.test_split_
                    val_split = self.val_split_
                    # Can set validation ratio
                    transform = T.RandomLinkSplit(num_val=val_split, num_test=test_split, is_undirected=True,
                                                  add_negative_train_samples=False, split_labels=True)
                    train_data, val_data, test_data = transform(batch)
                else:
                    train_data = batch.to(self.device)
                    # 直接使用 edge_index 作为 pos_edge_label_index
                    train_data.pos_edge_label_index = train_data.edge_index
                    val_data = test_data = train_data

                train_data = train_data.to(self.device)
                val_data = val_data.to(self.device)
                test_data = test_data.to(self.device)
                ## Loss Calculation(training mode)
                self.process_batch(train_data, self.device)
                torch.cuda.empty_cache()

                # Validation of Model, Monitoring, Early Stopping
                vali_loss = self.validate_batch(val_data, test_data, self.device)
                vali_loss = vali_loss.detach().cpu() #.numpy()
                # 在每个epoch结束时调用scheduler.step()来更新学习率
                # self.scheduler.step()

            ## early_stopping
            self.early_stopping(vali_loss, self.model)
            if self.early_stopping.early_stop:
                print('\n')
                print('EarlyStopping: run {} epoch'.format(self.epoch + 1))
                break

        self.is_trained_ = True
        self.training_time += (time.time() - begin)
        ## 保存模型
        # self.save()

    @classmethod
    def _get_init_params_from_dict(cls, dct):
        init_params = {
            # Input options
            'data_dir': dct['data_dir_'],
            'project_name': dct['project_name_'],
            'adata_list': dct['adata_list_'],
            'profile': dct['profile_'],
            'data_type': dct['data_type_'],
            'weight': dct['weight_'],
            'genome': dct['genome_'],
            'sample_col': dct['sample_col_'],

            ## whether to use metacell mode
            'metacell': dct['metacell_'],
            'metacell_size': dct['metacell_size_'],
            'single_n_top_genes': dct['single_n_top_genes_'],
            'n_pcs': dct['n_pcs_'],

            # Preprocessing options
            'filter_cells_rna': dct['filter_cells_rna_'],
            'min_features': dct['min_features_'],
            'min_cells': dct['min_cells_'],
            'keep_mt': dct['keep_mt_'],
            'normalize': dct['normalize_'],
            'target_sum': dct['target_sum_'],
            'used_hvg': dct['used_hvg_'],
            'used_scale': dct['used_scale_'],
            'rna_n_top_features': dct['rna_n_top_features_'],
            'atac_n_top_features': dct['atac_n_top_features_'],
            'n_neighbors': dct['n_neighbors_'],
            'svd_solver': dct['svd_solver_'],
            'method': dct['method_'],
            'metric': dct['metric_'],
            'resolution_tol': dct['resolution_tol_'],
            'leiden_runs': dct['leiden_runs_'],
            'leiden_seed': dct['leiden_seed_'],
            'verbose': dct['verbose_'],

            # Model options
            'gnn_layer': dct['gnn_layer_'],
            'conv_type': dct['conv_type_'],
            'hidden_dims': dct['hidden_dims_'],
            'bottle_neck_neurons': dct['bottle_neck_neurons_'],
            'svd_q': dct['svd_q_'],
            'cluster_num': dct['cluster_num_'],
            'num_heads': dct['num_heads_'],
            'concat': dct['concat_'],
            'used_edge_weight': dct['used_edge_weight_'],
            'used_recon_exp': dct['used_recon_exp_'],
            'used_DSBN': dct['used_DSBN_'],
            'used_mmd': dct['used_mmd_'],
            'test_split': dct['test_split_'],
            'val_split': dct['val_split_'],
            'batch_size': dct['batch_size_'],
            'loader_type': dct['loader_type_'],
            'num_neighbors': dct['num_neighbors_'],
            'epochs': dct['epochs_'],
            'dropout': dct['dropout_'],
            'mmd_temperature': dct['mmd_temperature_'],
            'instance_temperature': dct['instance_temperature_'],
            'cluster_temperature': dct['cluster_temperature_'],
            'l2_reg': dct['l2_reg_'],
            'patience': dct['patience_'],
            'monitor_only_val_losses': dct['monitor_only_val_losses_'],
            'gradient_clipping': dct['gradient_clipping_'],
            'learning_rate': dct['learning_rate_'],
            'weight_decay': dct['weight_decay_'],

            # Other options
            'projection': dct['projection_'],
            'impute': dct['impute_'],
            'outdir': dct['outdir_'],
            'load': False
        }

        return init_params

    def label_transfer(self,
                       ref_adata,
                       ref_adata_emb,
                       query_adata,
                       query_adata_emb,
                       ref_adata_obs,
                       label_keys,
                       n_neighbors=50,
                       threshold=1,
                       pred_unknown=False,
                       mode="package"):
        knn_transformer = weighted_knn_trainer(
            train_adata=ref_adata,
            train_adata_emb=ref_adata_emb,  # location of our joint embedding
            n_neighbors=n_neighbors,
        )

        labels, uncert = weighted_knn_transfer(
            query_adata=query_adata,
            query_adata_emb=query_adata_emb,  # location of our embedding, query_adata.X in this case
            label_keys=label_keys,  # (start of) obs column name(s) for which to transfer labels
            knn_model=knn_transformer,
            ref_adata_obs=ref_adata_obs,
            threshold=threshold,
            pred_unknown=pred_unknown,
            mode=mode
        )

        # 定义列名的映射
        if pred_unknown:
            rename_mapping_labels = {label_keys: f"transferred_{label_keys}_filtered"}
        else:
            rename_mapping_labels = {label_keys: f"transferred_{label_keys}_unfiltered"}
        rename_mapping_uncert = {label_keys: f"transferred_{label_keys}_uncert"}

        # 重命名列并加入到 'query_adata.obs'
        query_adata.obs = query_adata.obs.join(
            labels.rename(columns=rename_mapping_labels)
        )
        # 重命名列并加入到 'query_adata.obs'
        query_adata.obs = query_adata.obs.join(
            uncert.rename(columns=rename_mapping_uncert)
        )

        return query_adata

    @torch.no_grad()
    def validate(self, val_data, device):
        self.model.eval()
        self.iter_logs = defaultdict(list)

        ## validation loss
        vali_loss = self.calculate_losses(val_data, device)

        # Get Validation Logs
        for key in self.iter_logs:
            self.logs["val_" + key].append(np.array(self.iter_logs[key]).mean())

        self.model.train()

        return vali_loss

    @torch.no_grad()
    def test_metrics(self, test_data):
        self.model.eval()

        # Calculate test metrics
        roc_score, ap_score, acc, f1 = self.model.VGAE.single_test(test_data)

        ## roc_auc, ap info
        self.logs["test_" + 'roc_auc'].append(np.array(roc_score).mean())
        self.logs["test_" + 'precision'].append(np.array(ap_score).mean())
        self.logs["test_" + 'accuracy'].append(np.array(acc).mean())
        self.logs["test_" + 'f1_score'].append(np.array(f1).mean())

        self.model.train()

    ## Loss curve
    def plot_losses_curve(self, title="Training Losses Curve", show=True, save=False, dir_path=None):
        if save:
            show = False
            if dir_path is None:
                save = False

        fig = plt.figure()
        elbo_train = self.logs["epoch_total_loss"]
        elbo_test = self.logs["val_total_loss"]
        x = np.linspace(0, len(elbo_train), num=len(elbo_train))
        plt.plot(x, elbo_train, label="Train")
        plt.plot(x, elbo_test, label="Validate")
        plt.ylim(min(elbo_test) - 50, max(elbo_test) + 50)
        plt.legend()
        plt.title(title)
        if save:
            plt.savefig(f'{dir_path}.png', bbox_inches='tight')
        if show:
            plt.show()
        plt.clf()

    ## 获取隐变量
    @torch.no_grad()
    def get_latent(self):
        self.model.eval()
        print('eval mode')
        print('Perform get_latent for cells via mini-batch mode')

        cmu = None
        recon = None
        Data_test = self.create_batches(is_train=False)

        for batch in Data_test:
            batch = batch.to(self.device)
            if self.used_recon_exp_:
                if self.impute_:
                    print('impute mode')
                    _, x_recon = self.model.encodeBatch(batch)  # mu, x_recon
                    ## expression profile
                    if recon is None:
                        recon = x_recon.clone().detach().cpu()
                    else:
                        recon = torch.cat([recon, x_recon.cpu()], dim=0)
                else:
                    mu, _ = self.model.encodeBatch(batch)  # mu, x_recon
            else:
                mu = self.model.encodeBatch(batch)

            ## latent
            if cmu is None:
                cmu = mu.clone().detach().cpu()
            else:
                cmu = torch.cat([cmu, mu.cpu()], dim=0)

        latent = cmu.detach().numpy()

        if self.impute_:
            exp_mat = recon.detach().numpy()
            return latent, exp_mat
        else:
            return latent

    @torch.no_grad()
    def get_latent_representation(self):
        if self.impute_:
            self.merged_adata.obsm['X_gf'], self.merged_adata.layers['X_impute'] = self.get_latent()
        else:
            self.merged_adata.obsm['X_gf'] = self.get_latent()
        sc.pp.neighbors(self.merged_adata, use_rep='X_gf')
        # 保存数据
        if self.outdir_ is not None:
            outdir = self.outdir_
        else:
            outdir = self.data_dir_
        self.merged_adata.write(os.path.join(outdir, 'checkpoint/adata_ref.h5ad'), compression='gzip')

        return self.merged_adata



