###### Trainer ##########
import warnings
from typing import Optional, Union
from types import SimpleNamespace
from collections import defaultdict
import os
import time
# from tqdm import tqdm, trange
import numpy as np
import random
import scanpy as sc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.sampler import NeighborSampler
from torch_geometric.loader import NeighborLoader, DataLoader, NodeLoader
import torch_geometric.transforms as T

from .._settings import settings
from ._tools import EarlyStopping, print_progress
from ._loss import VGAE_loss, InstanceLoss, ClusterLoss, mmd_loss_calc
from .prepare_Data import UserDataset
from .Garfield_net import Garfield


warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")
class GarfieldTrainer(object):
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
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        #         self.device = torch.device("cpu")
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
        print("\nPreparing dataset.\n")
        self.training_graphs = UserDataset(root="{}/".format(self.args.data_dir),
                                           project_name=self.args.project_name,
                                           adata_list=self.args.adata_list,
                                           profile=self.args.profile,
                                           data_type=self.args.data_type,
                                           sample_col=self.args.sample_col,
                                           filter_cells_rna=self.args.filter_cells_rna,
                                           min_features=self.args.min_features,
                                           min_cells=self.args.min_cells,
                                           keep_mt=self.args.keep_mt,
                                           normalize=self.args.normalize,
                                           target_sum=self.args.target_sum,
                                           used_hvg=self.args.used_hvg,
                                           used_scale=self.args.used_scale,
                                           single_n_top_genes=self.args.single_n_top_genes,
                                           rna_n_top_features=self.args.rna_n_top_features,
                                           atac_n_top_features=self.args.atac_n_top_features,
                                           metacell_size=self.args.metacell_size,
                                           metacell=self.args.metacell,
                                           n_pcs=self.args.n_pcs,
                                           n_neighbors=self.args.n_neighbors,
                                           metric=self.args.metric,
                                           svd_solver=self.args.svd_solver,
                                           method=self.args.method,
                                           resolution_tol=self.args.resolution_tol,
                                           leiden_runs=self.args.leiden_runs,
                                           leiden_seed=self.args.leiden_seed,
                                           verbose=self.args.verbose
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
        self.model = Garfield(self.args, self.number_of_genes, self.num_classes)
        self.model = self.model.to(self.device)
        self.epoch = -1
        self.n_epochs = None
        self.training_time = 0
        self.logs = defaultdict(list) ## logs

        if self.args.outdir is not None:
            outdir = self.args.outdir
        else:
            outdir = self.args.data_dir
        os.makedirs(os.path.join(outdir, 'checkpoint'), exist_ok=True)

        self.early_stopping = EarlyStopping(patience=self.args.patience,
                                            checkpoint_file=os.path.join(outdir, 'checkpoint/model.pt'))
        self.criterion_instance = InstanceLoss(self.args.instance_temperature, self.device).to(
            self.device)
        self.criterion_cluster = ClusterLoss(self.args.cluster_num, self.args.cluster_temperature, self.device).to(
            self.device)

    def save(self):
        """
        Saving model.
        """
        if self.args.outdir is not None:
            outdir = self.args.outdir
        else:
            outdir = self.args.data_dir
        torch.save(self.model.state_dict(), os.path.join(outdir, 'checkpoint/model.pt'))  # self.args.save
        print(f"Model is saved under {os.path.join(outdir, 'checkpoint/model.pt')}.")

    def load(self):
        """
        Loading model.
        """
        if self.args.outdir is not None:
            outdir = self.args.outdir
        else:
            outdir = self.args.data_dir
        if self.args.load:
            self.model.load_state_dict(torch.load(os.path.join(outdir, 'checkpoint/model.pt')))
            print(f"Model is loaded from {os.path.join(outdir, 'checkpoint/model.pt')}.")

    def create_batches(self, is_train=True):
        """
        Creating batches from the training graph list.
        :return batches: Zipped loaders as list.
        """
        if is_train:
            if self.args.num_neighbors is None:
                self.num_neighbors = [3, 3]
            else:
                self.num_neighbors = self.args.num_neighbors
            loader = NeighborLoader(self.training_graphs,
                                    num_neighbors=self.num_neighbors,
                                    batch_size=self.args.batch_size,
                                    shuffle=True)
        else:
            node_sampler = NeighborSampler(self.testing_graphs, num_neighbors=[0])  # num_neighbors=[0]表示不采样任何邻居
            # 设置 DataLoader
            loader = NodeLoader(
                self.testing_graphs,
                batch_size=self.args.batch_size,
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
        self.optimizer.zero_grad()

        ## calculate loss
        loss = self.calculate_losses(batch_data, device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clipping)
        self.optimizer.step()

    def on_epoch_end(self, val_data, test_data, device):
        # Get Train Epoch Logs
        for key in self.iter_logs:
            self.logs["epoch_" + key].append(np.array(self.iter_logs[key]).mean())

        # Validate Model
        if self.args.test_split > 0 or self.args.val_split > 0:
            vali_loss = self.validate(val_data, test_data, device)

        # Monitor Logs
        if self.args.monitor_only_val_losses:
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
        b_gene = batch_data.x
        cell_batch = batch_data.y

        if self.args.used_recon_exp:
            z, z_1, z_2, c_1, c_2, mu, logstd, recon_features = self.model(batch_data)
        else:
            z, z_1, z_2, c_1, c_2, mu, logstd = self.model(batch_data)

        # loss calculations
        cell_batch = cell_batch.detach().cpu()
        unique_groups, group_indices = np.unique(cell_batch, return_inverse=True)

        # VGAE ELBO loss
        vgae_loss = VGAE_loss(self.model.VGAE, z, mu, logstd, batch_data.pos_edge_label_index).to(device)

        # Contrastive losses
        loss_instance = self.criterion_instance(z_1, z_2)
        loss_cluster = self.criterion_cluster(c_1, c_2)
        cl_loss = loss_instance + loss_cluster

        # Regularization loss
        regu_loss = torch.tensor(0.0).to(device)
        for param in self.model.parameters():
            regu_loss += param.norm(2).square()
        regu_loss *= self.args.l2_reg

        if self.args.used_recon_exp:
            recon_loss = F.mse_loss(b_gene, recon_features) * b_gene.size(-1)
        else:
            recon_loss = torch.tensor(0.0).to(device)

        # Total loss
        if self.args.used_mmd:
            # MMD loss
            grouped_z_cell = {group: z[group_indices == i] for i, group in enumerate(unique_groups)}
            group_labels = list(unique_groups)
            num_groups = len(group_labels)
            mmd_loss = torch.tensor(0.0).to(device)
            for i in range(num_groups):
                for j in range(i + 1, num_groups):
                    z_i = grouped_z_cell[group_labels[i]]
                    z_j = grouped_z_cell[group_labels[j]]
                    mmd_loss_tmp = mmd_loss_calc(z_i, z_j)
                    mmd_loss += mmd_loss_tmp
        else:
            mmd_loss = torch.tensor(0.0).to(device)

        ### total loss
        mmd_temperature = self.args.mmd_temperature
        loss = vgae_loss + cl_loss + recon_loss + regu_loss + mmd_loss * mmd_temperature  # 0.2

        self.iter_logs["total_loss"].append(loss.item())
        self.iter_logs["vgae_loss"].append(vgae_loss.item())
        self.iter_logs["instance_loss"].append(loss_instance.item())
        self.iter_logs["cluster_loss"].append(loss_cluster.item())
        self.iter_logs["recon_loss"].append(recon_loss.item())
        self.iter_logs["regu_loss"].append(regu_loss.item())
        self.iter_logs["mmd_loss"].append(mmd_loss.item())

        return loss

    def fit(self):
        """
        Training a model.
        """
        print("\nModel training.\n")
        begin = time.time()
        self.optimizer = torch.optim.Adam(  # AdamW
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        self.model.train()

        # epochs = trange(self.args.epochs, leave=True, desc="Epoch")
        self.n_epochs = self.args.epochs
        num_epochs = self.args.epochs
        for self.epoch in range(num_epochs):
            self.iter_logs = defaultdict(list)
            loader = self.create_batches(is_train=True)
            for index, batch in enumerate(loader):
                batch.train_mask = batch.val_mask = batch.test_mask = None
                if (self.args.test_split is not None):
                    test_split = self.args.test_split
                    val_split = self.args.val_split
                else:
                    # PyTorch Geometric does not allow 0 training samples (all test), so we need to store all test data as 'training'.
                    test_split = 0.0
                    val_split = 0.0

                # Can set validation ratio
                try:
                    transform = T.RandomLinkSplit(num_val=val_split, num_test=test_split, is_undirected=True,
                                                  add_negative_train_samples=False,
                                                  split_labels=True)  # add_negative_train_samples
                    train_data, val_data, test_data = transform(batch)
                    train_data = train_data.to(self.device)
                    val_data = val_data.to(self.device)
                    test_data = test_data.to(self.device)
                except IndexError as ie:
                    print()
                    print(colored('Exception: ' + str(ie), 'red'))
                    sys.exit(1)

                ## Loss Calculation(training mode)
                self.process_batch(train_data, self.device)
                torch.cuda.empty_cache()

                # Validation of Model, Monitoring, Early Stopping
                vali_loss = self.on_epoch_end(val_data, test_data, self.device)
                vali_loss = vali_loss.detach().cpu()#.numpy()

            ## early_stopping
            self.early_stopping(vali_loss, self.model)
            if self.early_stopping.early_stop:
                print('\n')
                print('EarlyStopping: run {} epoch'.format(self.epoch + 1))
                break

        ## 保存模型
        self.training_time += (time.time() - begin)
        self.save()

    @torch.no_grad()
    def validate(self, val_data, test_data, device):
        self.model.eval()
        self.iter_logs = defaultdict(list)

        # Calculate Validation Losses
        roc_auc, ap = self.model.VGAE.single_test(test_data)

        ## test loss
        vali_loss = self.calculate_losses(val_data, device)

        # Get Validation Logs
        for key in self.iter_logs:
            self.logs["val_" + key].append(np.array(self.iter_logs[key]).mean())
        ## roc_auc, ap info
        self.logs["test_" + 'roc_auc'].append(np.array(roc_auc).mean())
        self.logs["test_" + 'precision'].append(np.array(ap).mean())

        self.model.train()

        return vali_loss

    ## 获取隐变量
    @torch.no_grad()
    def get_latent(self):
        self.model.eval();
        print('eval mode')
        print('Perform get_latent for cells via mini-batch mode')

        cmu = None
        Data_test = self.create_batches(is_train=False)

        for batch in Data_test:
            batch = batch.to(self.device)
            if self.args.used_recon_exp:
                mu, _ = self.model.encodeBatch(batch)  # mu, x_recon
            else:
                mu = self.model.encodeBatch(batch)

            if cmu is None:
                cmu = mu.clone().detach().cpu()
            else:
                cmu = torch.cat([cmu, mu.cpu()], dim=0)

        output = cmu.detach().numpy()
        return output

    @torch.no_grad()
    def get_latent_representation(self):
        self.merged_adata.obsm['X_gf'] = self.get_latent()
        sc.pp.neighbors(self.merged_adata, use_rep='X_gf')
        return self.merged_adata


