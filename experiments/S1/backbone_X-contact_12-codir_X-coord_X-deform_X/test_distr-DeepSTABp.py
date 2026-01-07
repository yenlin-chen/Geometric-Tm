import os, sys, random, torch, torchinfo
import numpy as np
import torch_geometric as pyg
import matplotlib.pyplot as plt

self_dir = os.getcwd()
root_dir = os.path.normpath(self_dir + '/..' * 3)
package_dir = os.path.join(root_dir, 'src')
sys.path.append(package_dir)

from ml_modules.training.model_arch import S1_M1, M2
from ml_modules.training.trainer import Trainer
from ml_modules.training.metrics import pcc, rmse, mae, mse, r2
from ml_modules.data.datasets import Dataset


# fix random generator for reproducibility
random_seed = 69
torch.manual_seed(random_seed)
rand_gen = torch.Generator().manual_seed(random_seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# def backbone_merge(data):
#     if data['residue', 'backbone-complementary', 'residue'].edge_index.numel():
#         data['residue', 'merged', 'residue'].edge_index = torch.cat((
#             data['residue', 'merged', 'residue'].edge_index,
#             data['residue', 'backbone-complementary', 'residue'].edge_index
#         ), dim=1)
#     return data


# for multiprocessing
def main():

    test_set_name = 'test set (2,044 entries)'

    merge_edge_types = False
    # if merge_edge_types:
    #     edge_types_to_merge = ['backbone', 'contact', 'codir']

    edge_types_to_use = ['contact']
    edge_policy = '1CONT'
    model_type = 'S1_M1'

    dataset_version = 'v7a'
    entries_should_be_ready = False

    # transform = backbone_merge if 'backbone' in edge_types_to_merge else None

    ####################################################################
    # AUTO-CONFIGURATION
    ####################################################################

    if model_type == 'M2':
        graph_dims = edge_types_to_use  # M2 only
        mgcn_hidden_channels = 256  # M2 only
        dim_node_hidden_ls = [32]  # M2 only

        assert len(graph_dims) > 1

    elif model_type == 'S1_M1':
        dim_node_hidden_dict = {  # S1_M1 only
            et: [32] for et in edge_types_to_use
        }

        assert len(edge_types_to_use) == 1

    else:
        raise ValueError(f'Unknown model type: {model_type}')

    thresholds = {
        'contact': '12',
        'codir': edge_policy,
        'coord': edge_policy,
        'deform': edge_policy,
    }
    embedding = 'prottrans'

    n_folds = 10

    use_ogt = True
    use_pi = False

    batch_size = 64
    learning_rate = 0.01

    ####################################################################
    # EXPERIMENT SETUP
    ####################################################################

    loss_type = 'mse'
    metrics = {'pcc': pcc, 'rmse': rmse, 'mae': mae, 'mse': mse, 'r2': r2}

    save_dir = f'test - {test_set_name}'
    os.makedirs(f'{save_dir}', exist_ok=True)

    # file to save training history
    prediction_file_test = f'{save_dir}/predicted_values-test_set.csv'

    # machine-specific parameters
    num_workers = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {num_workers} workers for dataloading')
    print(f'Training on {device} device\n')

    ### DATASET
    exp_root_dir = os.path.normpath(self_dir + '/..' * 2)
    test_meta_file = os.path.join(
        root_dir,
        'datasets',
        f'metadata - {test_set_name}.csv'
    )

    test_set = Dataset(
        meta_file=test_meta_file,
        version=dataset_version,
        thresholds=thresholds,
        merge_edge_types=merge_edge_types,
        sequence_embedding=embedding,
        time_limit=60,
        transform=None,
        device=device,
        entries_should_be_ready=entries_should_be_ready,
    )
    assert (
        len(test_set)
        == np.loadtxt(test_meta_file, dtype=np.str_, delimiter=',').shape[0]
    )

    test_size = len(test_set)
    test_accessions_ordered = test_set.processable_accessions
    test_order = {test_accessions_ordered[i]: i for i in range(test_size)}

    test_Tm = [test_set.Tm_dict[a] for a in test_accessions_ordered]

    ### EXPORT ENTRY IDENTIFIERS FOR FUTURE REFERENCE
    np.savetxt(
        f'{save_dir}/test_entries.txt', test_accessions_ordered, fmt='%s'
    )

    ### DATASET STATISTICS
    test_min, test_max = np.amin(test_Tm), np.amax(test_Tm)
    test_mean, test_std = np.mean(test_Tm), np.std(test_Tm)
    test_median = np.median(test_Tm)

    ### SAVE TO FILE
    # dataset statistics
    lines = [
        '# dataset,min,max,mean,std,median',
        f'test,{test_min},{test_max},{test_mean},{test_std},{test_median}',
    ]
    with open(f'{save_dir}/dataset_statistics.csv', 'w+') as f:
        f.write('\n'.join(lines) + '\n')

    # prediction on test set
    header = '# epoch,' + ','.join(test_accessions_ordered)
    line_1 = '# true_labels,' + ','.join([f'{Tm:.8f}' for Tm in test_Tm])
    with open(prediction_file_test, 'w+') as f:
        f.write(header + '\n')
        f.write(line_1 + '\n')

    ####################################################################
    # INSTANTIATE MODEL, OPTIMIZER, AND LOSS
    ####################################################################
    if model_type == 'M2':
        model = M2(  # M2 only
            # FEATURE SELECTION
            graph_dims=graph_dims,  # M2 only
            use_ogt=use_ogt,
            use_pi=use_pi,
            # GRAPH CONVOLUTION SETUP
            node_feat_name='x',
            node_feat_size=1024,
            # gnn_type='gcn',  # S1_M1 only
            # gat_atten_heads=None,  # S1_M1 only
            # dim_node_hidden_dict=dim_node_hidden_dict,  # S1_M1 only
            dim_node_hidden_ls=dim_node_hidden_ls,  # M2 only
            mgcn_hidden_channels=mgcn_hidden_channels,  # M2 only
            conv_norm=True,
            norm_graph_input=False,
            norm_graph_output=False,
            graph_global_pool='mean',
            graph_dropout_rate=0,
            dropfeat_rate=0,
            dropedge_rate=0,
            dropnode_rate=0,
            jk_concat=None,
            # PERSISTENCE IMAGES SETUP
            pi_dropout_rate=0 if use_pi else None,
            dim_pi_embedding=32 if use_pi else None,
            # OGT EMBEDDING SETUP
            embed_ogt=True if use_ogt else None,
            ogt_dropout_rate=0 if use_ogt else None,
            # FC SETUP
            fc_hidden_ls=None,
            n_fc_hidden_layers=2,
            fc_norm=True,
            norm_fc_input=False,
            fc_dropout_rate=0,
            # OTHERS
            debug=False,
        )
    else:
        model = S1_M1(  # S1_M1 only
            # FEATURE SELECTION
            # graph_dims=graph_dims,  # M2 only
            use_ogt=use_ogt,
            use_pi=use_pi,
            # GRAPH CONVOLUTION SETUP
            node_feat_name='x',
            node_feat_size=1024,
            gnn_type='gcn',  # S1_M1 only
            gat_atten_heads=None,  # S1_M1 only
            dim_node_hidden_dict=dim_node_hidden_dict,  # S1_M1 only
            # dim_node_hidden_ls=dim_node_hidden_ls,  # M2 only
            # mgcn_hidden_channels=mgcn_hidden_channels,  # M2 only
            conv_norm=True,
            norm_graph_input=False,
            norm_graph_output=False,
            graph_global_pool='mean',
            graph_dropout_rate=0,
            dropfeat_rate=0,
            dropedge_rate=0,
            dropnode_rate=0,
            jk_concat=None,
            # PERSISTENCE IMAGES SETUP
            pi_dropout_rate=0 if use_pi else None,
            dim_pi_embedding=32 if use_pi else None,
            # OGT EMBEDDING SETUP
            embed_ogt=True if use_ogt else None,
            ogt_dropout_rate=0 if use_ogt else None,
            # FC SETUP
            fc_hidden_ls=None,
            n_fc_hidden_layers=2,
            fc_norm=True,
            norm_fc_input=False,
            fc_dropout_rate=0,
            # OTHERS
            debug=False,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, amsgrad=False
    )
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.CyclicLR(
    #     optimizer, base_lr=0.0001, max_lr=0.01, mode='triangular',
    #     cycle_momentum=False
    # )
    if loss_type == 'mae':
        loss_fn = torch.nn.L1Loss(reduction='sum')
    elif loss_type == 'mse':
        loss_fn = torch.nn.MSELoss(reduction='sum')
    elif loss_type == 'smae':
        loss_fn = torch.nn.SmoothL1Loss(reduction='sum')
    else:
        raise ValueError(f'Unknown loss type: {loss_type}')

    # information on model architecture
    model.save_args(f'{save_dir}')

    # placeholders for ensemble
    all_bte_losses = np.empty((n_folds, 1))
    all_bte_outputs = np.empty((n_folds, test_size))
    all_bte_labels = np.empty((n_folds, test_size))

    for fold_idx in range(n_folds):

        print()
        print(f' >> Fold {fold_idx} <<')

        fold_dir = f'fold{fold_idx}'

        ################################################################
        # INSTANTIATE DATASET AND DATALOADERS
        ################################################################

        # DATALOADER
        test_loader = pyg.loader.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=rand_gen,
        )

        ### DATASET STATISTICS
        train_statistics = np.loadtxt(
            f'{fold_dir}/dataset_statistics.csv',
            delimiter=',',
            usecols=(1, 2, 3, 4, 5),
            dtype=np.float_,
        )
        train_min, train_max = train_statistics[0, :2]
        train_mean, train_std = train_statistics[0, 2:4]
        train_median = train_statistics[0, 4]

        # save to file
        line = (
            f'train_{fold_idx},{train_min},{train_max},{train_mean},'
            f'{train_std},{train_median}'
        )
        with open(f'{save_dir}/dataset_statistics.csv', 'a+') as f:
            f.write(line + '\n')

        ################################################################
        # train / valid loop
        ################################################################

        ### INSTANTIATE THE MODEL-TRAINING CONTROLLER
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            num_workers=num_workers,
            device=device,
            # CAUTION: TO AVOID DATA LEAKAGE, DO NOT USE VALIDATION SET
            min_max=(train_min, train_max),
            mean_std=(train_mean, train_std),
        )

        ################################################################
        # plot true vs pred on best epoch
        ################################################################

        trainer.load_model_state_dict(
            torch.load(f'{fold_dir}/model-best.pt', map_location=device)
        )

        ### ONE PASS OVER TEST SET
        bte_loss, bte_outputs, bte_labels, bte_accessions = trainer.evaluate(
            test_loader
        )

        ### SAVE PREDICTION FOR TEST SET
        # order outputted values by acccession
        idx_order = np.argsort([test_order[a] for a in bte_accessions.tolist()])
        bte_loss_ordered = bte_loss
        bte_outputs_ordered = bte_outputs.detach().cpu().numpy()[idx_order]
        bte_labels_ordered = bte_labels.detach().cpu().numpy()[idx_order]

        line = f'{fold_idx},' + ','.join(bte_outputs_ordered.astype(np.str_))
        with open(prediction_file_test, 'a+') as f:
            f.write(line + '\n')

        all_bte_losses[fold_idx] = bte_loss_ordered
        all_bte_outputs[fold_idx] = bte_outputs_ordered
        all_bte_labels[fold_idx] = bte_labels_ordered

        # compute various metrics
        bte_metrics = {
            m: metrics[m](
                torch.tensor(bte_outputs_ordered),
                torch.tensor(bte_labels_ordered)
            )
            for m in metrics
        }

        # save metrics to file
        np.savetxt(
            f'{fold_dir}/{save_dir}-test_performance.csv',
            [list(bte_metrics.values())],
            delimiter=',',
            header=','.join(bte_metrics.keys()),
        )

        # ### PLOTS
        # plt.scatter(
        #     bte_labels_ordered,
        #     bte_outputs_ordered,
        #     marker='x', s=1, alpha=0.7, zorder=3
        # )
        # plt.plot(
        #     np.linspace(30, 95),
        #     np.linspace(30, 95),
        #     '--',
        #     c='k',
        #     alpha=0.3,
        #     zorder=1,
        # )
        # plt.title(
        #     f'mae: {bte_metrics["mae"]:.3f}, rmse: {bte_metrics["rmse"]:.3f}, \n'
        #     rf'$r^2$: {bte_metrics["r2"]:.3f}, pcc: {bte_metrics["pcc"]:.3f}'
        # )
        # plt.xlabel(r'true $T_m$ (°C)')
        # plt.ylabel(r'predicted $T_m$ (°C)')
        # plt.grid()
        # plt.gca().set_aspect('equal')
        # plt.savefig(f'{fold_dir}/{save_dir}-true_vs_pred.png', dpi=300, bbox_inches='tight')
        # plt.close()

    ####################################################################
    # COMPUTE ENESEMBLE
    ####################################################################

    bte_outputs = np.mean(all_bte_outputs, axis=0)
    bte_labels = np.mean(all_bte_labels, axis=0)

    # compute various metrics
    bte_metrics = {
        m: metrics[m](
            torch.tensor(bte_outputs),
            torch.tensor(bte_labels)
        )
        for m in metrics
    }

    # save metrics to file
    np.savetxt(
        f'{save_dir}/test_performance.csv',
        [list(bte_metrics.values())],
        delimiter=',',
        header=','.join(bte_metrics.keys()),
    )

    ### PLOTS
    plt.scatter(bte_labels, bte_outputs, marker='x', s=1, alpha=0.7, zorder=3)
    plt.plot(
        np.linspace(30, 95),
        np.linspace(30, 95),
        '--',
        c='k',
        alpha=0.3,
        zorder=1,
    )
    plt.title(
        f'mae: {bte_metrics["mae"]:.3f}, rmse: {bte_metrics["rmse"]:.3f}, \n'
        rf'$r^2$: {bte_metrics["r2"]:.3f}, pcc: {bte_metrics["pcc"]:.3f}'
    )
    plt.xlabel(r'true $T_m$ (°C)')
    plt.ylabel(r'predicted $T_m$ (°C)')
    plt.grid()
    plt.gca().set_aspect('equal')
    plt.savefig(f'{save_dir}-true_vs_pred.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
