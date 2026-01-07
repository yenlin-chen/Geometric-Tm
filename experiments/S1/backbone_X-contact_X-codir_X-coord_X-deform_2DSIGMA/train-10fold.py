import os, sys, time, json, random, torch
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
    worker_seed = torch.initial_seed() % 2 ** 32
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

    merge_edge_types = False
    # if merge_edge_types:
    #     edge_types_to_merge = ['backbone', 'contact', 'codir']

    edge_types_to_use = ['deform']
    edge_policy = '2DSIGMA'
    model_type = 'S1_M1'

    fold_file = '../../the fold.json'

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

    use_ogt = True
    use_pi = False

    # hyperparameters
    n_epochs = 30  # FLAG
    batch_size = 64
    learning_rate = 0.01

    ####################################################################
    # EXPERIMENT SETUP
    ####################################################################

    tv_set_name = 'train set (19,938 entries)'
    test_set_name = 'test set (2,044 entries)'

    loss_type = 'mse'
    metrics = {'pcc': pcc, 'rmse': rmse, 'mae': mae, 'mse': mse, 'r2': r2}

    with open(fold_file, 'r') as f:
        fold_acc = json.load(f)
    assert len(fold_acc['train']) == len(fold_acc['valid'])
    n_folds = len(fold_acc['train'])

    # machine-specific parameters
    num_workers = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {num_workers} workers for dataloading')
    print(f'Training on {device} device\n')

    exp_root_dir = os.path.normpath(self_dir + '/..' * 2)
    tv_meta_file = os.path.join(
        root_dir,
        'datasets',
        f'metadata - {tv_set_name}.csv'
    )
    test_meta_file = os.path.join(
        root_dir,
        'datasets',
        f'metadata - {test_set_name}.csv'
    )

    tv_set = Dataset(
        meta_file=tv_meta_file,
        sequence_embedding=embedding,
        thresholds=thresholds,
        merge_edge_types=merge_edge_types,
        version=dataset_version,
        time_limit=60,
        transform=None,
        device=device,
        entries_should_be_ready=entries_should_be_ready,
    )
    test_set = Dataset(
        meta_file=test_meta_file,
        sequence_embedding=embedding,
        thresholds=thresholds,
        merge_edge_types=merge_edge_types,
        version=dataset_version,
        time_limit=60,
        transform=None,
        device=device,
        entries_should_be_ready=entries_should_be_ready,
    )

    assert (
        len(tv_set)
        == np.loadtxt(tv_meta_file, dtype=np.str_, delimiter=',').shape[0]
    )
    assert (
        len(test_set)
        == np.loadtxt(test_meta_file, dtype=np.str_, delimiter=',').shape[0]
    )

    np.savetxt('tv_entries.txt', tv_set.processable_accessions, fmt='%s')
    np.savetxt('test_entries.txt', test_set.processable_accessions, fmt='%s')

    ####################################################################
    # INSTANTIATE TEST DATASET AND DATALOADERS
    ####################################################################

    test_loader = pyg.loader.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=rand_gen,
    )

    test_size = len(test_set)
    test_accessions_ordered = test_set.processable_accessions
    test_order = {test_accessions_ordered[i]: i for i in range(test_size)}

    test_Tm = [test_set.Tm_dict[a] for a in test_accessions_ordered]

    for fold_idx in range(n_folds):

        print(f' >> Fold {fold_idx} <<')

        fold_dir = f'fold{fold_idx}'
        os.makedirs(f'{fold_dir}', exist_ok=True)

        # file to save training history
        history_file = f'{fold_dir}/training_history.csv'
        test_history_file = f'{fold_dir}/training_history-test.csv'
        prediction_file_valid = f'{fold_dir}/predicted_values-valid_set.csv'
        prediction_file_train = f'{fold_dir}/predicted_values-train_set.csv'
        prediction_file_test = f'{fold_dir}/predicted_values-test_set.csv'
        best_performance_file = f'{fold_dir}/best_performance.csv'

        # skip fold if already done
        if os.path.exists(history_file):
            if os.path.getsize(history_file) > 0:
                hist = np.loadtxt(history_file, delimiter=',')
                if hist.shape == (n_epochs, 3 + 2 * len(metrics)):
                    print('  Skipping fold...\n')
                    continue

        ################################################################
        # INSTANTIATE MODEL, OPTIMIZER, AND LOSS
        ################################################################
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
        if fold_idx == 0:
            model.save_args('.')

        ################################################################
        # INSTANTIATE TRAIN & VALID DATASET AND DATALOADERS
        ################################################################

        ### SPLIT DATASET INTO TRAIN AND VALID ACCORDING TO RECORD
        train_acc = fold_acc['train'][str(fold_idx)]
        valid_acc = fold_acc['valid'][str(fold_idx)]
        # reverse engineer the indices
        for acc in train_acc:
            if acc not in tv_set.processable_accessions:
                print(acc)

        np.savetxt('train_acc.txt', train_acc, fmt='%s')
        np.savetxt(
            'tv_set.processable_accessions.txt',
            tv_set.processable_accessions,
            fmt='%s',
        )

        train_idx = torch.tensor(
            [
                np.where(tv_set.processable_accessions == a)[0][0]
                for a in train_acc
            ]
        )
        valid_idx = torch.tensor(
            [
                np.where(tv_set.processable_accessions == a)[0][0]
                for a in valid_acc
            ]
        )

        train_set = torch.utils.data.Subset(tv_set, train_idx)
        valid_set = torch.utils.data.Subset(tv_set, valid_idx)

        ### INSTANTIATE DATALOADERS
        train_loader = pyg.loader.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=rand_gen,
        )
        train_size = len(train_set)
        train_accessions_ordered = tv_set.processable_accessions[train_idx]
        train_order = {
            train_accessions_ordered[i]: i for i in range(train_size)
        }

        valid_loader = pyg.loader.DataLoader(
            valid_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=rand_gen,
        )
        valid_size = len(valid_set)
        valid_accessions_ordered = tv_set.processable_accessions[valid_idx]
        valid_order = {
            valid_accessions_ordered[i]: i for i in range(valid_size)
        }

        train_Tm = [tv_set.Tm_dict[a] for a in train_accessions_ordered]
        valid_Tm = [tv_set.Tm_dict[a] for a in valid_accessions_ordered]

        ### EXPORT ENTRY IDENTIFIERS FOR FUTURE REFERENCE
        np.savetxt(
            f'{fold_dir}/training_entries.txt',
            train_accessions_ordered,
            fmt='%s',
        )
        np.savetxt(
            f'{fold_dir}/validation_entries.txt',
            valid_accessions_ordered,
            fmt='%s',
        )

        ### DATASET STATISTICS
        train_min, train_max = np.amin(train_Tm), np.amax(train_Tm)
        train_mean, train_std = np.mean(train_Tm), np.std(train_Tm)
        train_median = np.median(train_Tm)

        valid_min, valid_max = np.amin(valid_Tm), np.amax(valid_Tm)
        valid_mean, valid_std = np.mean(valid_Tm), np.std(valid_Tm)
        valid_median = np.median(valid_Tm)

        ### SAVE TO FILE
        lines = [
            '# dataset,min,max,mean,std,median',
            f'train,{train_min},{train_max},{train_mean},{train_std},{train_median}',
            f'valid,{valid_min},{valid_max},{valid_mean},{valid_std},{valid_median}',
            # f'test {test_min} {test_max} {test_mean} {test_std} {test_median}',
        ]
        with open(f'{fold_dir}/dataset_statistics.csv', 'w+') as f:
            f.write('\n'.join(lines) + '\n')

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
            # CAUTION: TO PREVENT DATA LEAKAGE, DO NOT USE VALIDATION SET
            min_max=(np.amin(train_Tm), np.amax(train_Tm)),
            mean_std=(np.mean(train_Tm), np.std(train_Tm)),
        )

        ### FILE TO KEEP TRACK OF TRAINING PERFORMANCE
        # train and validation history
        header = '# epoch,train_loss,valid_loss'
        for m in metrics.keys():
            header += f',train_{m},valid_{m}'
        with open(history_file, 'w+') as f:
            f.write(header + '\n')
        # epoch where performance improves
        with open(best_performance_file, 'w+') as f:
            f.write(header + '\n')
        # test history :)
        header = '# epoch,test_loss'
        for m in metrics.keys():
            header += f',test_{m}'
        with open(test_history_file, 'w+') as f:
            f.write(header + '\n')

        # prediction on validation set
        header = '# epoch,' + ','.join(valid_accessions_ordered)
        line_1 = '# true_labels,' + ','.join([f'{Tm:.8f}' for Tm in valid_Tm])
        with open(prediction_file_valid, 'w+') as f:
            f.write(header + '\n')
            f.write(line_1 + '\n')
        # prediction on training set
        header = '# epoch,' + ','.join(train_accessions_ordered)
        line_1 = '# true_labels,' + ','.join([f'{Tm:.8f}' for Tm in train_Tm])
        with open(prediction_file_train, 'w+') as f:
            f.write(header + '\n')
            f.write(line_1 + '\n')
        # prediction on test set
        header = '# epoch,' + ','.join(test_accessions_ordered)
        line_1 = '# true_labels,' + ','.join([f'{Tm:.8f}' for Tm in test_Tm])
        with open(prediction_file_test, 'w+') as f:
            f.write(header + '\n')
            f.write(line_1 + '\n')

        ### TRAIN FOR N_EPOCHS
        best_v_loss = 1e8
        for i in range(n_epochs):
            epoch = i + 1
            print(f'Epoch {epoch}')

            # get learning rate of this epoch (NOTE: is this working properly?)
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']

            # time it
            start = time.time()

            ### ONE PASS OVER TRAINING SET
            t_loss, t_outputs, t_labels, t_accessions = trainer.train_one_epoch(
                train_loader
            )
            print(f'    train loss: {t_loss:.8f}')

            # compute various metrics
            t_metrics = [t_loss] + [
                m(t_outputs, t_labels) for m in metrics.values()
            ]

            ### ONE PASS OVER VALID SET
            v_loss, v_outputs, v_labels, v_accessions = trainer.evaluate(
                valid_loader
            )
            print(f'    valid loss: {v_loss:.8f}')

            # compute various metrics
            v_metrics = [v_loss] + [
                m(v_outputs, v_labels) for m in metrics.values()
            ]

            ### ONE PASS OVER TEST SET
            te_loss, te_outputs, te_labels, te_accessions = trainer.evaluate(
                test_loader
            )
            print(f'    test  loss: {te_loss:.8f}')

            # compute various metrics
            te_metrics = [te_loss] + [
                m(te_outputs, te_labels) for m in metrics.values()
            ]

            ### SAVE MODEL PERFORMANCE
            # train and valid
            line = f'{epoch}'
            for i in range(len(metrics) + 1):  # metrics + loss
                line += f',{t_metrics[i]:.8f},{v_metrics[i]:.8f}'
            with open(history_file, 'a+') as f:
                f.write(line + '\n')
            if v_loss < best_v_loss:
                best_v_loss = v_loss
                best_epoch = epoch
                with open(best_performance_file, 'a+') as f:
                    f.write(line + '\n')
                torch.save(model.state_dict(), f'{fold_dir}/model-best.pt')
            # test
            line = f'{epoch}'
            for i in range(len(metrics) + 1):  # metrics + loss
                line += f',{te_metrics[i]:.8f}'
            with open(test_history_file, 'a+') as f:
                f.write(line + '\n')

            ### SAVE PREDICTION FOR TRAINING SET
            # order outputted values by acccession
            idx_order = np.argsort(
                [train_order[a] for a in t_accessions.tolist()]
            )
            t_outputs_ordered = t_outputs.detach().cpu().numpy()[idx_order]
            line = f'{epoch},' + ','.join(t_outputs_ordered.astype(np.str_))
            with open(prediction_file_train, 'a+') as f:
                f.write(line + '\n')

            ### SAVE PREDICTION FOR VALIDATION SET
            # order outputted values by acccession
            idx_order = np.argsort(
                [valid_order[a] for a in v_accessions.tolist()]
            )
            v_outputs_ordered = v_outputs.detach().cpu().numpy()[idx_order]
            line = f'{epoch},' + ','.join(v_outputs_ordered.astype(np.str_))
            with open(prediction_file_valid, 'a+') as f:
                f.write(line + '\n')

            ### SAVE PREDICTION FOR TEST SET
            # order outputted values by acccession
            idx_order = np.argsort(
                [test_order[a] for a in te_accessions.tolist()]
            )
            te_outputs_ordered = te_outputs.detach().cpu().numpy()[idx_order]
            line = f'{epoch},' + ','.join(te_outputs_ordered.astype(np.str_))
            with open(prediction_file_test, 'a+') as f:
                f.write(line + '\n')

            # time it
            print(f' >> Time Elapsed: {time.time()-start:.4f}s\n')

        ################################################################
        # plot learning curve
        ################################################################
        history = np.loadtxt(history_file, delimiter=',')
        test_history = np.loadtxt(test_history_file, delimiter=',')

        # loss
        plt.plot(history[:, 0], history[:, 1], label='training')
        plt.plot(history[:, 0], history[:, 2], label='validation')
        plt.plot(test_history[:, 0], test_history[:, 1], label='test')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel(f'loss ({loss_type})')
        plt.ylim(0, 1)
        plt.savefig(
            f'{fold_dir}/learning_curve-loss.png', dpi=300, bbox_inches='tight'
        )
        plt.close()


if __name__ == '__main__':
    main()
