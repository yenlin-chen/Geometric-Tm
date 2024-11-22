def check_args(**kwargs):

    # populate namespace with the key in kwargs
    for key in kwargs:
        print(key, kwargs[key])
        if isinstance(kwargs[key], str):
            exec(f'{key} = "{kwargs[key]}"')
        else:
            exec(f'{key} = {kwargs[key]}')

    ### IN THE CASE OF NO GRAPH CONVOLUTION
    if kwargs['graph_dims'] == []:

        # arguments that should NOT be specified
        kw_list = [
            'node_feat_name',
            'node_feat_size',
            'gnn_type',
            'gat_atten_heads',
            'dim_node_hidden_ls',
            'n_conv_layers',
            'dim_shape',
            'dim_node_hidden',
            'conv_norm',
            'norm_graph_input',
            'norm_graph_output',
            'graph_global_pool',
            'graph_dropout_rate',
            'dropfeat_rate',
            'dropedge_rate',
            'dropnode_rate',
            'jk_mode',
        ]
        if any([kwargs[kw] is not None for kw in kw_list]):
            raise ValueError(
                'None of the following arguments should be specified '
                'if `graph_dims` is empty: '
                f'{kw_list}'
            )

        # another data pipeline must be turned on
        if not kwargs['use_ogt'] and not kwargs['feat2ffc']:
            raise ValueError(
                'At least one of the following must be specified if '
                '`graph_dims` is empty: `use_ogt` or `feat2ffc`'
            )

        # must NOT specify whether the graph convoluted outputs
        # should be embedded by fully connected layers
        if kwargs['embed_graph_outputs'] is not None:
            raise ValueError(
                '`embed_graph_outputs` must not be specified if '
                '`graph_dims` is empty'
            )

    ### IF GRAPH PIPELINES ARE ON
    else:

        # arguments regarding convolutions that must be specified
        kw_list = [
            'node_feat_name',
            'node_feat_size',
            'gnn_type',
            'conv_norm',
            'norm_graph_input',
            'norm_graph_output',
            'graph_global_pool',
            'graph_dropout_rate',
            'dropfeat_rate',
            'dropedge_rate',
            'dropnode_rate',
        ]
        if None in [kwargs[kw] for kw in kw_list]:
            raise ValueError(
                'All of the following arguments must be specified if '
                '`graph_dims` is not empty: '
                f'{kw_list}'
            )

        # must specify whether the graph convoluted outputs should
        # be embedded by fully connected layers
        if kwargs['embed_graph_outputs'] is None:
            raise ValueError(
                '`embed_graph_outputs` must be specified if '
                '`graph_dims` is not empty'
            )

        # arguments that induces constraints on other arguments
        if kwargs['gnn_type'] == 'gat' and not kwargs['gat_atten_heads']:
            raise ValueError(
                '`gat_atten_heads` must be specified if ' '`gnn_type` is gat'
            )
        if kwargs['n_conv_layers'] == 1:
            kw_list = ['dim_shape', 'jk_mode']
            if any([kwargs[kw] is not None for kw in kw_list]):
                raise ValueError(
                    'None of the following arguments should be specified '
                    'if `n_conv_layers` is 1: '
                    f'{kw_list}'
                )
        if not kwargs['conv_norm']:
            kw_list = ['norm_graph_input', 'norm_graph_output']
            if any([kwargs[kw] is not None for kw in kw_list]):
                raise ValueError(
                    'None of the following arguments should be specified '
                    'if `conv_norm` is False: '
                    f'{kw_list}'
                )

        # mutually exclusive arguments
        kw_list = ['n_conv_layers', 'dim_node_hidden']
        if kwargs['dim_node_hidden_ls'] is None:
            if not all([kwargs[kw] for kw in kw_list]):
                raise ValueError(
                    'All the following arguments must be specified if '
                    '`dim_node_hidden_ls` is None: '
                    f'{kw_list}'
                )
        else:
            if any([kwargs[kw] is not None for kw in kw_list]):
                raise ValueError(
                    'None of the following arguments should be specified '
                    'if `dim_node_hidden_ls` is given: '
                    f'{kw_list}'
                )

    ### RULES FOR GRAPH EMBEDDING SETUP
    if not kwargs['embed_graph_outputs']:
        # arguments that should NOT be specified
        kw_list = [
            'graph_embedding_hidden_ls',
            'graph_embedding_dim',
            'n_graph_embedding_layers',
            'graph_embedding_dropout_rate',
        ]
        if any([kwargs[kw] is not None for kw in kw_list]):
            raise ValueError(
                'None of the following arguments should be specified if '
                '`embed_graph_outputs` is False: '
                f'{kw_list}'
            )
    else:

        # arguments that must be specified
        if kwargs['graph_embedding_dropout_rate'] is None:
            raise ValueError(
                '`graph_embedding_dropout_rate` must be specified if '
                '`embed_graph_outputs` is True'
            )

        # mutually exclusive arguments
        kw_list = ['graph_embedding_dim', 'n_graph_embedding_layers']
        if kwargs['graph_embedding_hidden_ls'] is None:
            if not all([kwargs[kw] for kw in kw_list]):
                raise ValueError(
                    'All of the following arguments must be specified if '
                    '`graph_embedding_hidden_ls` is None: '
                    f'{kw_list}'
                )
        else:
            if any([kwargs[kw] is not None for kw in kw_list]):
                raise ValueError(
                    'None of the following arguments should be specified '
                    'if `graph_embedding_hidden_ls` is given: '
                    f'{kw_list}'
                )

    ### RULES FOR pLDDT SETUP
    if kwargs['use_node_pLDDT'] or kwargs['pLDDT2ffc']:
        # must specify whether the pLDDT should be embedded
        if kwargs['embed_pLDDT'] is None:
            raise ValueError(
                '`embed_pLDDT` must be specified if '
                '`use_node_pLDDT` or `pLDDT2ffc` is True'
            )
    else:
        # must NOT specify whether the pLDDT should be embedded
        if kwargs['embed_pLDDT'] is not None:
            raise ValueError(
                '`embed_pLDDT` must not be specified if '
                '`use_node_pLDDT` or `pLDDT2ffc` is False'
            )

    if not kwargs['embed_pLDDT']:
        if kwargs['pLDDT_dropout_rate'] is not None:
            raise ValueError(
                '`pLDDT_dropout_rate` must not be specified if '
                '`embed_pLDDT` is None or False'
            )
    else:
        if kwargs['pLDDT_dropout_rate'] is None:
            raise ValueError(
                '`pLDDT_dropout_rate` must be specified if '
                '`embed_pLDDT` is True'
            )

    ### RULES FOR BFACTOR SETUP
    if kwargs['use_node_bfactor'] or kwargs['bfactor2ffc']:
        # must specify whether the b-factor should be embedded
        if kwargs['embed_bfactors'] is None:
            raise ValueError(
                '`embed_bfactors` must be specified if '
                '`use_node_bfactor` or `bfactor2ffc` is True'
            )
    else:
        # must NOT specify whether the b-factor should be embedded
        if kwargs['embed_bfactors'] is not None:
            raise ValueError(
                '`embed_bfactors` must not be specified if '
                '`use_node_bfactor` or `bfactor2ffc` is False'
            )

    if not kwargs['embed_bfactors']:
        if kwargs['bfactor_dropout_rate'] is not None:
            raise ValueError(
                '`bfactor_dropout_rate` must not be specified if '
                '`embed_bfactors` is None or False'
            )
    else:
        if kwargs['bfactor_dropout_rate'] is None:
            raise ValueError(
                '`bfactor_dropout_rate` must be specified if '
                '`embed_bfactors` is True'
            )

    ### RULES FOR OGT EMBEDDING
    if kwargs['use_ogt']:
        # must specify whether OGT should be embedded
        if kwargs['embed_ogt'] is None:
            raise ValueError(
                '`embed_ogt` must be specified if ' '`use_ogt` is True'
            )
    else:
        # must NOT specify whether OGT should be embedded
        if kwargs['embed_ogt'] is not None:
            raise ValueError(
                '`embed_ogt` must not be specified if ' '`use_ogt` is False'
            )

    if not kwargs['embed_ogt']:
        if kwargs['ogt_dropout_rate'] is not None:
            raise ValueError(
                '`ogt_dropout_rate` must not be specified if '
                '`embed_ogt` is None or False'
            )
    else:
        if kwargs['ogt_dropout_rate'] is None:
            raise ValueError(
                '`ogt_dropout_rate` must be specified if ' '`embed_ogt` is True'
            )

    ### RULES FOR FEAT2FFC
    kw_list = [
        'feat2ffc_feat_name',
        'feat2ffc_feat_size',
        'feat2ffc_global_pool',
    ]
    if kwargs['feat2ffc']:
        if None in [kwargs[kw] for kw in kw_list]:
            raise ValueError(
                'All of the following arguments must be specified if '
                '`feat2ffc` is True: '
                f'{kw_list}'
            )

        # must specify whether feat2ffc should be embedded
        if kwargs['embed_feat2ffc'] is None:
            raise ValueError(
                '`embed_feat2ffc` must be specified if `feat2ffc` is True'
            )
    else:
        if any([kwargs[kw] is not None for kw in kw_list]):
            raise ValueError(
                'None of the following arguments should be specified if '
                '`feat2ffc` is False: '
                f'{kw_list}'
            )

        # must NOT specify whether feat2ffc should be embedded
        if kwargs['embed_feat2ffc'] is not None:
            raise ValueError(
                '`embed_feat2ffc` must not be specified if '
                '`feat2ffc` is False'
            )

    ### RULES FOR FEAT2FFC EMBEDDING
    if not kwargs['embed_feat2ffc']:
        kw_list = [
            'feat2ffc_embedding_dim',
            'feat2ffc_embedding_hidden_ls',
            'n_feat2ffc_embedding_layers',
            'feat2ffc_dropout_rate',
        ]
        if any([kwargs[kw] is not None for kw in kw_list]):
            raise ValueError(
                'None of the following arguments should be specified if '
                '`embed_feat2ffc` is None or False: '
                f'{kw_list}'
            )
    else:
        if kwargs['feat2ffc_dropout_rate'] is None:
            raise ValueError(
                '`feat2ffc_dropout_rate` must be specified if '
                '`embed_feat2ffc` is True'
            )

        # mutually exclusive arguments
        kw_list = ['feat2ffc_embedding_dim', 'n_feat2ffc_embedding_layers']
        if kwargs['feat2ffc_embedding_hidden_ls'] is None:
            if not all([kwargs[kw] for kw in kw_list]):
                raise ValueError(
                    'All of the following arguments must be specified if '
                    '`feat2ffc_embedding_hidden_ls` is None: '
                    f'{kw_list}'
                )
        else:
            if any([kwargs[kw] is not None for kw in kw_list]):
                raise ValueError(
                    'None of the following arguments should be specified if '
                    '`feat2ffc_embedding_hidden_ls` is not None: '
                    f'{kw_list}'
                )

    ### RULES FOR FINAL FC

    # arguments that must be specified
    kw_list = ['fc_norm', 'fc_dropout_rate']
    if None in [kwargs[kw] for kw in kw_list]:
        raise ValueError(
            'All of the following arguments must be specified: ' f'{kw_list}'
        )

    # mutually exclusive arguments
    if kwargs['fc_hidden_ls'] is None:
        if not kwargs['n_fc_hidden_layers']:
            raise ValueError(
                'All of the following arguments must be specified if '
                '`fc_hidden_ls` is None: '
                '`n_fc_hidden_layers`'
            )
    else:
        if kwargs['n_fc_hidden_layers']:
            raise ValueError(
                'None of the following arguments should be specified if '
                '`fc_hidden_ls` is not None: '
                '`n_fc_hidden_layers`'
            )

    # arguments that induces constraints on other arguments
    if not kwargs['fc_norm']:
        if kwargs['norm_fc_input'] is not None:
            raise ValueError(
                '`norm_fc_input` must not be specified if ' '`fc_norm` is False'
            )
