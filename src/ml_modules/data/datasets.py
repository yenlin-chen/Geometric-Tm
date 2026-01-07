if __name__ == '__main__':
    from __init__ import (
        external_dir,
        collation_dir,
        processed_dir,
        res_to_1hot,
    )
    from retrievers import AlphaFold_Retriever
    from enm import TNM_Computer
    from encoders import ProtTrans_Encoder
else:
    from . import (
        external_dir,
        collation_dir,
        processed_dir,
        res_to_1hot,
    )
    from .retrievers import AlphaFold_Retriever
    from .enm import TNM_Computer
    from .encoders import ProtTrans_Encoder

import os, torch, prody, json, warnings, time
import numpy as np
import torch_geometric as pyg

from tqdm import tqdm

df_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

prody.confProDy(verbosity='none')
tnm_setup_filename = '{}{}_CA{:.1f}_ALL_PHIPSIPSI'

coupling_types = ['coord', 'codir', 'deform']


class Dataset(pyg.data.Dataset):
    '''
    To be completed.
    '''

    def __init__(
        self,
        meta_file,
        version,
        sequence_embedding,
        thresholds,
        merge_edge_types=False,
        include_edge_weights=False,
        time_limit=60,
        transform=None,
        device=df_device,
        entries_should_be_ready=False,
        rebuild=False,
        float16_embeddings=False,
    ):

        self.device = device
        self.version = version
        self.sequence_embedding = sequence_embedding.lower()
        self.thresholds = thresholds
        self.time_limit = time_limit
        self.include_edge_weights = include_edge_weights

        self.entries_should_be_ready = entries_should_be_ready
        self.rebuild = rebuild
        self.float16_embeddings = float16_embeddings

        self.threshold_name = (
            f'contact_{thresholds["contact"]}-'
            f'codir_{thresholds["codir"]}-'
            f'coord_{thresholds["coord"]}-'
            f'deform_{thresholds["deform"]}'
        )
        self.merge_edge_types = merge_edge_types

        ### COMMON DIRECTORIES
        self.v_process_dir = os.path.join(processed_dir, self.version)
        if not merge_edge_types:
            self.graph_dir = os.path.join(  # for graphs
                self.v_process_dir, 'graphs', self.threshold_name
            )
        else:
            self.graph_dir = os.path.join(  # for graphs
                self.v_process_dir, 'graphs-merged', self.threshold_name
            )
        self.embedding_dir = os.path.join(  # for sequence embeddings
            self.v_process_dir, 'sequence_embeddings', self.sequence_embedding
        )

        if not self.entries_should_be_ready:
            os.makedirs(
                self.graph_dir, exist_ok=not self.entries_should_be_ready
            )
            os.makedirs(
                self.embedding_dir, exist_ok=not self.entries_should_be_ready
            )

        for ct, v in thresholds.items():
            if isinstance(v, str):
                self.thresholds[ct] = v.upper()
        print(f' -> Thresholds: {self.thresholds}')

        ### PROCESS META DATA
        self.meta_file = meta_file
        print(f' -> Generating dataset from {self.meta_file}')
        try:
            self.meta = np.loadtxt(
                self.meta_file, dtype=np.str_, delimiter=','  # , skiprows=1
            )
        except FileNotFoundError as err:
            raise FileNotFoundError(
                f'{self.meta_file} does not exist. '
            ) from err
        if self.meta.ndim == 1:
            self.meta = self.meta.reshape((1, -1))
        print(f' -> Number of entries in meta file    : {len(self.meta)}')

        # CAUTION: duplicates will be overwritten
        self.all_Tm_dict = {e[0]: float(e[1]) for e in self.meta}
        self.all_accessions = list(self.all_Tm_dict.keys())
        all_Tm = list(self.all_Tm_dict.values())
        self.all_ogt_dict = {e[0]: float(e[2]) for e in self.meta}
        self.all_species_dict = {e[0]: str(e[3]) for e in self.meta}

        ### STATISTICS BEFORE PROCESSING (FOR REFERENCE ONLY)
        all_Tm_mean = np.mean(all_Tm)
        all_Tm_max = np.amax(all_Tm)
        all_Tm_min = np.amin(all_Tm)
        print(f'     >> mean value of Tm : {all_Tm_mean:.4f}')
        print(f'     >> range of Tm      : {all_Tm_min:.4f}-{all_Tm_max:.4f}')

        print(
            ' -> Number of unique accessions       :', len(self.all_accessions)
        )

        if len(self.meta) != len(self.all_accessions):
            raise ValueError('Duplicate accessions found in meta file.')

        ### INSTANTIATE UTILITIES
        self.struct_retriever = AlphaFold_Retriever(version=6)
        self.tnm_computer = TNM_Computer()

        ### CONSTRUCTOR OF PARENT CLASS (PREPROCESSING)
        super().__init__(self.raw_dir, transform, None, None)

        ### STATISTICS AFTER PROCESSING
        self.processable_accessions = self.get_processable_accessions()
        print(
            ' -> Final number of accessions        :',
            len(self.processable_accessions),
        )

        self.Tm_dict = {
            a: self.all_Tm_dict[self.struct_retriever.unmodify_accession(a)]
            for a in self.processable_accessions
        }
        all_Tm = list(self.Tm_dict.values())

        self.Tm_mean = np.mean(all_Tm)
        self.Tm_max = np.amax(all_Tm)
        self.Tm_min = np.amin(all_Tm)
        print(f'     >> mean value of Tm : {self.Tm_mean:.4f}')
        print(f'     >> range of Tm      : {self.Tm_min:.4f}-{self.Tm_max:.4f}')
        print(' -> Number of unique accessions       :', len(self.Tm_dict))

        print('Dataset instantiation complete.')

    @property
    def raw_dir(self):
        return os.path.join(external_dir, 'AlphaFoldDB', 'pdb')

    @property
    def downloadable_accessions(self):
        '''Accessions for which AlphaFold structures should be found.

        Reads list of accessions whose AlphaFold structures are
        unavailable and removes them from the dataset. Returned values
        are UniProt accessions. For project-internal accessions, call
        `AlphaFold_Retriever.modify_accession()` on the outputted list.
        '''

        failed_accessions = self.struct_retriever.get_failed_accessions()
        return np.setdiff1d(self.all_accessions, failed_accessions)

    @property
    def raw_file_names(self):
        # project-internal accessions are used as file names
        # "<UniProt accession>-AFv<version number>"

        modified_accessions = self.struct_retriever.modify_accession(
            self.downloadable_accessions
        )
        return [f'{a}.pdb' for a in modified_accessions]

    @property
    def processed_dir(self):
        return self.v_process_dir

    def get_processable_accessions(self):
        '''Accessions for which TNM processing should be successful.

        Reads list of accessions whose TNM results are unobtainable and
        removes them from the dataset.
        '''

        if self.entries_should_be_ready:
            return np.array(
                self.struct_retriever.modify_accession(self.all_accessions)
            )

        modified_downloadable = self.struct_retriever.modify_accession(
            self.downloadable_accessions
        )

        failed_accessions = self.tnm_computer.get_failed_accessions()

        return np.setdiff1d(modified_downloadable, failed_accessions)

    @property
    def processed_file_names(self):

        files = []
        for acc_name in self.get_processable_accessions():
            files.append(os.path.join(self.graph_dir, f'{acc_name}.pt'))
            files.append(os.path.join(self.embedding_dir, f'{acc_name}.pt'))

        return files

    def download(self):
        if self.entries_should_be_ready:
            print('Entries should be ready. Skipping download.')
            return

        print('\nDownloading AlphaFold structures...')  # and PAEs...')
        successful_accessions, _ = self.struct_retriever.batch_retrieve(
            self.all_accessions  #, item_list=['pdb']  # , 'pae']
        )
        print(
            ' -> Accessions successfully downloaded:',
            len(successful_accessions),
        )

    def process(self):
        # pyg prints "Processing..." internally
        # no need for print statement
        # print('Processing AlphaFold structures...')

        if self.entries_should_be_ready:
            raise Exception(
                'Entering process(), aborting program since '
                '"entries_should_be_ready" is set to True.'
            )

        ################################################################
        # MODAL ANALYSIS BY TNM
        ################################################################

        modified_downloadable = self.struct_retriever.modify_accession(
            self.downloadable_accessions
        )
        pdb_path_list = [
            self.struct_retriever.path_to_file(a)  #, item_type='pdb')
            for a in self.downloadable_accessions
        ]
        start_time = time.time()
        successful_accessions, _ = self.tnm_computer.batch_run(
            modified_downloadable,
            pdb_path_list,
            timeout=self.time_limit,  # increase if necessary
        )
        print(f' -> TNM processing time: {time.time() - start_time:.2f} seconds')

        print(
            ' -> Accessions with TNM results       :',
            len(successful_accessions),
        )

        ################################################################
        # GATHER DATASET-WIDE STATISTICS (IF REQUIRED)
        ################################################################

        stats_dir = os.path.join( # FLAG
            os.path.dirname(__file__),
            'stats - preliminary training set (896 entries)',
        )

        dataset_wide_thresholds = {}
        for ct in coupling_types:
            threshold_file = os.path.join(stats_dir, f'thresholds - {ct}.json')
            if not os.path.exists(threshold_file):
                raise FileNotFoundError(
                    f'No dataset-wide threshold for {ct}, '
                    f'run summarize_couplings.py to generate statistics'
                )
            with open(threshold_file) as f:
                dataset_wide_thresholds[ct] = json.load(f)

        ################################################################
        # BUILD PYG GRAPHS
        ################################################################

        ### INSTANTIATE ENCODERS (TO BE DELETED AFTER USE)
        if self.sequence_embedding == 'prottrans':
            encoder = ProtTrans_Encoder(
                float16_embeddings=self.float16_embeddings
            )
        else:
            raise ValueError(f'Invalid encoder {self.sequence_embedding}')

        start_time = time.time()
        pbar = tqdm(
            self.get_processable_accessions(), dynamic_ncols=True, ascii=True
        )
        for accession in pbar:

            pbar.set_description(f'Graphs {accession:<12s}')

            # chain to use
            chains_for_tnm = 'A'
            chains_to_encode = 'A'

            path_to_tnm_files = self.tnm_computer.path_to_outputs(
                accession, chains=chains_for_tnm
            )
            path_to_tnm_merged = self.tnm_computer.path_to_merged(accession)
            path_to_pdb = self.struct_retriever.path_to_file(accession)

            unmodified_accession = self.struct_retriever.unmodify_accession(
                accession
            )
            atoms_all = prody.parsePDB(path_to_pdb, subset='ca')
            atoms = atoms_all.select(
                f'protein and chain {chains_to_encode}'
            )

            # tnm cannot process HETATM residues correctly, so we discard them
            hetatm_flags = atoms.getFlags('hetatm')
            if np.any(hetatm_flags):
                tqdm.write(
                    f' -> {accession} contains HETATM residues, discarding.'
                )
                continue

            ### CREATE HETERODATA OBJECT
            data = pyg.data.HeteroData()
            data.accession = accession
            data.ogt = self.all_ogt_dict[unmodified_accession]
            data.species = self.all_species_dict[unmodified_accession]

            ### NODE EMBEDDINGS (SEQUENTIAL INFORMATION)
            resnames, identifiers = self.tnm_computer.get_resnames(
                accession, return_ids=True
            )
            loc_chain = np.where(
                identifiers['auth_asym_id'] == chains_to_encode
            )[0]
            dc_idx_to_include = identifiers['dc_idx'][loc_chain]
            auth_seq_ids = identifiers['auth_seq_ids'][loc_chain]
            resnames = resnames[loc_chain]
            sequence = ''.join(resnames)
            n_residues = resnames.size
            data.num_nodes = n_residues

            assert n_residues == atoms.getResnums().size
            try:
                auth_seq_ids = auth_seq_ids.astype(np.int_)
            except ValueError:
                tqdm.write(
                    f' -> {accession} contains non-integer resids, discarding.'
                )
                continue

            # convert resIDs to one-hot-encoding
            resnames_1hot = np.zeros((n_residues, 20), dtype=np.int_)
            for j, resname in enumerate(resnames):
                resnames_1hot[j, res_to_1hot[resname]] = 1
            data['residue'].res1hot = torch.from_numpy(resnames_1hot)
            data['residue'].x = torch.from_numpy(resnames_1hot)

            # ### ADD pLDDT AS NODE ATTRIBUTES
            # # AlphaFold populates the B-factor column with pLDDT scores
            # # which we will include as node attributes
            # pLDDT = atoms.getBetas() # range: [0,100]
            # data['residue'].pLDDT = torch.from_numpy(pLDDT)

            ### BUILD CONTACT GRAPH WITH PRODY (ANM)
            # contact maps are built based on the position of Ca atoms
            anm = prody.ANM(name=f'{accession}_CA')
            anm.buildHessian(
                atoms,
                cutoff=(
                    float(self.thresholds['contact'])
                    if self.thresholds['contact'] != 'X' else 12
                ),
            )
            cont = -anm.getKirchhoff().astype(np.int_)  # the Laplacian matrix
            # np.fill_diagonal(cont, 1) # contact map completed here (with loops)
            edge_index = np.argwhere(cont == 1).T  # undirected graph
            n_cont_edges = int(edge_index.shape[1] / 2)
            if self.thresholds['contact'] != 'X':
                data['residue', 'contact', 'residue'].edge_index = torch.from_numpy(
                    edge_index
                )
                threshold_values = {
                    'contact': float(self.thresholds['contact'])
                }
            else:
                threshold_values = {'contact': 'n/a'}

            edge_index = []
            for dc_idx in range(n_residues-1):
                if auth_seq_ids[dc_idx+1] - auth_seq_ids[dc_idx] == 1:
                    edge_index.append([dc_idx, dc_idx+1])
            edge_index = np.array(edge_index).T
            edge_index = np.hstack((edge_index, np.flip(edge_index, axis=0)))
            data[
                'residue', 'backbone', 'residue'
            ].edge_index = torch.from_numpy(edge_index)

            ### ADD DYNAMICAL COUPLING EDGES
            # 1. self-loops are not included
            # 2. indices are 0-based
            # (TNM implements two different indexing systems)
            # read merged TNM output
            with open(path_to_tnm_merged, 'r') as f:
                merged_data = json.load(f)

            for ct in coupling_types:

                keyname = os.path.basename(path_to_tnm_files[ct])

                # read TNM output
                raw_data = np.loadtxt(
                    merged_data[keyname].split('\n'), dtype=np.str_
                ).reshape((-1, 3))
                edge_index = raw_data[:, :2].astype(np.int_).T
                couplings = raw_data[:, 2].astype(np.float_)
                # filter out residues on chain A
                indices = np.logical_and.reduce(
                    np.isin(edge_index, dc_idx_to_include),
                    axis=0
                )
                edge_index = edge_index[:, indices]
                couplings = couplings[indices]

                # COMPUTE THRESHOLDS
                if self.thresholds[ct] == 'NONE':
                    threshold = np.amin(couplings)

                # dataset-wide thresholds
                elif self.thresholds[ct].endswith('DCONT'):
                    threshold = dataset_wide_thresholds[ct][self.thresholds[ct]]

                elif self.thresholds[ct].endswith('DN'):
                    threshold = dataset_wide_thresholds[ct][self.thresholds[ct]]

                elif self.thresholds[ct].endswith('DSIGMA'):
                    threshold = dataset_wide_thresholds[ct][self.thresholds[ct]]

                # entry-specific thresholds
                elif self.thresholds[ct].endswith('SIGMA'):
                    n_sigmas = float(self.thresholds[ct][:-5])
                    mean = np.mean(couplings)
                    sigma = np.std(couplings)
                    threshold = mean + sigma * n_sigmas

                elif self.thresholds[ct].endswith('N'):
                    n_N = float(self.thresholds[ct][:-1])
                    max_coupling = np.amax(couplings)
                    range_coupling = max_coupling - np.amin(couplings)
                    threshold = max_coupling - range_coupling * n_N / 100

                elif self.thresholds[ct].endswith('CONT'):
                    n_cont = float(self.thresholds[ct][:-4])
                    n_edges = int(n_cont * n_cont_edges)
                    threshold = np.sort(couplings)[-n_edges]

                elif self.thresholds[ct].endswith('PAIR'):
                    n_pair = float(self.thresholds[ct][:-4])
                    n_edges = int(
                        (n_residues * (n_residues - 1) / 2) * n_pair / 100
                    )
                    threshold = np.sort(couplings)[-n_edges]

                elif self.thresholds[ct].endswith('X'):
                    threshold_values[ct] = 'n/a'
                    continue

                # elif self.thresholds[ct] == 'DEF':
                #     raise NotImplementedError('DEF not implemented yet')
                #     threshold = 'tnm default'

                # elif self.thresholds[ct] == 'MEAN':
                #     threshold = np.mean(couplings)

                # elif self.thresholds[ct] == 'MEDIAN':
                #     threshold = np.median(couplings)

                # elif self.thresholds[ct].endswith('Q'):
                #     q = float(self.thresholds[ct][:-1])
                #     threshold = np.percentile(couplings, q/4*100)

                else:
                    try:
                        threshold = float(self.thresholds[ct])
                    except:
                        raise ValueError(
                            f'Invalid threshold for {ct}: {self.thresholds[ct]}'
                        )
                threshold_values[ct] = threshold

                # assign edge indices to HeteroData object
                if self.thresholds[ct] != 'NONE':
                    edge_index = edge_index[:, couplings > threshold]

                edge_index = np.hstack(
                    (edge_index, np.flip(edge_index, axis=0))
                )
                data['residue', ct, 'residue'].edge_index = torch.from_numpy(
                    edge_index
                )

                if self.include_edge_weights or self.thresholds[ct] == 'NONE':
                    # keep record of edge weights
                    data[
                        'residue', ct, 'residue'
                    ].edge_weight = torch.from_numpy(couplings)

            ### MERGE EDGES (IF SPECIFIED)
            if self.merge_edge_types:
                edge_types_to_merge = [
                    k for k, v in self.thresholds.items() if v != 'X'
                ]
                data['residue', 'merged', 'residue'].edge_index = torch.unique(
                    torch.cat(
                        [
                            data['residue', et, 'residue'].edge_index
                            for et in edge_types_to_merge
                        ],
                        dim=1,
                    ),
                    sorted=False,
                    dim=1,
                )

                diff_set = torch.tensor(list(
                    set(
                        (edge[0].item(), edge[1].item()) for edge in data['residue', 'backbone', 'residue'].edge_index.T
                    ) - set(
                        (edge[0].item(), edge[1].item()) for edge in data['residue', 'merged', 'residue'].edge_index.T
                    )
                ))

                data[
                    'residue', 'backbone-complementary', 'residue'
                ].edge_index = (
                    diff_set.T
                    if diff_set.numel()
                    else torch.empty(2,0, dtype=torch.int64)
                )

                # remove merged edge types
                for et in edge_types_to_merge:
                    del data['residue', et, 'residue']

            # ### ADD PREDICTED B-FACTORS AS NODE ATTRIBUTES
            # # assign B-factors as predicted by TNM to graph nodes
            # raw_data = np.loadtxt(path_to_tnm_files['bfactor'], dtype=np.str_)[:,1]
            # bfactor = raw_data.astype(np.float_)
            # data['residue'].bfactor = torch.from_numpy(bfactor)

            # ### ADD PREDICTED ALIGNED ERROR AS EDGES
            # # a cutoff is applied to the PAE matrix to determine
            # # which edges to keep (done for computational efficiency)

            # # PAE is directed, i.e. the PAE matrix is not symmetric.
            # # However the upper and lower triangle are very similar
            # # so we will make all edges undirected
            # pae_path = self.struct_retriever.path_to_file(
            #     accession, item_type='pae'
            # )
            # with open(pae_path, 'r') as f:
            #     pae_dict = json.load(f)[0]
            # pae = np.array(pae_dict['predicted_aligned_error'])

            # edge_index = np.argwhere(pae<=4).T # directed edges
            # edge_index = np.hstack( # will include duplicates
            #     (edge_index, np.flip(edge_index, axis=0))
            # )
            # edge_index = np.unique(edge_index, axis=1) # remove duplicates
            # data['residue', 'pae', 'residue'].edge_index = torch.from_numpy(
            #     edge_index
            # )

            data.threshold_values = threshold_values

            assert data.is_undirected()
            graph_file = os.path.join(self.graph_dir, f'{accession}.pt')
            embedding_file = os.path.join(self.embedding_dir, f'{accession}.pt')

            if self.rebuild or not os.path.exists(graph_file):
                torch.save(
                    data, os.path.join(self.graph_dir, f'{accession}.pt')
                )
            if self.rebuild or not os.path.exists(embedding_file):
                torch.save(
                    encoder(sequence),
                    os.path.join(self.embedding_dir, f'{accession}.pt'),
                )

        print(f' -> Graph processing time: {time.time() - start_time:.2f} seconds')

        # reclaim device memory
        del encoder

        print(
            ' -> Accessions successfully processed :',
            len(self.get_processable_accessions()),
        )

    def len(self):
        return self.processable_accessions.size

    def get(self, idx):

        filename = f'{self.processable_accessions[idx]}.pt'

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            data = torch.load(
                os.path.join(self.graph_dir, filename),
                weights_only=False
            )
            data['residue'].x = torch.load(
                os.path.join(self.embedding_dir, filename),
                weights_only=False
            )

        return data


if __name__ == '__main__':

    # meta_filename = 'metadata - preliminary training set (896 entries).csv'
    meta_filename = 'metadata - testing (temp).csv'
    meta_path = f'../../../datasets/{meta_filename}'

    edge_policy = '1CONT'
    thresholds = {
        'contact': '12',
        'codir': edge_policy,
        'coord': edge_policy,
        'deform': edge_policy,
    }
    dataset_version = 'v7a'
    embedding = 'prottrans'

    meta = np.loadtxt(
        meta_path,
        delimiter=',',
        dtype=np.str_
    )

    print(meta[:3])
    print(meta.shape)

    ds = Dataset(
        meta_file=meta_path,
        version=dataset_version,
        sequence_embedding=embedding,
        thresholds=thresholds,
        merge_edge_types=False,
        include_edge_weights=False,
        time_limit=60,
        transform=None,
        device=df_device,
        entries_should_be_ready=False,
        rebuild=False,
        float16_embeddings=False,
    )

    print(len(ds))
    print(ds[0])

    # np.savetxt(
    #     f'stats - {set_name}/processable_accessions.txt',
    #     ds.processable_accessions,
    #     fmt='%s',
    # )
