if __name__ == '__main__':
    from retrievers import AlphaFold_Retriever
    from enm import TNM_Computer
else:
    from .retrievers import AlphaFold_Retriever
    from .enm import TNM_Computer

import os, sys, json, prody, sortednp
import numpy as np
from tqdm import tqdm

coupling_types = ['codir', 'coord', 'deform']
edge_types = ['contact'] + coupling_types


def summarize_couplings(
    ref_accessions, stats_dir
):  # , contact_threshold=12

    print('# accessions in reference set:', ref_accessions.size)

    np.savetxt(
        f'{stats_dir}/reference_accessions.txt',
        ref_accessions,
        fmt='%s',
    )

    ### INSTANTIATE UTILITIES
    struct_retriever = AlphaFold_Retriever()
    tnm_computer = TNM_Computer()

    # ### ENSURE ALL ACCESSIONS ARE ACCESSIBLE
    # successful_accessions, _ = tnm_computer.batch_run(
    #     ref_accessions,
    #     [
    #         struct_retriever.path_to_file(a, item_type='pdb')
    #         for a in struct_retriever.unmodify_accession(ref_accessions)
    #     ],
    #     merge_output_files=True,  # merge small files into one
    #     timeout=20,  # increase if necessary
    #     debug=False,  # set to true if can't shutil.rmtree() on WSL
    # )
    # assert len(successful_accessions) == ref_accessions.size

    ### COLLATED INFORMATION IS SAVED HERE
    os.makedirs(stats_dir, exist_ok=True)

    ####################################################################
    # GET NUMBER OF CONTACT EDGES (12Å)
    ####################################################################

    contact_edge_file = f'{stats_dir}/count - contact.txt'
    if os.path.exists(contact_edge_file):
        contact_edge_numbers = np.loadtxt(contact_edge_file, dtype=np.int_)
        assert contact_edge_numbers.size == ref_accessions.size

    else:
        contact_edge_numbers = np.zeros((ref_accessions.size,), dtype=np.int_)

        pbar = tqdm(ref_accessions, dynamic_ncols=True, ascii=True)
        for acc_idx, acc in enumerate(pbar):
            pbar.set_description(f'# contact {acc}')

            path_to_pdb = struct_retriever.path_to_file(acc)

            atoms = prody.parsePDB(path_to_pdb, subset='ca')
            anm = prody.ANM(name=f'{acc}_CA')
            anm.buildHessian(atoms, cutoff=12.0)
            cont = -anm.getKirchhoff().astype(np.int_)  # the Laplacian matrix
            # np.fill_diagonal(cont, 1) # contact map completed here (with loops)
            edge_index = np.argwhere(cont == 1).T  # undirected graph
            n_cont_edges = int(edge_index.shape[1] / 2)

            contact_edge_numbers[acc_idx] = n_cont_edges

        np.savetxt(contact_edge_file, contact_edge_numbers, fmt='%d')

    total_contact_edges = np.sum(contact_edge_numbers)
    n_edges_to_write = 3 * total_contact_edges

    print(f'total # of contact edges: {total_contact_edges:,}')
    print(f'total # of edges to keep in memory: {n_edges_to_write:,}')
    print(
        f'array size: 3x {sys.getsizeof(np.zeros((n_edges_to_write,), dtype=np.float_)):,} bytes'
    )

    ####################################################################
    # ITERATE OVER ALL ACCESSIONS
    ####################################################################

    coupling_statistic_files = {
        ct: f'{stats_dir}/statistics - {ct}.txt' for ct in coupling_types
    }
    header = '# n_pairs,min,max,mean,std,' + ','.join(
        f'{p}%' for p in np.linspace(0, 100, 11, dtype=np.int_)[1:-1]
    )
    print(header)

    coupling_statistic_handles = {
        ct: open(coupling_statistic_files[ct], 'w') for ct in coupling_types
    }
    for ct in coupling_statistic_handles:
        coupling_statistic_handles[ct].write(header + '\n')

    # placeholder for DCONT
    top_ranking_values = {
        ct: np.zeros((n_edges_to_write,), dtype=np.float_)
        for ct in coupling_types
    }

    pbar = tqdm(ref_accessions, dynamic_ncols=True, ascii=True)
    for acc_idx, acc in enumerate(pbar):
        pbar.set_description(f'COUPLING {acc}')

        path_to_tnm_files = tnm_computer.path_to_outputs(acc)
        path_to_tnm_merged = tnm_computer.path_to_merged(acc)

        # READ MERGED TNM OUTPUT
        with open(path_to_tnm_merged, 'r') as f:
            merged_data = json.load(f)

        for ct in coupling_types:
            keyname = os.path.basename(path_to_tnm_files[ct])

            # READ COUPLINGS
            raw_data = np.loadtxt(
                merged_data[keyname].split('\n'), dtype=np.str_
            ).reshape((-1, 3))
            # edge_index = raw_data[:, :2].astype(np.int_).T
            couplings = np.sort(raw_data[:, 2].astype(np.float_))

            # CALCULATE STATISTICS
            n_pairs = couplings.size
            min_, max_ = couplings[0], couplings[-1]
            mean = np.mean(couplings)
            std = np.std(couplings)
            percentiles = np.percentile(
                couplings, np.linspace(0, 100, 21)[1:-1]
            )

            # WRITE TO FILE
            coupling_statistic_handles[ct].write(
                ','.join(
                    [
                        f'{v:.4f}'
                        for v in [n_pairs, min_, max_, mean, std]
                        + list(percentiles)
                    ]
                )
                + '\n'
            )

            # UPDATE TOP_RANKING_VALUES
            top_ranking_values[ct] = sortednp.merge(
                top_ranking_values[ct],
                couplings,
            )[-n_edges_to_write:]

    # close files
    for k in coupling_statistic_handles:
        coupling_statistic_handles[k].close()

    ####################################################################
    # COMPUTE THRESHOLD VALUES
    ####################################################################
    threshold_values = {
        'codir': {},
        'coord': {},
        'deform': {},
    }
    for ct in coupling_types:

        threshold_values[ct] = {}

        stats = np.loadtxt(coupling_statistic_files[ct], delimiter=',')
        n_pairs = stats[:, 0].astype(np.int_)
        min_, max_ = stats[:, 1], stats[:, 2]
        mean, std = stats[:, 3], stats[:, 4]

        amin, amax = np.min(min_), np.max(max_)
        amean = np.average(mean, weights=n_pairs)
        astd = np.sqrt(
            np.average((mean - amean) ** 2 + std ** 2, weights=n_pairs)
        )

        ### DCONT
        for n_dcont in [
            '0.25',
            '0.5',
            '0.75',
            '1',
            '1.25',
            '1.5',
            '1.75',
            '2',
            '2.5',
            '3',
        ]:
            n_edges_to_keep = int(float(n_dcont) * total_contact_edges)
            threshold = top_ranking_values[ct][-n_edges_to_keep]
            threshold_values[ct][f'{n_dcont}DCONT'] = threshold

        ### DSIGMA
        for n_dsigma in [
            '0',
            '0.5',
            '1',
            '1.5',
            '2',
            '2.5',
            '3',
            '3.5',
            '4',
            '4.5',
            '5',
        ]:
            threshold = amean + float(n_dsigma) * astd
            threshold_values[ct][f'{n_dsigma}DSIGMA'] = threshold

        ### DN
        for n_dn in ['5', '10', '15', '20', '25', '30', '35', '40', '45', '50']:
            threshold = amax - float(n_dn) / 100 * (amax - amin)
            threshold_values[ct][f'{n_dn}DN'] = threshold

        with open(f'{stats_dir}/thresholds - {ct}.json', 'w') as f:
            json.dump(threshold_values[ct], f, indent=4)


if __name__ == '__main__':

    # the threshold values will be determined from this set
    set_name = 'testing (temp)'
    # set_name = 'preliminary training set (896 entries)'

    meta_path = f'../../../datasets/metadata - {set_name}.csv'

    ref_accessions = np.array([
        f'{accession}-AFv6' for accession in np.loadtxt(
            meta_path,
            delimiter=',',
            usecols=0,
            dtype=np.str_,
        )
    ])

    summarize_couplings(
        ref_accessions,
        stats_dir=f'stats - {set_name}'
    )
