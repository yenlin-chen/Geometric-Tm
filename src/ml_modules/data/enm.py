if __name__ in ['__main__', 'enm']:
    from __init__ import collation_dir
else:
    from .__init__ import collation_dir

import os, shutil, subprocess, platform, json
import numpy as np
from tqdm import tqdm


class TNM_Computer:
    '''Computes TNM features for a given protein structure.

    The TNM program must be installed and accessible via command line
    (i.e. in PATH). The program is published at
    https://github.com/ugobas/tnm, and is described in
    Phys. Rev. Lett. 104, 228103 (2010).
    '''

    def __init__(
        self, n_sigmas=-10000  # output all couplings (no thresholding)
    ):

        self.tnm_dir = os.path.join(collation_dir, 'TNM')
        self.setup_name = f'run-sigma{n_sigmas}'

        failure_filename = 'tnm-failed_entries.tsv'
        self.failure_path = os.path.join(self.tnm_dir, failure_filename)

        # template_path = os.path.join(self.tnm_dir, 'template.in')
        template_path = os.path.join(
            os.path.dirname(__file__), 'template-tnm.in'
        )
        with open(template_path, 'r') as f:
            self.template_script = f.read()

        self.n_sigmas = n_sigmas

    def get_failed_accessions(self, return_reason=False):
        '''Returns accessions for which TNM computation failed.'''

        if (
            os.path.exists(self.failure_path)
            and os.stat(self.failure_path).st_size != 0
        ):
            entries = np.loadtxt(
                self.failure_path, dtype=np.str_, delimiter='\t'
            ).reshape((-1, 2))
        else:
            entries = np.empty((0, 2), dtype=np.str_)

        if return_reason:
            return np.unique(entries, axis=0)
        else:
            return np.unique(entries[:, 0])

    def path_to_work(self, accession):
        return os.path.join(self.tnm_dir, self.setup_name, accession)

    def path_to_outputs(self, accession, chains='A'):

        work_dir = self.path_to_work(accession)

        # chains = accession.split('-')[1].replace('_', '')
        prefix = f'{accession}{chains}_MIN{4.5:.1f}_ALL_PHIPSIPSI'
        filenames = {
            'mapping': f'{prefix}.names.dat',
            'cont': f'{prefix}_Cont_Mat.txt',
            'coord': f'{prefix}.coordination_coupling.dat',
            'codir': f'{prefix}.directionality_coupling.dat',
            'deform': f'{prefix}.deformation_coupling.dat',
            'bfactor': f'{prefix}.MSF.dat',
        }

        return {k: os.path.join(work_dir, filenames[k]) for k in filenames}

    def path_to_merged(self, accession):
        return os.path.join(self.tnm_dir, self.setup_name, f'{accession}.json')

    def run(
        self,
        accession,
        pdb_path,
        timeout,
    ):

        '''Runs TNM program on a given protein structure.

        Parameters
        ----------
        accession : str
            PDB accession code.
        pdb_path : str
            Path to PDB file.
        timeout : int, optional
            Maximum time (in seconds) to wait for the program to finish.

        Returns
        -------
        work_dir : str
            Path to working directory.
        '''

        work_dir = self.path_to_work(accession)
        script_file = os.path.join(work_dir, 'tnm.in')
        tnm_log_file = os.path.join(work_dir, 'tnm.log')

        path_to_files = self.path_to_outputs(accession)

        ################################################################
        # check if program output exists
        ################################################################
        # check if merged file exists
        if os.path.exists(self.path_to_merged(accession)):
            return work_dir
        else:
            os.makedirs(work_dir, exist_ok=True)

        # check if execution is successful by reading log file
        if os.path.exists(tnm_log_file) and os.stat(tnm_log_file).st_size != 0:
            with open(tnm_log_file, 'r') as f:
                final_line = f.readlines()[-1]
        else:
            final_line = ''

        # check if required output files exist
        file_existence = []
        for key in path_to_files:
            path_to_file = path_to_files[key]
            file_existence.append(os.path.exists(path_to_file))

        if all(file_existence) and final_line.startswith('Total Time'):
            self.merge_output_file(accession)
            return work_dir

        ################################################################
        # run program
        ################################################################
        # modify template script
        replacements = [
            ('PDBID_PLACEHOLDER', pdb_path),
            ('CHAINID_PLACEHOLDER', 'ALL'),
            ('CUTOFF', '4.5'),
            ('SIGMA_PLACEHOLDER', str(self.n_sigmas)),
        ]
        script_content = self.template_script
        for old, new in replacements:
            script_content = script_content.replace(old, new)

        # save modified script
        with open(script_file, 'w+') as f:
            f.write(script_content)

        # change directory to `work_dir` and execute the script
        cwd = os.getcwd()
        f_log = open(tnm_log_file, 'w')
        os.chdir(work_dir)
        try:
            # tnm software must be in PATH
            subprocess.run(
                ['tnm', '-file', script_file], stdout=f_log, timeout=timeout
            )
        except subprocess.TimeoutExpired as err:
            # timeouts are considered failures
            os.chdir(cwd)
            f_log.close()
            shutil.rmtree(work_dir)
            with open(self.failure_path, 'a+') as f:
                f.write(
                    f'{accession}\t'
                    f'timeout ({timeout}s on {platform.node()})\n'
                )
            return None
        except Exception as err:
            # remove `work_dir` in case of other errors
            os.chdir(cwd)
            f_log.close()
            shutil.rmtree(work_dir)
            raise
        else:
            os.chdir(cwd)
            f_log.close()

        # check if execution is successful by reading log file
        with open(tnm_log_file, 'r') as f:
            final_line = f.readlines()[-1]
        if not final_line.startswith('Total Time'):
            shutil.rmtree(work_dir)
            with open(self.failure_path, 'a+') as f:
                f.write(f'{accession}\tincomplete execution\n')
            return None
        # also check if required output files exist
        file_existence = []
        for key in path_to_files:
            path_to_file = path_to_files[key]
            file_existence.append(os.path.exists(path_to_file))
        if not all(file_existence):
            shutil.rmtree(work_dir)
            with open(self.failure_path, 'a+') as f:
                f.write(f'{accession}\tmissing files\n')
            return None

        if os.path.exists(work_dir):
            self.merge_output_file(accession)
            return work_dir # does not remove entry in failure file
        else:
            with open(self.failure_path, 'a+') as f:
                f.write(f'{accession}\tmissing directory\n')
            return None

    def batch_run(
        self,
        accessions,
        pdb_path_list,
        retry=False,
        timeout=60,
    ):

        '''Executes TNM software on a list of protein accessions.

        Calls `run` method for each accession in `accessions` list. Also
        saves a file containing accessions that failed to run.

        Parameters
        ----------
        accessions : list of str
            List of PDB accession codes.
        pdb_path_list : list of str
            List of paths to PDB files.
        retry : bool
            If True, retry failed accessions.
        '''

        successful_accessions = []
        failed_accessions = []

        accessions_to_skip = self.get_failed_accessions()

        pbar = tqdm(accessions, dynamic_ncols=True, ascii=True)
        for i, accession in enumerate(pbar):
            # if i == 5:
            # raise Exception
            pbar.set_description(f'TNM {accession:<12s}')
            pdb_path = pdb_path_list[i]

            if accession in accessions_to_skip:
                # skip entries that failed in previous runs
                if not retry:
                    failed_accessions.append(accession)
                    continue
                # remove previous record from failure file
                else:
                    np.savetxt(
                        self.failure_path,
                        self.get_failed_accessions(return_reason=True)[
                            accessions_to_skip!=accession
                        ],
                        fmt='%s\t%s'
                    )
                    accessions_to_skip = self.get_failed_accessions()

            # keep a list of accessions successfully processed
            work_dir = self.run(
                accession=accession,
                pdb_path=pdb_path,
                timeout=timeout,
            )
            if work_dir is not None:
                successful_accessions.append(accession)
            else:
                failed_accessions.append(accession)

        # remove duplicated accessions in failure file
        np.savetxt(
            self.failure_path,
            self.get_failed_accessions(return_reason=True),
            fmt='%s\t%s'
        )

        # failed accessions
        failed_accessions = np.unique(failed_accessions)

        return successful_accessions, failed_accessions

    def merge_output_file(self, accession):
        '''Merge output files generated by TNM program.

        Parameters
        ----------
        accessions : str
            PDB accession codes.
        '''

        path_to_work = self.path_to_work(accession)
        filenames = os.listdir(path_to_work)

        # read all output files
        all_content = {}
        for filename in filenames:
            with open(os.path.join(path_to_work, filename), 'r') as f:
                content = f.read()
            all_content[filename] = content

        # save merged output file
        output_dir = self.path_to_merged(accession)

        with open(output_dir, 'w+', encoding='utf-8') as f:
            json.dump(
                all_content,
                f,
                indent=4,
                sort_keys=True,
            )

        shutil.rmtree(path_to_work)

    def cleanup_failed(self):
        '''Remove directories of failed entries. Does not work in Windows WSL.'''
        failed_accessions = self.get_failed_accessions()

        removed_accessions = []
        for accession in failed_accessions:
            work_dir = self.path_to_work(accession)
            if os.path.exists(work_dir):
                removed_accessions.append(accession)
                shutil.rmtree(work_dir)

        print('# of directories removed:', len(removed_accessions))
        return removed_accessions

    def get_resnames(self, accession, return_ids=False):
        '''Get residue names from TNM output files.

        Reads `names.dat` generated by TNM program. Ensures that there
        are no missing residues in the sequence.

        Parameters
        ----------
        accession : str
            PDB accession code.

        Returns
        -------
        resnames : list of str (np.ndarray)
            List of residue names.
        resids : list of int (np.ndarray)
            List of residue identifiers.

        '''

        map_path = self.path_to_outputs(accession)['mapping']
        keyname = os.path.basename(map_path)

        # read merged output
        with open(self.path_to_merged(accession), 'r') as f:
            mapping_str = json.load(f)[keyname]

        mapping = np.loadtxt(mapping_str.split('\n'), dtype=np.str_)

        ### get residue identifier for TNM (used in dynamic coupling files)
        dc_idx = mapping[:, 0]

        # check if dc_idx are all integers
        if not all([s.isdecimal() for s in dc_idx]):
            raise ValueError('dc_idx are not all integers')
        # further check if dc_idx is sequential
        dc_idx = np.array(dc_idx, dtype=np.int_)
        if not (np.diff(dc_idx) == 1).all():
            raise ValueError('dc_idx is not sequential')

        ### get residue identifier in .pdb file (input to TNM program)
        ### also get residue type
        sp = np.char.split(mapping[:, 1], sep='_')
        resnames = np.array([r[0] for r in sp])
        auth_seq_ids = np.array([r[1] for r in sp], dtype=np.str_)
        auth_asym_id = np.array([r[2] for r in sp], dtype=np.str_)

        if return_ids:
            identifiers = {
                'dc_idx': dc_idx,
                'auth_seq_ids': auth_seq_ids,
                'auth_asym_id': auth_asym_id,
            }
            return resnames, identifiers
        else:
            return resnames


if __name__ == '__main__':

    meta_filename = 'metadata - testing (temp).csv'
    meta_path = f'../../../datasets/{meta_filename}'

    accessions = np.loadtxt(
        meta_path,
        delimiter=',',
        usecols=0,
        dtype=np.str_
    )[:3]

    accessions_mod = [f'{accession}-AFv6' for accession in accessions]
    struct_path_list = [
        f'../../../../external/AlphaFoldDB/pdb/{accession_mod}.pdb'
        for accession_mod in accessions_mod
    ]

    print(accessions_mod)
    print(len(accessions_mod))

    tnm_computer = TNM_Computer()
    successful, failed = tnm_computer.batch_run(
        accessions_mod, struct_path_list, retry=False
    )

    print(successful)
    print(failed)

    # # REMOVE FAILED ENTRIES, INCLUDING DIRECTORIES
    # removed_accessions = tnm_computer.cleanup_failed()
    # print(removed_accessions)
    # print(len(removed_accessions))
