if __name__ in ['__main__', 'persistence_image']:
    from __init__ import processed_dir
else:
    from .__init__ import processed_dir

import os, prody, torch
import numpy as np
import gudhi as gd
import gudhi.representations
from tqdm import tqdm

df_simplex = 'alpha'
df_pi_range = [0, 50, 0, 50*(2**0.5)/2]
df_pi_size = [25, 25]

class PI_Computer:

    def __init__(self, simplex=df_simplex):

        self.pi_dir = os.path.join(processed_dir, 'persistence_images')

        failure_filename = 'pi-failed_entries.tsv'
        self.failure_path = os.path.join(self.pi_dir, failure_filename)

        self.simplex = simplex
        os.makedirs(os.path.join(self.pi_dir, self.simplex), exist_ok=True)

    def get_failed_accessions(self, return_reason=False):
        '''Returns accessions for which TNM computation failed.'''

        if (os.path.exists(self.failure_path) and
            os.stat(self.failure_path).st_size!=0):
            entries = np.loadtxt(self.failure_path,
                                 dtype=np.str_,
                                 delimiter='\t').reshape((-1,2))
        else:
            entries = np.empty((0,2), dtype=np.str_)

        if return_reason:
            return np.unique(entries, axis=0)
        else:
            return np.unique(entries[:,0])

    def path_to_file(self, accession):
        return os.path.join(self.pi_dir, self.simplex, f'{accession}.pt')

    def compute(self, accession, pdb_path, debug=False):

        save_path = self.path_to_file(accession)

        atoms = prody.parsePDB(pdb_path)
        coords = atoms.getCoords().tolist()

        # simplicial  complex
        scx = gd.AlphaComplex(points=coords).create_simplex_tree()

        # persistence image
        pi = gd.representations.PersistenceImage(
            bandwidth=1,
            weight=lambda x: max(0, x[1]*x[1]),
            im_range=df_pi_range,
            resolution=df_pi_size
        )

        scx.persistence()

        pInterval_d1 = scx.persistence_intervals_in_dimension(1)
        pInterval_d2 = scx.persistence_intervals_in_dimension(2)

        if pInterval_d1.size!=0 and pInterval_d2.size!=0:
            pers_img = pi.fit_transform([
                np.vstack((pInterval_d1, pInterval_d2))
            ])
        elif pInterval_d1.size!=0 and pInterval_d2.size==0:
            pers_img = pi.fit_transform([pInterval_d1])
        elif pInterval_d1.size==0 and pInterval_d2.size!=0:
            pers_img = pi.fit_transform([pInterval_d2])
        else:
            # computation failed
            return None

        # if computation is successful
        # np.save(save_path, pers_img)
        torch.save(torch.from_numpy(pers_img), save_path)

        return save_path

    def batch_compute(self, accessions, pdb_path_list, retry=False,
                      debug=False):

        successful_accessions = []
        failed_accessions = []

        accessions_to_skip = self.get_failed_accessions()
        pbar = tqdm(accessions, dynamic_ncols=True, ascii=True)
        for i, accession in enumerate(pbar):
            # if i == 5:
                # raise Exception
            pbar.set_description(f'PI {accession:<12s}')
            pdb_path = pdb_path_list[i]

            # skip entries that failed in previous runs
            if accession in accessions_to_skip and not retry:
                continue

            # keep a list of accessions successfully processed
            work_dir = self.compute(accession=accession,
                                    pdb_path=pdb_path,
                                    debug=debug)
            if work_dir is not None:
                successful_accessions.append(accession)
            else:
                failed_accessions.append(accession)

        # remove duplicated accessions in failure file
        np.savetxt(self.failure_path,
                   self.get_failed_accessions(return_reason=True),
                   fmt='%s\t%s')

        # failed accessions
        failed_accessions = np.unique(failed_accessions)

        return successful_accessions, failed_accessions

if __name__ == '__main__':

    pi_computer = PI_Computer()

    pi_computer.batch_compute(

        ['A0A0A0MQ89-AFv4', 'A0A0A0MQ68-AFv4', 'A0A0A0MQ99-AFv4'],

        ['temp/A0A0A0MQ89-AFv4.pdb', 'temp/A0A0A0MQ68-AFv4.pdb', 'temp/A0A0A0MQ99-AFv4.pdb']
    )
