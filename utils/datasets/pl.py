import os
import pickle
import lmdb
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from joblib import Parallel, delayed, cpu_count
from joblib.externals.loky import set_loky_pickler
set_loky_pickler("dill")
import pickle

from ..protein_ligand import PDBProtein, parse_sdf_file
from ..data import ProteinLigandData, torchify_dict


def pmap_multi(pickleable_fn, data, n_jobs=None, verbose=1, desc=None, **kwargs):
    """

    Parallel map using joblib.

    Parameters
    ----------
    pickleable_fn : callable
        Function to map over data.
    data : iterable
        Data over which we want to parallelize the function call.
    n_jobs : int, optional
        The maximum number of concurrently running jobs. By default, it is one less than
        the number of CPUs.
    verbose: int, optional
        The verbosity level. If nonzero, the function prints the progress messages.
        The frequency of the messages increases with the verbosity level. If above 10,
        it reports all iterations. If above 50, it sends the output to stdout.
    kwargs
        Additional arguments for :attr:`pickleable_fn`.

    Returns
    -------
    list
        The i-th element of the list corresponds to the output of applying
        :attr:`pickleable_fn` to :attr:`data[i]`.
    """
    if n_jobs is None:
        n_jobs = cpu_count() - 1
    # n_jobs = 60
    results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
    delayed(pickleable_fn)(*d, **kwargs) for i, d in tqdm(enumerate(data),desc=desc)
    )

    return results

def custom_preprocess(data, transform):
    from torch_geometric.nn.pool import knn_graph, radius, knn
    data = transform(data)
    protein_atom_feature = data.protein_atom_feature
    protein_pos = data.protein_pos
    
    is_backbone = (protein_atom_feature[:,5+20:5+20+1]).view(-1).bool()
    is_N = (protein_atom_feature[:,:5].argmax(dim=1)==1).view(-1)
    
    assign_index = radius(x = data.ligand_pos, y = protein_pos, r=5, num_workers=16)
    is_in_5A = torch.zeros_like(is_N) == 1
    is_in_5A[assign_index[0].unique()]=True
    
    protein_pos = protein_pos[(is_backbone&is_N)|is_in_5A]
    protein_atom_feature = protein_atom_feature[(is_backbone&is_N)|is_in_5A]

    assign_index = radius(x = data.ligand_pos, y = protein_pos, r=5, num_workers=16)
    # assign_index = knn(x = data.ligand_pos, y = protein_pos, k=5, num_workers=16)

    dist = torch.norm(data.ligand_pos[assign_index[1]] - protein_pos[assign_index[0]], dim=-1)
    s_idx = torch.argsort(dist)[:5]
    assign_index = assign_index[:,s_idx]
    
    random_choice = lambda x: x[:,torch.randint(x.shape[0], (1,))]
    data.start_edge = random_choice(assign_index)
    start_idx = data.start_edge[0]
    
    protein_one_knn_edge_index = knn_graph(protein_pos, k=48, flow='target_to_source', num_workers=16)#check 一下是compose还是protein[3564, 3]在dataloader处
    
    data.start_idx = start_idx
    data.protein_is_backbone = is_backbone
    data.protein_is_in_5A = is_in_5A
    data.protein_is_N = is_N
    data.protein_one_knn_edge_index = protein_one_knn_edge_index
    return data



def handle_per_task(i, pocket_fn, ligand_fn, raw_path, transform):
    from ..mol_utils import get_metric
    try:
        pocket_dict = PDBProtein(os.path.join(raw_path, pocket_fn)).to_dict_atom()
        ligand_dict = parse_sdf_file(os.path.join(raw_path, ligand_fn))
        
        rdmol = ligand_dict['rdmol']
        metrics = get_metric(pocket_fn, rdmol, use_vina=True, task_id=(pocket_fn+"_SEP_"+ligand_fn).replace("/", "__"))
        ligand_dict.update(metrics)

        data = ProteinLigandData.from_protein_ligand_dicts(
            protein_dict=torchify_dict(pocket_dict),
            ligand_dict=torchify_dict(ligand_dict),
        )
        data.protein_filename = pocket_fn
        data.ligand_filename = ligand_fn
        
        data = custom_preprocess(data, transform)
        
        return (i, data)
    except:
        return None

def cut_list(lists, cut_len):
    res_data = []
    if len(lists) > cut_len:
        for i in range(int(len(lists) / cut_len)):
            cut_a = lists[cut_len * i:cut_len * (i + 1)]
            res_data.append(cut_a)

        last_data = lists[int(len(lists) / cut_len) * cut_len:]
        if last_data:
            res_data.append(last_data)
    else:
        res_data.append(lists)
    return res_data



class PocketLigandPairDataset(Dataset):

    def __init__(self, raw_path, transform=None, name=None):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.name = name
        if name is None:
            self.processed_path = os.path.join(os.path.dirname(self.raw_path), os.path.basename(self.raw_path) + '_processed.lmdb')
            self.name2id_path = os.path.join(os.path.dirname(self.raw_path), os.path.basename(self.raw_path) + '_name2id.pt')
        else:
            self.processed_path = os.path.join(os.path.dirname(self.raw_path), os.path.basename(self.raw_path) + f'_processed_{name}.lmdb')
            self.name2id_path = os.path.join(os.path.dirname(self.raw_path), os.path.basename(self.raw_path) + f'_name2id.pt')
        self.transform = transform
        self.db = None

        self.keys = None

        if not os.path.exists(self.processed_path):
            self._process()
            self._precompute_name2id()

        self.name2id = torch.load(self.name2id_path)

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=50*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))


    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
        
    def _process(self):
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)
            # index = index[:1000]

        tasks = []
        num_skipped = 0
        for i, (pocket_fn, ligand_fn, _, rmsd_str) in enumerate(tqdm(index)):
            if pocket_fn is None: continue
            
            tasks.append((i, pocket_fn, ligand_fn))
        
        processed_data = 0
        invalid_data = 0
        for subtasks in cut_list(tasks, 10000):
            db = lmdb.open(
                self.processed_path,
                map_size=50*(1024*1024*1024),   # 15GB
                create=True,
                subdir=False,
                readonly=False, # Writable
            )
            
            with db.begin(write=True, buffers=True) as txn:
                data_list = pmap_multi(handle_per_task, [(i, pocket_fn, ligand_fn) for (i, pocket_fn, ligand_fn) in subtasks], raw_path=self.raw_path, transform=self.transform)
                
                L1 = len(data_list)
                data_list = [one for one in data_list if one is not None]
                L2 = len(data_list)
                invalid_data += (L1 - L2)
                
                
                
                for i, data in tqdm(data_list):
                    txn.put(
                            key = str(i).encode(),
                            value = pickle.dumps(data)
                        )
            db.close()
            processed_data += len(subtasks)
            print("processed_data: {}".format(processed_data))
            
        print("invalid data: {}".format(invalid_data))
                

    def _precompute_name2id(self):
        name2id = {}
        for i in tqdm(range(self.__len__()), 'Indexing'):
            try:
                data = self.__getitem__(i)
            except AssertionError as e:
                print(i, e)
                continue
            name = (data.protein_filename, data.ligand_filename)
            name2id[name] = i
        torch.save(name2id, self.name2id_path)
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data.id = idx
        assert data.protein_pos.size(0) > 0
        if self.transform is not None:
            data = self.transform(data)
        return data
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    PocketLigandPairDataset(args.path)
