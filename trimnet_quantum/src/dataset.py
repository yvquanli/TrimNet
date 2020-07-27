import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

import networkx as nx
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.transforms as T
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

from utils import Complete, angle, area_triangle, cal_dist


def load_dataset(path, specify_target):
    # apply transform
    class SpecifyTarget(object):
        def __call__(self, data):
            data.y = data.y[specify_target].view(-1)
            return data

    transform = T.Compose([SpecifyTarget(), Complete(), T.Distance(norm=True)])

    print('Check split dataset...')
    save_path = path + 'train_valid_test.ckpt'
    if os.path.isfile(save_path):
        trn, val, test = torch.load(save_path)
        trn.transform = transform
        val.transform = transform
        test.transform = transform
        return trn, val, test

    print('Load dataset...')
    dataset = QM9Dataset(root=path).shuffle()

    print('Split the dataset...')
    one_tenth = len(dataset) // 10
    test_dataset = dataset[: one_tenth]
    valid_dataset = dataset[one_tenth: one_tenth * 2]
    train_dataset = dataset[one_tenth * 2:]
    assert len(train_dataset) + len(valid_dataset) + len(test_dataset) == len(dataset)

    print('Save dataset...')
    torch.save([train_dataset, valid_dataset, test_dataset], save_path)
    return load_dataset(path, specify_target)


class QM9Dataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(QM9Dataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['gdb9.sdf', 'gdb9.sdf.csv']

    @property
    def processed_file_names(self):
        return 'processed_qm9.pt'

    def download(self):
        pass

    def process(self):
        data_path = self.raw_paths[0]
        target_path = self.raw_paths[1]
        self.property_names = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0',
                               'u298', 'h298', 'g298', 'cv']
        self.target = pd.read_csv(target_path, index_col='mol_id')
        self.target = self.target[self.property_names]
        supplier = Chem.SDMolSupplier(data_path, removeHs=False)
        data_list = []
        for i, mol in tqdm(enumerate(supplier)):
            data = self.mol2graph(mol)
            if data is not None:
                data.y = torch.FloatTensor(self.target.iloc[i, :])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_nodes(self, g):
        feat = []
        for n, d in g.nodes(data=True):
            h_t = []
            h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F', 'Cl', 'S']]
            h_t.append(d['a_num'])
            h_t.append(d['acceptor'])
            h_t.append(d['donor'])
            h_t.append(int(d['aromatic']))
            h_t += [int(d['hybridization'] == x) \
                    for x in (Chem.rdchem.HybridizationType.SP, \
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3)]
            h_t.append(d['num_h'])
            # 5 more
            h_t.append(d['formal_charge'])
            h_t.append(d['explicit_valence'])
            h_t.append(d['implicit_valence'])
            h_t.append(d['num_explicit_hs'])
            h_t.append(d['num_radical_electrons'])
            feat.append((n, h_t))
        feat.sort(key=lambda item: item[0])
        node_attr = torch.FloatTensor([item[1] for item in feat])
        return node_attr

    def get_edges(self, g):
        e = {}
        for n1, n2, d in g.edges(data=True):
            e_t = [int(d['b_type'] == x)
                   for x in (Chem.rdchem.BondType.SINGLE, \
                             Chem.rdchem.BondType.DOUBLE, \
                             Chem.rdchem.BondType.TRIPLE, \
                             Chem.rdchem.BondType.AROMATIC)]
            e_t.append(d['anglemax'])
            e_t.append(d['anglesum'])
            e_t.append(d['anglemean'])

            e_t.append(d['areamax'])
            e_t.append(d['areasum'])
            e_t.append(d['areamean'])

            e_t.append(d['dikmax'])
            e_t.append(d['diksum'])
            e_t.append(d['dikmean'])
            e_t.append(d['dij1'])
            e_t.append(d['dij2'])

            e[(n1, n2)] = e_t
        edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
        edge_attr = torch.FloatTensor(list(e.values()))
        return edge_index, edge_attr

    def mol2graph(self, mol):
        if mol is None: return None

        g = nx.DiGraph()

        # Create nodes
        assert len(mol.GetConformers()) == 1
        geom = mol.GetConformers()[0].GetPositions()
        for i in range(mol.GetNumAtoms()):
            atom_i = mol.GetAtomWithIdx(i)
            g.add_node(i,
                       a_type=atom_i.GetSymbol(),
                       a_num=atom_i.GetAtomicNum(),
                       acceptor=0,  # 0 for placeholder
                       donor=0,
                       aromatic=atom_i.GetIsAromatic(),
                       hybridization=atom_i.GetHybridization(),
                       num_h=atom_i.GetTotalNumHs(includeNeighbors=True),
                       # 5 more node features
                       formal_charge=atom_i.GetFormalCharge(),
                       explicit_valence=atom_i.GetExplicitValence(),
                       implicit_valence=atom_i.GetImplicitValence(),
                       num_explicit_hs=atom_i.GetNumExplicitHs(),
                       num_radical_electrons=atom_i.GetNumRadicalElectrons(),
                       )

        # Electron donor and acceptor
        fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
        feats = factory.GetFeaturesForMol(mol)
        for f in range(len(feats)):
            if feats[f].GetFamily() == 'Donor':
                for atom_id in feats[f].GetAtomIds():
                    g.nodes[atom_id]['donor'] = 1
            elif feats[f].GetFamily() == 'Acceptor':
                for atom_id in feats[f].GetAtomIds():
                    g.nodes[atom_id]['acceptor'] = 1

        # Read Edges
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                e_ij = mol.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    # cal angle and area
                    assert mol.GetNumAtoms() == len(geom)
                    angles_ijk = []
                    areas_ijk = []
                    dists_ik = []
                    for neighbor in mol.GetAtomWithIdx(j).GetNeighbors():
                        k = neighbor.GetIdx()
                        if mol.GetBondBetweenAtoms(j, k) is not None and i != k:
                            vector1 = geom[j] - geom[i]
                            vector2 = geom[k] - geom[i]
                            angles_ijk.append(angle(vector1, vector2))
                            areas_ijk.append(area_triangle(vector1, vector2))
                            dists_ik.append(cal_dist(geom[i], geom[k]))
                    angles_ijk = np.array(angles_ijk) if angles_ijk != [] else np.array([0.])
                    areas_ijk = np.array(areas_ijk) if areas_ijk != [] else np.array([0.])
                    dists_ik = np.array(dists_ik) if dists_ik != [] else np.array([0.])
                    dist_ij1 = cal_dist(geom[i], geom[j], ord=1)
                    dist_ij2 = cal_dist(geom[i], geom[j], ord=2)

                    g.add_edge(i, j,
                               b_type=e_ij.GetBondType(),

                               anglemax=angles_ijk.max(),
                               anglesum=angles_ijk.sum(),
                               anglemean=angles_ijk.mean(),

                               areamax=areas_ijk.max(),
                               areasum=areas_ijk.sum(),
                               areamean=areas_ijk.mean(),

                               dikmax=dists_ik.max(),
                               diksum=dists_ik.sum(),
                               dikmean=dists_ik.mean(),
                               dij1=dist_ij1,
                               dij2=dist_ij2,
                               )

        # Build pyg data
        node_attr = self.get_nodes(g)
        edge_index, edge_attr = self.get_edges(g)
        pos = torch.FloatTensor(geom)
        data = Data(
            x=node_attr,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=None,  # None as a placeholder
            # name=mol.GetProp('_Name'),
        )
        return data


if __name__ == '__main__':
    dataset = QM9Dataset('../dataset_qm9/')
