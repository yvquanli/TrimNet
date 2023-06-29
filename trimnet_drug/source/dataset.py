import os

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from utils import scaffold_randomized_spliting, onehot_encoding_unk
from utils import onehot_encoding, split_multi_label_containNan


def load_dataset_random_nan(path, dataset, seed, tasks=None):
    save_path = path + "processed/train_valid_test_{}_seed_{}.ckpt".format(
        dataset, seed
    )
    if os.path.isfile(save_path):
        trn, val, test = torch.load(save_path)
        return trn, val, test
    df = pd.read_csv(os.path.join(path, "raw/{}.csv".format(dataset)))
    smilesList = df.smiles.values
    print("number of all smiles: ", len(smilesList))
    remained_smiles = []
    canonical_smiles_list = []
    for smiles in smilesList:
        try:
            canonical_smiles_list.append(
                Chem.MolToSmiles(
                    Chem.MolFromSmiles(smiles), isomericSmiles=True
                )  # noqua
            )
            remained_smiles.append(smiles)
        except Exception:
            print("not successfully processed smiles: ", smiles)
            pass
    print("number of successfully processed smiles: ", len(remained_smiles))
    df = df[df["smiles"].isin(remained_smiles)].reset_index()

    weights = []
    random_seed = seed
    for i, task in enumerate(tasks):
        negative_df = df[df[task] == 0][["smiles", task]]
        positive_df = df[df[task] == 1][["smiles", task]]
        negative_test = negative_df.sample(frac=1 / 10,
                                           random_state=random_seed)
        negative_valid = negative_df.drop(negative_test.index).sample(
            frac=1 / 9, random_state=random_seed
        )
        negative_train = negative_df.drop(negative_test.index).drop(
            negative_valid.index
        )

        positive_test = positive_df.sample(frac=1 / 10,
                                           random_state=random_seed)
        positive_valid = positive_df.drop(positive_test.index).sample(
            frac=1 / 9, random_state=random_seed
        )
        positive_train = positive_df.drop(positive_test.index).drop(
            positive_valid.index
        )

        weights.append(
            [
                (positive_train.shape[0] + negative_train.shape[0])
                / negative_train.shape[0],
                (positive_train.shape[0] + negative_train.shape[0])
                / positive_train.shape[0],
            ]
        )

    trn = NanDataset(path, dataset, tasks, "train", seed)
    val = NanDataset(path, dataset, tasks, "valid", seed)
    test = NanDataset(path, dataset, tasks, "test", seed)
    trn.weights = weights
    torch.save([trn, val, test], save_path)
    return load_dataset_random_nan(path, dataset, seed)


def load_dataset_random(path, dataset, seed, tasks=None):
    save_path = path + "processed/train_valid_test_{}_seed_{}.ckpt".format(
        dataset, seed
    )
    if os.path.isfile(save_path):
        trn, val, test = torch.load(save_path)
        return trn, val, test
    pyg_dataset = MultiDataset(root=path, dataset=dataset, tasks=tasks)
    df = pd.read_csv(os.path.join(path, "raw/{}.csv".format(dataset)))
    smilesList = df.smiles.values
    print("number of all smiles: ", len(smilesList))
    remained_smiles = []
    canonical_smiles_list = []
    for smiles in smilesList:
        try:
            canonical_smiles_list.append(
                Chem.MolToSmiles(Chem.MolFromSmiles(smiles),
                                 isomericSmiles=True)
            )
            remained_smiles.append(smiles)
        except Exception:
            print("not successfully processed smiles: ", smiles)
            pass
    print("number of successfully processed smiles: ", len(remained_smiles))

    df = df[df["smiles"].isin(remained_smiles)].reset_index()
    if (
        dataset == "sider"
        or dataset == "clintox"
        or dataset == "tox21"
        or dataset == "ecoli"
        or dataset == "AID1706_binarized_sars"
    ):
        train_size = int(0.8 * len(pyg_dataset))
        val_size = int(0.1 * len(pyg_dataset))
        pyg_dataset = pyg_dataset.shuffle()
        trn, val, test = (
            pyg_dataset[:train_size],
            pyg_dataset[train_size: (train_size + val_size)],
            pyg_dataset[(train_size + val_size):],
        )
        weights = []
        for i, task in enumerate(tasks):
            negative_df = df[df[task] == 0][["smiles", task]]
            positive_df = df[df[task] == 1][["smiles", task]]
            neg_len = len(negative_df)
            pos_len = len(positive_df)
            weights.append(
                [(neg_len + pos_len) / neg_len, (neg_len + pos_len) / pos_len]
            )
        trn.weights = weights

    elif (
        dataset == "esol" or dataset == "freesolv" or
        dataset == "lipophilicity"
    ):  # 黎育权：esol  freesolv lip support
        train_size = int(0.8 * len(pyg_dataset))
        val_size = int(0.1 * len(pyg_dataset))
        # test_size = len(pyg_dataset) - train_size - val_size
        pyg_dataset = pyg_dataset.shuffle()
        trn, val, test = (
            pyg_dataset[:train_size],
            pyg_dataset[train_size:(train_size + val_size)],
            pyg_dataset[(train_size + val_size):],
        )
        trn.weights = "regression task has no class weights!"
    else:
        print("This dataset should not use this split method")
    torch.save([trn, val, test], save_path)
    return load_dataset_random(path, dataset, seed, tasks)


def load_dataset_scaffold(path, dataset="hiv", seed=628, tasks=None):
    save_path = path + "processed/train_valid_test_{}_seed_{}.ckpt".format(
        dataset, seed
    )
    if os.path.isfile(save_path):
        trn, val, test = torch.load(save_path)
        return trn, val, test

    pyg_dataset = MultiDataset(root=path, dataset=dataset, tasks=tasks)
    df = pd.read_csv(os.path.join(path, "raw/{}.csv".format(dataset)))
    smilesList = df.smiles.values
    print("number of all smiles: ", len(smilesList))
    remained_smiles = []
    canonical_smiles_list = []
    for smiles in smilesList:
        try:
            canonical_smiles_list.append(
                Chem.MolToSmiles(Chem.MolFromSmiles(smiles),
                                 isomericSmiles=True)
            )
            remained_smiles.append(smiles)
        except Exception:
            print("not successfully processed smiles: ", smiles)
            pass
    print("number of successfully processed smiles: ", len(remained_smiles))
    df = df[df["smiles"].isin(remained_smiles)].reset_index()

    trn_id, val_id, test_id, weights = scaffold_randomized_spliting(
        df, tasks=tasks, random_seed=seed
    )
    trn, val, test = (
        pyg_dataset[torch.LongTensor(trn_id)],
        pyg_dataset[torch.LongTensor(val_id)],
        pyg_dataset[torch.LongTensor(test_id)],
    )
    trn.weights = weights

    torch.save([trn, val, test], save_path)
    return load_dataset_scaffold(path, dataset, seed, tasks)


def atom_attr(mol, explicit_H=True, use_chirality=True):
    feat = []
    for i, atom in enumerate(mol.GetAtoms()):
        # if atom.GetDegree()>5:
        #     print(Chem.MolToSmiles(mol))
        #     print(atom.GetSymbol())
        results = (
            onehot_encoding_unk(
                atom.GetSymbol(),
                [
                    "B",
                    "C",
                    "N",
                    "O",
                    "F",
                    "Si",
                    "P",
                    "S",
                    "Cl",
                    "As",
                    "Se",
                    "Br",
                    "Te",
                    "I",
                    "At",
                    "other",
                ],
            )
            + onehot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # noqa: E501
            + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
            + onehot_encoding_unk(
                atom.GetHybridization(),
                [
                    Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2,
                    "other",
                ],
            )
            + [atom.GetIsAromatic()]
        )
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if not explicit_H:
            results = results + onehot_encoding_unk(
                atom.GetTotalNumHs(), [0, 1, 2, 3, 4]
            )
        if use_chirality:
            try:
                results = (
                    results
                    + onehot_encoding_unk(atom.GetProp("_CIPCode"), ["R", "S"])
                    + [atom.HasProp("_ChiralityPossible")]
                )
            #
            except Exception:
                results = results + [0, 0] + [atom.HasProp("_ChiralityPossible")]  # noqa: E501
        feat.append(results)

    return np.array(feat)


def bond_attr(mol, use_chirality=True):
    feat = []
    index = []
    n = mol.GetNumAtoms()
    for i in range(n):
        for j in range(n):
            if i != j:
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    bt = bond.GetBondType()
                    bond_feats = [
                        bt == Chem.rdchem.BondType.SINGLE,
                        bt == Chem.rdchem.BondType.DOUBLE,
                        bt == Chem.rdchem.BondType.TRIPLE,
                        bt == Chem.rdchem.BondType.AROMATIC,
                        bond.GetIsConjugated(),
                        bond.IsInRing(),
                    ]
                    if use_chirality:
                        bond_feats = bond_feats + onehot_encoding_unk(
                            str(bond.GetStereo()),
                            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"],
                        )
                    feat.append(bond_feats)
                    index.append([i, j])

    return np.array(index), np.array(feat)


class MultiDataset(InMemoryDataset):
    def __init__(
        self, root, dataset, tasks, transform=None, pre_transform=None,
        pre_filter=None
    ):
        self.tasks = tasks
        self.dataset = dataset

        self.weights = 0
        super(MultiDataset, self).__init__(root, transform, pre_transform,
                                           pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # os.remove(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["{}.csv".format(self.dataset)]

    @property
    def processed_file_names(self):
        return ["{}.pt".format(self.dataset)]

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        smilesList = df.smiles.values
        print("number of all smiles: ", len(smilesList))
        remained_smiles = []
        canonical_smiles_list = []
        for smiles in smilesList:
            try:
                canonical_smiles_list.append(
                    Chem.MolToSmiles(Chem.MolFromSmiles(smiles),
                                     isomericSmiles=True)
                )
                remained_smiles.append(smiles)
            except Exception:
                print("not successfully processed smiles: ", smiles)
                pass
        print("number of successfully processed smiles: ",
              len(remained_smiles))

        df = df[df["smiles"].isin(remained_smiles)].reset_index()
        target = df[self.tasks].values
        smilesList = df.smiles.values
        data_list = []

        for i, smi in enumerate(tqdm(smilesList)):
            mol = MolFromSmiles(smi)
            data = self.mol2graph(mol)

            if data is not None:
                label = target[i]
                label[np.isnan(label)] = 6
                data.y = torch.LongTensor([label])
                if (
                    self.dataset == "esol"
                    or self.dataset == "freesolv"
                    or self.dataset == "lipophilicity"
                ):
                    data.y = torch.FloatTensor([label])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def mol2graph(self, mol):
        if mol is None:
            return None
        node_attr = atom_attr(mol)
        edge_index, edge_attr = bond_attr(mol)
        # pos = torch.FloatTensor(geom)
        data = Data(
            x=torch.FloatTensor(node_attr),
            # pos=pos,
            edge_index=torch.LongTensor(edge_index).t(),
            edge_attr=torch.FloatTensor(edge_attr),
            y=None,  # None as a placeholder
        )
        return data


class NanDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        dataset,
        tasks,
        mode,
        seed,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.tasks = tasks
        self.dataset = dataset
        self.mode = mode
        self.seed = seed
        self.weights = 0
        super(NanDataset, self).__init__(root, transform, pre_transform,
                                         pre_filter)
        if self.mode == "train":
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif self.mode == "valid":
            self.data, self.slices = torch.load(self.processed_paths[1])
        else:
            self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        return ["{}.csv".format(self.dataset)]

    @property
    def processed_file_names(self):
        return [
            "{}_train_{}.pt".format(self.dataset, self.seed),
            "{}_val_{}.pt".format(self.dataset, self.seed),
            "{}_test_{}.pt".format(self.dataset, self.seed),
        ]

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        smilesList = df.smiles.values
        print("number of all smiles: ", len(smilesList))
        remained_smiles = []
        canonical_smiles_list = []
        for smiles in smilesList:
            try:
                canonical_smiles_list.append(
                    Chem.MolToSmiles(Chem.MolFromSmiles(smiles),
                                     isomericSmiles=True)
                )
                remained_smiles.append(smiles)
            except Exception:
                print("not successfully processed smiles: ", smiles)
                pass
        print("number of successfully processed smiles: ",
              len(remained_smiles))

        df = df[df["smiles"].isin(remained_smiles)].reset_index()

        trn_df, val_df, test_df, weights = split_multi_label_containNan(
            df, self.tasks, self.seed
        )
        self.weights = weights

        for n, dfs in enumerate([trn_df, val_df, test_df]):
            target = dfs[self.tasks].values
            smilesList = dfs.smiles.values
            data_list = []

            for i, smi in enumerate(tqdm(smilesList)):
                mol = MolFromSmiles(smi)
                data = self.mol2graph(mol)

                if data is not None:
                    label = target[i]
                    label[np.isnan(label)] = 6
                    data.y = torch.LongTensor([label])
                    data_list.append(data)

            if self.pre_filter is not None:
                data_list = [data for data in data_list
                             if self.pre_filter(data)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[n])

    def mol2graph(self, mol):
        if mol is None:
            return None
        node_attr = atom_attr(mol)
        edge_index, edge_attr = bond_attr(mol)
        # pos = torch.FloatTensor(geom)
        data = Data(
            x=torch.FloatTensor(node_attr),
            # pos=pos,
            edge_index=torch.LongTensor(edge_index).t(),
            edge_attr=torch.FloatTensor(edge_attr),
            y=None,  # None as a placeholder
        )
        return data
