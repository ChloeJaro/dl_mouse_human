import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from skorch.helper import DataFrameTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from torch.utils.data import Subset, TensorDataset
from torch import Tensor

#from joblib import Parallel, delayed
#from joblib.externals.loky.backend.context import get_context # For parallel launch with hydra


class GeneDataset(Dataset):
    def __init__(self, data_path, intersct_data_path, labelcol):
        df = pd.read_csv(data_path)

        region_columns_mask = df.columns.str.match("Region")

        input_df = df.loc[:, ~region_columns_mask]

        intersct_df = pd.read_csv(intersct_data_path)

        input_df = input_df.loc[:, input_df.columns.isin(intersct_df.columns)]

        target_df = df[[labelcol]].copy()

        target_df[labelcol] = target_df[labelcol].astype("category")

        dftx = DataFrameTransformer()

        input_arr = dftx.fit_transform(input_df)
        target_arr = dftx.fit_transform(target_df)

        self.input_arr = input_arr["X"]
        self.target_arr = target_arr[labelcol]

    def __len__(self):
        return len(self.input_arr)

    def __getitem__(self, idx):
        x = self.input_arr[idx]
        y = self.target_arr[idx]

        return torch.tensor(x), torch.tensor(y)
    
class GetWeights(Dataset):
    def __init__(self, ref_data_path, labelcol):
        ref = pd.read_csv(ref_data_path)
        df_labels = ref[[labelcol]].copy()
        df_labels[labelcol] = df_labels[labelcol].astype("category")
        dtfx=DataFrameTransformer()
        target_arr = dtfx.fit_transform(df_labels)
        y_temp = target_arr[labelcol]
        _, n_classes = np.unique(y_temp, return_counts=True)
        self.balance_weights = len(y_temp) / (n_classes * np.bincount(y_temp))
    def __len__(self):
        return len(self.balance_weights)



class CrossValDataset(Dataset):
    def __init__(self, ref_data_path, cor_data_path, sag_data_path, labelcol, seed):
        # Get genes in the coronal dataset (includes ducplicates)
        ref = pd.read_csv(ref_data_path)
        cor = pd.read_csv(cor_data_path)
        sag = pd.read_csv(sag_data_path)

        # Identify and remove regions column
        reg_col_ref = ref.columns.str.match('Region')
        input_ref = ref.loc[:, ~reg_col_ref]
        reg_col_cor = cor.columns.str.match('Region')
        input_cor = cor.loc[:, ~reg_col_cor]
        reg_col_sag = sag.columns.str.match('Region')
        input_sag = sag.loc[:, ~reg_col_sag]

        # Rename the columns with duplicates
        input_cor.columns = input_cor.columns.str.replace('\.\.\.[0-9]+', '', regex = True)

        # Intersect genes with ref (homologous gene set - NB there are still duplicate exps in the cor set)
        input_cor_ = input_cor[input_cor.columns.intersection(input_ref.columns)]
        input_sag_ = input_sag[input_sag.columns.intersection(input_ref.columns)]
        
        # Get intersection of coronal and sagittal sets
        input_cor__ = input_cor_[input_cor_.columns.intersection(input_sag_.columns)]
        input_sag__ = input_sag_[input_sag_.columns.intersection(input_cor_.columns)]

        # Get unique genes 
        genes_unique = np.unique(input_cor__.columns)
        genes_cor = input_cor__.columns
        genes_sag = input_sag__.columns

        df_labels = cor[[labelcol]].copy()
        df_labels[labelcol] = df_labels[labelcol].astype("category")

        # Initialize random number generator
        rng = np.random.default_rng(seed = seed)

        # Initialize train and val dataframes
        
        train = pd.DataFrame(np.empty((cor.shape[0], len(genes_unique)),  dtype = 'float'), columns = genes_unique)
        val = pd.DataFrame(np.empty_like(train, dtype = 'float'), columns = genes_unique)
        
        for i, gene in enumerate(genes_unique):
            if np.sum(genes_cor == gene) > 1:

                # Extract duplicate experiments
                df_choices = input_cor__.loc[:, genes_cor == gene]

                # Randomly choose one exp for the training set
                choices = np.arange(0, df_choices.shape[1])
                choice_train = rng.choice(choices, 1)

                # Randomly choose one of the duplicates for validation
                choice_val = rng.choice(choices[choices != choice_train], 1)

                # Assign data to training and validation sets
                train.iloc[:,i] = df_choices.iloc[:, choice_train]
                val.iloc[:,i] = df_choices.iloc[:, choice_val]
            
            # If the gene is unique in the coronal set, choose betw cor and sag
            else:
                # Random binary choice
                choice_train = rng.choice([0,1], 1)

                # Assign cor data to train and sag to val, or vice versa
                if choice_train[0] == 0:
                    train.iloc[:,i] = input_cor__.loc[:, input_cor__.columns == gene]
                    val.iloc[:,i] = input_sag__.loc[:, input_sag__.columns == gene]
                else:
                    train.iloc[:,i] = input_sag__.loc[:, input_sag__.columns == gene]
                    val.iloc[:,i] = input_cor__.loc[:, input_cor__.columns == gene]

        dftx = DataFrameTransformer()
        scale = StandardScaler()
        center = StandardScaler(with_std = False)
        transpose = FunctionTransformer(np.transpose)

        processing_pipeline = Pipeline([('transpose1', transpose), 
                                            ('scale', scale),  
                                            ('transpose2', transpose),
                                            ('center', center),
                                            ])
        
        X_train_pp = processing_pipeline.fit_transform(train)
        X_val_pp = processing_pipeline.fit_transform(val)
        target_arr = dftx.fit_transform(df_labels)
        
        y_temp = target_arr[labelcol]
  
        self.train_arr = X_train_pp
        self.val_arr = X_val_pp 
        self.target_arr = y_temp
        
    def __len__(self):
        return len(self.target_arr)

    def __getitem__(self, idx):
        train = self.train_arr[idx]
        val = self.val_arr[idx]
        y = self.target_arr[idx]
        train_tens = torch.tensor(train)
        val_tens = torch.tensor(val)

        return train_tens, val_tens, torch.tensor(y)


class MouseHumanDataModule(pl.LightningDataModule):
    def __init__(
        self,
        seed: int,
        mouse_voxel_data_path: str,
        human_voxel_data_path: str,
        mouse_region_data_path: str,
        human_region_data_path: str,
        coronal_maskcoronal_path: str,
        coronal_masksagittal_path: str,
        sagittal_masksagittal_path: str,
        mouse_labelcol: str,
        human_labelcol: str,
        train_bsize: int,
        valid_bsize: int,
        num_workers: int,
    ):
        super().__init__()

        self.seed = seed
        self.mouse_voxel_data_path = mouse_voxel_data_path
        self.human_voxel_data_path = human_voxel_data_path
        self.coronal_maskcoronal_path = coronal_maskcoronal_path
        self.coronal_masksagittal_path = coronal_masksagittal_path
        self.sagittal_masksagittal_path = sagittal_masksagittal_path
        self.mouse_labelcol = mouse_labelcol
        self.human_labelcol = human_labelcol
        self.train_bsize = train_bsize
        self.valid_bsize = valid_bsize
        self.num_workers = num_workers

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        """ self.train_ds = GeneDataset(
            data_path=self.mouse_voxel_data_path,
            intersct_data_path=self.human_voxel_data_path,
            labelcol=self.mouse_labelcol,
        )
        """
        #full_ds = GeneDataset( Change GeneDataset to CrossValDataset return train and val tensors
        crossval_ds = CrossValDataset(
            ref_data_path=self.coronal_maskcoronal_path,
            cor_data_path=self.coronal_masksagittal_path,
            sag_data_path=self.sagittal_masksagittal_path,
            labelcol=self.mouse_labelcol,
            seed=self.seed,
        )
        self.train_ds = TensorDataset(torch.tensor(crossval_ds.train_arr.astype('float32')), torch.tensor(crossval_ds.target_arr).type(torch.LongTensor))
        self.val_ds = TensorDataset(torch.tensor(crossval_ds.val_arr.astype('float32')), torch.tensor(crossval_ds.target_arr).type(torch.LongTensor))

    def train_dataloader(self):
        #sampler=torch.utils.data.sampler.WeightedRandomSampler(self.weights, self.train_bsize)
        return DataLoader(
            self.train_ds,
            batch_size=self.train_bsize,
            num_workers=self.num_workers,
            shuffle=True, # changed to False
            #multiprocessing_context=get_context('loky'), # for use with joblib
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.valid_bsize,
            num_workers=self.num_workers,
            shuffle=False,
            #multiprocessing_context=get_context('loky'),
        )

def encode(
    trainer,
    ckpt_path,
    data_path,
    intersct_data_path,
    labelcol,
    output_file_path,
    bsize,
    num_workers,
):
    dataset = GeneDataset(
        data_path=data_path,
        intersct_data_path=intersct_data_path,
        labelcol=labelcol,
    )

    dataloader = DataLoader(
        dataset, batch_size=bsize, num_workers=num_workers, shuffle=False
    )

    #preds, decode, classifs, in_labels = trainer.predict(dataloaders=dataloader, ckpt_path=ckpt_path)
    preds = trainer.predict(dataloaders=dataloader, ckpt_path=ckpt_path)
    preds = [preds[i][0] for i in range(len(preds))]
    preds = torch.cat(preds, dim=0)

    preds = preds.detach().cpu().numpy()

    preds_df = pd.DataFrame(preds)

    data_df = pd.read_csv(data_path)

    preds_df["Region"] = data_df[labelcol]

    preds_df.to_csv(output_file_path, index=False)


def autoencode(
    trainer,
    model,
    data_path,
    intersct_data_path,
    labelcol,
    output_file_path,
    bsize,
    num_workers,
):
    dataset = GeneDataset(
        data_path=data_path,
        intersct_data_path=intersct_data_path,
        labelcol=labelcol,
    )

    dataloader = DataLoader(
        dataset, batch_size=bsize, num_workers=num_workers, shuffle=False
    )

    preds = trainer.predict(model, dataloaders=dataloader)
    preds = [preds[i][1] for i in range(len(preds))]

    preds = torch.cat(preds, dim=0)

    preds = preds.detach().cpu().numpy()

    preds_df = pd.DataFrame(preds)

    data_df = pd.read_csv(data_path)

    preds_df["Region"] = data_df[labelcol]

    preds_df.to_csv(output_file_path, index=False)



def classify(
    trainer,
    model,
    data_path,
    intersct_data_path,
    labelcol,
    output_file_path,
    labels_file_path,
    bsize,
    num_workers,
):
    dataset = GeneDataset(
        data_path=data_path,
        intersct_data_path=intersct_data_path,
        labelcol=labelcol,
    )

    dataloader = DataLoader(
        dataset, batch_size=bsize, num_workers=num_workers, shuffle=False
    )

    #encode, decode, preds, in_labels = trainer.predict(dataloaders=dataloader, model=model)
    preds =  trainer.predict(dataloaders=dataloader, model=model)
    preds = [preds[i][2] for i in range(len(preds))]
    in_labels = [preds[i][3] for i in range(len(preds))]
    
    preds = torch.cat(preds, dim=0)
    in_labels = torch.cat(in_labels, dim=0)

    preds = preds.detach().cpu().numpy()
    in_labels = in_labels.detach().cpu().numpy()

    preds_df = pd.DataFrame(preds)
    preds_df.to_csv(output_file_path, index=False)
    labels_df = pd.DataFrame(in_labels)

    labels_df.to_csv(labels_file_path, index=False)

