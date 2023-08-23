import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import pandas as pd
from skorch.helper import DataFrameTransformer



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


class MouseHumanDataModule(pl.LightningDataModule):
    def __init__(
        self,
        mouse_voxel_data_path: str,
        human_voxel_data_path: str,
        mouse_region_data_path: str,
        human_region_data_path: str,
        mouse_labelcol: str,
        human_labelcol: str,
        train_bsize: int,
        valid_bsize: int,
        num_workers: int,
    ):
        super().__init__()

        self.mouse_voxel_data_path = mouse_voxel_data_path
        self.human_voxel_data_path = human_voxel_data_path
        self.mouse_labelcol = mouse_labelcol
        self.human_labelcol = human_labelcol
        self.train_bsize = train_bsize
        self.valid_bsize = valid_bsize
        self.num_workers = num_workers

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        full_ds = GeneDataset(
            data_path=self.mouse_voxel_data_path,
            intersct_data_path=self.human_voxel_data_path,
            labelcol=self.mouse_labelcol,
        )
        train_size=int(0.8*len(full_ds))
        val_size=len(full_ds)-train_size
        self.train_ds, self.val_ds = random_split(full_ds, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.train_bsize,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.valid_bsize,
            num_workers=self.num_workers,
            shuffle=False,
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

    preds = trainer.predict(dataloaders=dataloader, ckpt_path=ckpt_path)
    preds = torch.cat(preds, dim=0)

    preds = preds.detach().cpu().numpy()

    preds_df = pd.DataFrame(preds)

    data_df = pd.read_csv(data_path)

    preds_df["Region"] = data_df[labelcol]

    preds_df.to_csv(output_file_path, index=False)
