# dataset_class.py
import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, random_split, Subset
from sklearn.model_selection import train_test_split


def infer_num_classes(ds):
    """Infer the number of classes for ImageFolder/OdirDataset/Subset."""
    if hasattr(ds, "num_classes"):
        return int(ds.num_classes)
    if hasattr(ds, "classes"):
        return len(ds.classes)
    if isinstance(ds, Subset):
        return infer_num_classes(ds.dataset)
    x, y = ds[0]
    if torch.is_tensor(y):
        return int(y.numel()) if y.ndim > 0 else int(max(int(y), 0) + 1)
    raise ValueError("Cannot infer num_classes from dataset")


def set_transform_for_subset(subset: Subset, transform):
    """Set transform for the underlying dataset of a Subset."""
    if isinstance(subset, Subset):
        subset.dataset.transform = transform


class Messidor2VectorAsImage(Dataset):
    """
    Reshape each 688-dim vector into a grayscale image (H×W),
    default H=16, W=43 (16*43=688).
    Returns (PIL.Image or Tensor, label)
    """

    def __init__(self, mat_path, H=16, W=43, transform=None, to_rgb=True, per_sample_minmax=True):
        self.transform = transform
        self.H, self.W = H, W
        self.to_rgb = to_rgb
        self.per_sample_minmax = per_sample_minmax

        import scipy.io as sio
        mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        data = mat["data"]

        self.samples = []
        for i in range(data.shape[0]):
            feats = data[i, 0]
            lab = int(np.array(data[i, 1]).ravel()[0])
            for k in range(feats.shape[0]):
                vec = feats[k].astype(np.float32)

                if self.per_sample_minmax:
                    vmin, vmax = float(vec.min()), float(vec.max())
                    if vmax > vmin:
                        vec = (vec - vmin) / (vmax - vmin)
                else:
                    vec = (vec - vec.mean()) / (vec.std() + 1e-6)
                    vec = (vec - vec.min()) / (vec.max() - vec.min() + 1e-6)

                img = (vec * 255.0).reshape(self.H, self.W).clip(0, 255).astype(np.uint8)
                self.samples.append((img, lab))

        self.classes = [0, 1]
        self.num_classes = 2

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        arr, label = self.samples[idx]
        img = Image.fromarray(arr, mode="L")
        if self.to_rgb:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def load_messidor2_as_image(mat_path, transform_train=None, transform_val=None,
                            val_ratio=0.2, seed=42):
    """
    Stratified split for Messidor2VectorAsImage:
    - keeps class distribution consistent
    - uses different transforms for train/val
    - attaches classes to Subset for downstream usage
    """
    full_plain = Messidor2VectorAsImage(mat_path, transform=None)

    y = np.array([lab for _, lab in full_plain.samples], dtype=np.int64)

    idx_all = np.arange(len(full_plain))
    train_idx, val_idx = train_test_split(
        idx_all,
        test_size=val_ratio,
        random_state=seed,
        stratify=y
    )

    ds_train_full = Messidor2VectorAsImage(mat_path, transform=transform_train)
    ds_val_full = Messidor2VectorAsImage(mat_path, transform=transform_val)

    train_set = Subset(ds_train_full, train_idx.tolist())
    val_set = Subset(ds_val_full, val_idx.tolist())

    train_set.classes = full_plain.classes
    val_set.classes = full_plain.classes
    train_set.num_classes = getattr(full_plain, "num_classes", len(full_plain.classes))
    val_set.num_classes = getattr(full_plain, "num_classes", len(full_plain.classes))

    return train_set, val_set


class MedMnistDataset(Dataset):
    """
    Compatible with medmnist exported *.npz:
    - {split}_images: (N, H, W) or (N, H, W, 3)
    - {split}_labels: (N, 1) or (N,); may also be one-hot
    """

    def __init__(self, npz_path, split, transform=None, as_rgb=True, labels_are_multilabel=False):
        data = np.load(npz_path)
        self.X = data[f"{split}_images"]
        self.y = data[f"{split}_labels"]
        self.transform = transform
        self.as_rgb = as_rgb
        self.labels_are_multilabel = labels_are_multilabel

        self.y = np.array(self.y)
        if self.y.ndim > 1 and self.y.shape[-1] == 1:
            self.y = self.y.squeeze(-1)

        if labels_are_multilabel:
            self.num_classes = int(self.y.shape[-1])
            self.classes = list(range(self.num_classes))
        else:
            self.num_classes = int(np.max(self.y)) + 1
            self.classes = list(range(self.num_classes))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        img = self.X[idx]
        if img.ndim == 2:
            img = Image.fromarray(img)
            if self.as_rgb:
                img = img.convert("RGB")
        else:
            img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)

        if self.labels_are_multilabel:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
        else:
            label = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return img, label


class OdirDataset(Dataset):
    """
    By default reads Training Images with labels
    (8-dim multi-label: N,D,G,C,A,H,M,O).
    Supports downstream patient-wise splitting.
    """

    LABEL_COLS = ["N", "D", "G", "C", "A", "H", "M", "O"]

    def __init__(self, root_dir, img_dir="Training Images", excel_file="data.xlsx",
                 use_eyes=("Left-Fundus", "Right-Fundus"),
                 transform=None, as_rgb=True, skip_missing=True, return_path=False):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, img_dir)
        self.excel_path = os.path.join(root_dir, excel_file)
        self.use_eyes = use_eyes
        self.transform = transform
        self.as_rgb = as_rgb
        self.skip_missing = skip_missing
        self.return_path = return_path

        df = pd.read_excel(self.excel_path)
        df.columns = [str(c).strip() for c in df.columns]

        samples = []
        for _, row in df.iterrows():
            labels = row[self.LABEL_COLS].values.astype("float32")
            for col in self.use_eyes:
                img_name = str(row[col]).strip()
                img_path = os.path.join(self.img_dir, img_name)
                if not os.path.exists(img_path):
                    if self.skip_missing:
                        continue
                    else:
                        raise FileNotFoundError(img_path)
                item = (img_path, torch.tensor(labels, dtype=torch.float32))
                if self.return_path:
                    item = (img_path, torch.tensor(labels, dtype=torch.float32), img_path)
                samples.append(item)

        self.samples = samples
        self.classes = self.LABEL_COLS[:]
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        if self.return_path:
            img_path, label, path = item
        else:
            img_path, label = item
        img = Image.open(img_path)
        if self.as_rgb and img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return (img, label, img_path) if self.return_path else (img, label)


def load_odir5k(root_dir, transform_train=None, transform_val=None,
                val_ratio=0.2, seed=42, patient_wise=True, id_col="ID"):
    """
    Load ODIR-5K and split into train/val.
    - patient_wise=True: split by patient ID
    - patient_wise=False: random split by sample
    """
    import math
    df = pd.read_excel(os.path.join(root_dir, "data.xlsx"))
    df.columns = [str(c).strip() for c in df.columns]

    if patient_wise and id_col in df.columns:
        ids = df[id_col].astype(str).unique().tolist()
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(ids), generator=g).tolist()
        split = int(math.floor(len(ids) * (1 - val_ratio)))
        train_ids = set([ids[i] for i in perm[:split]])
        val_ids = set([ids[i] for i in perm[split:]])

        def _filter_by_ids(_ids):
            mask = df[id_col].astype(str).isin(_ids)
            sub = df.loc[mask].reset_index(drop=True)
            tmp_path = os.path.join(root_dir, "_temp_split.xlsx")
            sub.to_excel(tmp_path, index=False)
            return tmp_path

        train_excel = _filter_by_ids(train_ids)
        val_excel = _filter_by_ids(val_ids)

        train_set = OdirDataset(
            root_dir,
            img_dir="Training Images",
            excel_file=os.path.basename(train_excel),
            transform=transform_train
        )
        val_set = OdirDataset(
            root_dir,
            img_dir="Training Images",
            excel_file=os.path.basename(val_excel),
            transform=transform_val
        )
        try:
            os.remove(train_excel)
            os.remove(val_excel)
        except Exception:
            pass
        return train_set, val_set

    else:
        full = OdirDataset(root_dir, img_dir="Training Images", transform=transform_train)
        val_len = int(len(full) * val_ratio)
        train_len = len(full) - val_len
        train_set, val_set = random_split(
            full,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(seed)
        )
        set_transform_for_subset(val_set, transform_val)
        return train_set, val_set


def multilabel_pos_weight(dataset_or_loader):
    """
    Estimate positive class weights for multi-label data.
    Suitable for BCEWithLogitsLoss(pos_weight=...).
    """
    if isinstance(dataset_or_loader, Subset):
        ds = dataset_or_loader.dataset
    else:
        ds = dataset_or_loader

    counts = None
    for _, y in ds:
        y = y.float().view(-1)
        if counts is None:
            counts = torch.zeros_like(y)
        counts += y
    total = len(ds)
    pos = counts.clamp(min=1)
    neg = (total - counts).clamp(min=1)
    return neg / pos