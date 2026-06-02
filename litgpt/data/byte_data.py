from torch.utils.data import Dataset
from litgpt.data import DataModule


class ByteStreamDataset(Dataset):
    """Plain next-byte pretraining over byte streams."""


class ByteFIMDataset(Dataset):
    """Bridge prediction over prefix/orphan/missing spans."""


class ByteDataModule(DataModule):
    """Experiment wrapper selecting byte-domain task."""
