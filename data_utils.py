import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
import ipdb
import random
from torch.utils.data import DataLoader




def load_dictionary(file_name, delim=' '):
    """
    Load text file as dictionary:
        1st column are keys, the rest columns are values
    """
    from collections import defaultdict
    d = defaultdict(list)
    with open(file_name) as f:
        for line in f:
            sl = line.split(delim, 1)
            clean_list = list(map(str.strip, sl))
            d[clean_list[0]] = clean_list[1:]
    return d


def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _ = line.strip().split(" ")
            # key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def gen_cyber_list(meta_file, feat_file):
    """
    Filter feature files and meta, full path to file
    meta_file: path to dir with metafiles fold#_{train,validation,evaluation}.tsv
    feat_file: total utterance list (wav.scp format)
    return: fids ['file_id',...],
            labs [1, 0, ...],
            paths ['/full/path/file.wav', ...]
    """

    scp_feats = load_dictionary(feat_file)
    fold_kv = pd.read_csv(meta_file, sep='\t', header=0)
    fold_kv = dict(fold_kv.values)

    # intersect file_ids from fold_list and file_ids from scp data_file
    dif = set(fold_kv) - set(scp_feats)
    

    if len(dif):
        # remove file ids which do not have features
        sample_keys = list(set(fold_kv) - dif)
        print('[WARNING] no features for files: %d / %d' % (len(dif), len(fold_kv)))
        print('Removing file ids without features...')
    else:
        sample_keys = list(fold_kv)
    sample_labs = [fold_kv[k] for k in sample_keys]
    sample_labs = [1 if l == "bonafide" else 0 for l in sample_labs]
    sample_paths = [scp_feats[k] for k in sample_keys]
    assert len(sample_paths) == len(sample_keys) == len(sample_labs)
    print(f'[INFO] sample from {len(sample_keys)} files')
    return sample_keys, sample_labs, sample_paths


def gen_asvspoof21_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, *_ = line.strip().split(" ")
            # key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def partial_cut(x, partial_length):

    x_len = x.shape[0]
    if x_len >= partial_length:
        return x[:partial_length]
    # need to pad
    num_repeats = int(partial_length / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :partial_length][0]

    return padded_x



def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len > max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key


class CyberDataset(Dataset_ASVspoof2019_train):
    def __init__(self, list_ids, labels, file_paths):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        super(CyberDataset, self).__init__(list_ids, labels, base_dir='./')
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        self.file_paths = file_paths

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        path = self.file_paths[index][0].replace('"', '')
        y = self.labels[index]


        X, _ = sf.read(path)
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        
        return x_inp, y, utt_id



class CyberEvalDataset(CyberDataset):
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        path = self.file_paths[index][0].replace('"', '')
        y = self.labels[index]

        X, _ = sf.read(path)
        # mean subtraction
        #mn = X.min()
        #mx = X.max()
        #X = (X - mn) / (mx - mn)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, y, utt_id


class CyberPatialDataset(object): #partial length is # of samples (fs = 16kHz)
    def __init__(self, list_ids, labels, file_paths):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_ids = list_ids
        self.file_paths = file_paths
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        utt_id = self.list_ids[index]
        path = self.file_paths[index]

        partial_length = random.choice([1, 2, 4])

        X, _ = sf.read(path)
        if len(X.shape) == 2:
            X = X.mean(axis=1)
        # X_pad = partial_cut(X, partial_length)
        # x_inp = Tensor(X_pad)
        x_inp = Tensor(X)
        return x_inp, utt_id



class RandomLengthDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        super(RandomLengthDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.custom_collate)

    def custom_collate(self, batch):
        # Choose a random length from [1, 2, 4]
        chosen_length = random.choice([1, 2, 4])

        # Create a new batch with the chosen length
        new_batch = []

        for item in batch:

            X_pad = partial_cut(item, chosen_length)
            new_batch.append(X_pad)

        return torch.stack(new_batch)
    

# do partial here
class Slide_TestingDataset(object):

    def __init__(self, list_ids, labels, file_paths):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_ids = list_ids
        self.file_paths = file_paths
        self.labels = labels
        self.window_duration = 64600  # take ~4 sec audio (64600 samples)

    def __getitem__(self, index):
        utt_id = self.list_ids[index]
        path = self.file_paths[index][0].replace('"', '')
        y = self.labels[index]

        X, _ = sf.read(path)
        if len(X.shape) == 2:
            X = X.mean(axis=1)
        utterance = X
        num_segments = len(utterance) // self.window_duration
        segments = []

        for i in range(num_segments):
            start = i * self.window_duration
            end = start + self.window_duration
            segment = utterance[start:end]
            segments.append(segment)


        # Handle remaining part of the utterance if it's shorter than window_duration
        remaining_duration = len(utterance) % self.window_duration
        if remaining_duration > 0:
            start = num_segments * self.window_duration
            end = start + remaining_duration
            segment = utterance[start:end]
            segments.append(segment)

        segments = Tensor(segments)

        return segments, start, end, y, utt_id  # Return start and end times for each segment, for each utterance

    def __len__(self):
        return len(self.list_ids)


