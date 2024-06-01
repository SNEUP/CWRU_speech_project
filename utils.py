import numpy as np
import pandas as pd
import scipy.io as sio
import mat73
import nltk
import torch
from nltk.metrics import aline
from gruut import sentences
import itertools
from torch import nn
from torch.utils.data import Dataset

nltk.download('cmudict')
cmudict = nltk.corpus.cmudict.dict()


class CustomDataset:
    def __init__(self, neural_data, confusion_matrix):
        self.neural_data = neural_data  # this is an n_samples x n_features matrix
        self.confusion_matrix = confusion_matrix  # this is a n_samples x n_samples matrix

    def __len__(self):
        return len(self.neural_data)

    def __call__(self, idx_list):
        x, y = np.meshgrid(idx_list, idx_list)
        nest_idx = np.column_stack((x.flatten(), y.flatten()))
        sub_matrix = self.confusion_matrix[nest_idx[:, 0], nest_idx[:, 1]]
        return self.neural_data[idx_list, :], sub_matrix.reshape(len(idx_list), len(idx_list))


class CustomDataLoader:
    def __init__(self, dataset: CustomDataset, batch_size=1, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.datalen = len(dataset)
        self._index = np.arange(self.datalen)
        if self.shuffle:
            np.random.shuffle(self._index)
        self.count = 0

    def __len__(self):
        return self.datalen//self.batch_size+int(self.datalen%self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < self.datalen:
            next_batch_idx = np.arange(self.count, np.minimum(self.batch_size + self.count, self.datalen))
            idx_list = self._index[next_batch_idx]
            self.count += self.batch_size
            return self.dataset(idx_list)
        else:
            raise StopIteration


def convert_to_ipa(x):
    x_phoneme = next(sentences(x, lang='en-us')).words[0].phonemes
    # break the phonemes if they contain a combined IPAs
    x_phoneme = [char for item in x_phoneme for char in item]
    # remove unnecessary string
    x_phoneme = [item for item in x_phoneme if item != 'ˈ' and item != '͡']
    return ''.join(x_phoneme)


def select_submatrix(confusion_matrix, indices):
    indices = list(indices) if not isinstance(indices, list) else indices
    x, y = np.meshgrid(indices, indices)
    nest_idx = np.column_stack((x.flatten(), y.flatten()))
    sub_matrix = confusion_matrix[nest_idx[:, 0], nest_idx[:, 1]]
    return sub_matrix.reshape(len(indices), len(indices))


def get_train_test(data, label, train=0.7):
    data = np.array(data)
    label = np.array(label)

    assert len(data) == len(label)
    n_trials = len(data)
    indices = np.arange(n_trials)
    pack = list(zip(indices, data, label))
    np.random.shuffle(pack)
    indices_shuffled, data_shuffled, label_shuffled = zip(*pack)
    return (np.array(data_shuffled)[:int(n_trials * train), :],
            np.array(data_shuffled)[int(n_trials * train):, :],
            np.array(label_shuffled)[:int(n_trials * train)],
            np.array(label_shuffled)[int(n_trials * train):],
            np.array(indices_shuffled)[:int(n_trials * train)],
            np.array(indices_shuffled)[int(n_trials * train):],
            )


def decoding(packed):
    decoder, X, labels_to_use = packed
    accuracy = []
    for n_bin in range(len(X)):
        X_train, X_test, y_train, y_test = get_train_test(X[n_bin], labels_to_use,
                                                          train=0.9)  ### TODO: this train/test split may need to be in the pack
        if X_train.shape[1] != 0:
            decoder.fit(X_train, y_train)
            y_predict = decoder.predict(X_test)
            correct = np.sum(y_predict == y_test)
            accuracy.append(correct / len(y_predict))
        else:
            accuracy.append(np.NaN)
    return accuracy


def reformat(data, bins_per_feature):
    reformatted_data = []
    for i in range(data.shape[1]):
        reformatted = data[:, i - np.minimum(i, bins_per_feature - 1):i + 1, :]
        reformatted_data.append(reformatted.reshape(data.shape[0], -1))
    return reformatted_data


def get_raw(date, band):
    data_path = f'../Bolu_IFG/processed_data/{band}_power/{date}_all_blocks.mat'
    try:
        raw_data_ = pd.DataFrame(sio.loadmat(data_path)['all_data'])
    except:
        raw_data_ = pd.DataFrame(mat73.loadmat(data_path)['all_data'])

    return raw_data_


def data_cleaning(data: pd.DataFrame, channel_cleaning_threshold=1, trial_cleaning_threshold=1.5):
    # channel cleaning comes first before trial cleaning
    # threshold get multiplied to the standard deviation
    # here we assume the standard deviation of each pre-zscored channel is stored in the raw data
    all_data = data[1].to_list()
    ## channel cleaning
    channel_std = data[0].to_list()[0]
    channel_std_std = channel_std.std()
    reject_channels = np.where(channel_std > channel_cleaning_threshold * channel_std_std)[0]
    channel_clean_data = [np.delete(a, reject_channels, axis=0) for a in all_data]
    ## trial cleaning
    trial_std = np.array([t.mean(axis=0).std() for t in channel_clean_data])
    reject_trials = np.where(trial_std > trial_cleaning_threshold * trial_std.std())[0]

    data = data.drop(data.index[reject_trials])

    return data, reject_channels, reject_trials, channel_std, trial_std


def get_phoneme_embeddings(word):
    # Convert the word to lowercase
    word = word.lower()

    # Check if the word exists in the CMUdict
    if word in cmudict:
        # Get all the phoneme sequences for the word
        phoneme_sequences = cmudict[word]

        embeddings = []
        for phonemes in phoneme_sequences:
            # Create a mapping of unique phonemes to integer indices
            unique_phonemes = sorted(set(phonemes))
            phoneme_to_index = {phoneme: index for index, phoneme in enumerate(unique_phonemes)}

            # Create a one-hot encoding for each phoneme
            num_phonemes = len(unique_phonemes)
            embedding_dim = num_phonemes
            embedding = np.zeros((len(phonemes), embedding_dim))

            for i, phoneme in enumerate(phonemes):
                embedding[i, phoneme_to_index[phoneme]] = 1

            embeddings.append(embedding)

        return embeddings
    else:
        print(f"Word '{word}' not found in the CMU Pronouncing Dictionary.")
        return None


class Sphere(nn.Module):
    '''
    This is used as a parametrization tool to turn a matrix into unit matrix
    '''

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x / x.norm(dim=self.dim, keepdim=True)

    def right_inverse(self, x):
        return x / x.norm(dim=self.dim, keepdim=True)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import numpy as np

    fake_data = np.random.randn(100, 30)
    fake_confusion_matrix = np.random.randn(100, 100)
    dataset = CustomDataset(fake_data, fake_confusion_matrix)
    dataloader = CustomDataLoader(dataset, batch_size=10, shuffle=True)
    print(next(dataloader))
