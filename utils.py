import numpy as np
import pandas as pd
import scipy.io as sio
import mat73
import nltk


def get_train_test(data, label, train=0.7):
    data=np.array(data)
    label=np.array(label)

    assert len(data) == len(label)
    n_trials=len(data)
    indices = np.arange(n_trials)
    pack=list(zip(indices,data,label))
    np.random.shuffle(pack)
    indices_shuffled,data_shuffled,label_shuffled=zip(*pack)
    return (np.array(data_shuffled)[:int(n_trials*train),:],
            np.array(data_shuffled)[int(n_trials*train):,:],
            np.array(label_shuffled)[:int(n_trials*train)],
            np.array(label_shuffled)[int(n_trials*train):],
            np.array(indices_shuffled)[:int(n_trials * train)],
            np.array(indices_shuffled)[int(n_trials * train):],
            )

def decoding(packed):
    decoder, X, labels_to_use =packed
    accuracy=[]
    for n_bin in range(len(X)):
        X_train, X_test, y_train, y_test =get_train_test(X[n_bin],labels_to_use,train=0.9) ### TODO: this train/test split may need to be in the pack
        if X_train.shape[1] !=0:
            decoder.fit(X_train, y_train)
            y_predict=decoder.predict(X_test)
            correct=np.sum(y_predict==y_test)
            accuracy.append(correct/len(y_predict))
        else:
            accuracy.append(np.NaN)
    return accuracy

def reformat(data, bins_per_feature):
    reformatted_data=[]
    for i in range(data.shape[1]):
        reformatted=data[:,i-np.minimum(i, bins_per_feature-1):i+1,:]
        reformatted_data.append(reformatted.reshape(data.shape[0],-1))
    return reformatted_data


def get_raw(date, band):
    data_path= f'../Bolu_IFG/processed_data/{band}_power/{date}_all_blocks.mat'
    try:
        raw_data_ = pd.DataFrame(sio.loadmat(data_path)['all_data'])
    except:
        raw_data_ = pd.DataFrame(mat73.loadmat(data_path)['all_data'])

    return raw_data_



def data_cleaning(data:pd.DataFrame,channel_cleaning_threshold=1,trial_cleaning_threshold=1.5):
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

    data=data.drop(data.index[reject_trials])


    return data,reject_channels,reject_trials,channel_std,trial_std

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



if __name__ == '__main__':
    # Download the CMU Pronouncing Dictionary
    nltk.download('cmudict')

    # Load the CMU Pronouncing Dictionary
    cmudict = nltk.corpus.cmudict.dict()

    word = "date"
    embeddings = get_phoneme_embeddings(word)
    print("Embeddings for 'date':")
    for embedding in embeddings:
        print(embedding)

    word = "bait"
    embeddings = get_phoneme_embeddings(word)
    print("\nEmbeddings for 'bait':")
    for embedding in embeddings:
        print(embedding)











