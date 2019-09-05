import pickle
import numpy as np

def sort_2d(arr, scores):
    index_sorted = (scores + np.arange(arr.shape[0])[..., None]*arr.shape[1]).reshape(-1)
    arr = arr.reshape(-1, *arr.shape[2:])[index_sorted].reshape(arr.shape)
    return arr

def load_array(paths, sort=False, flatten=True, sortby="Time"):
    """
    Loads arrays saved by generator function.
    
    Args:
        paths (list<str>): list of paths
        sort (bool, optional): Sort beats by "sort_by" argument. Defaults to False.
        flatten (bool, optional): flatten patient dimension. Defaults to True.
        sortby (str, optional): Defaults to "Time".
    
    Returns:
        beats, labels
    """
    beats, labels = [], []
    for path in paths:
        a,b = pickle.load(open(path, "rb"))    
        beats.append(a.reshape(-1, 1000, 800, 8))
        labels.append(np.array(b).reshape(-1, 1000))

    beats = np.concatenate(beats, axis=1)
    labels = np.concatenate(labels, axis=1)
    if sort is True:
        scores = np.vectorize(lambda x:x["anotations"][sortby])(labels).argsort(axis=1)
        beats = sort_2d(beats, scores)
        labels = sort_2d(labels, scores)
    
    if flatten is True:
        beats, labels = beats.reshape((-1,800,8)), labels.reshape((-1))
    
    return beats, labels