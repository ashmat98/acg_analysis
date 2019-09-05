import numpy as np 
# from utils import sort_2d

def low_variance_filter(beats, label):
    """
    Drops beats with high variance.
    Args:
        beats (ndarray): (patient x sample x time x channel) array.
        labels (ndarray<dict>): 1d array containing information about patients.
    Returns:
        beats, labels : filtered beats and labels
    """
    channels = beats.shape[-1]

    # differentiate
    diff=10
    beats_diff = beats[:,:,diff:,:] - beats[:,:,:-diff,:]
    
    # calculate statistics
    beats_diff_std = beats_diff[:,:,170:].std(axis=2)
    beats_std = beats[:,:,170:].std(axis=2)

    # score beats
    score = (
        np.sum(beats_diff_std / beats_diff_std.reshape(-1,channels).mean(axis=0), axis=-1) + 
        np.sum(beats_std / beats_std.reshape(-1,channels).mean(axis=0), axis=-1))


    return score.argsort(axis=1)
    # sort by score
    # beats = sort_2d(beats, score.argsort(axis=1))
    # labels = sort_2d(labels, score.argsort(axis=1))

    # return beats[:, :keep], labels[:, :keep]
