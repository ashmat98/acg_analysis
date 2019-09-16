import numpy as np
from collections import defaultdict
from label_processors import label_processor_to_group, \
                             label_processor_to_label, \
                             label_processor_to_patient, \
                             get_patient_mapping


def split(data, train_portion, val_portion, seed=42):
    """
    Usage:
    ((train_beats, train_labels),
     (qval_beats, qval_labels),
     (val_beats, val_labels)) = splitted_data(...)
    """
    beats_part1, labels_part1, beats_part2, labels_part2 = data

    group_indices = defaultdict(lambda: defaultdict(list))
    for j, labels in enumerate((labels_part1, labels_part2)):
        for i, label in enumerate(labels):
            grp, pat = label["group"], label["patient"]
            group_indices[grp].setdefault(
                pat, (list(), list()))[j].append(i)
    np.random.seed(seed)
    train_ids = ([], [])
    val_ids = ([], [])
    for grp, pats in group_indices.items():
        pats = list(pats.values())
        random_ids = list(np.random.choice(range(len(pats)),
         int((train_portion + val_portion)*len(pats)), replace=False))
        pos = int(val_portion * len(pats))
        for i in random_ids[:pos]:
            val_ids[0].extend(pats[i][0])
            val_ids[1].extend(pats[i][1])
        for i in random_ids[pos:]:
            train_ids[0].extend(pats[i][0])
            train_ids[1].extend(pats[i][1])
        
    return ((beats_part1[train_ids[0]], labels_part1[train_ids[0]]),
            (beats_part2[train_ids[1]], labels_part2[train_ids[1]]),
            (beats_part1[val_ids[0]], labels_part1[val_ids[0]]))

def split_n(data, first_n_patients):
    """
    Usage:
    ((train_beats, train_labels),
     (qval_beats, qval_labels),
     (val_beats, val_labels)) = splitted_data(...)
    """
    beats_part1, labels_part1, beats_part2, labels_part2 = data

    group_indices = defaultdict(lambda: defaultdict(list))
    pos = []
    for labels in (labels_part1, labels_part2):
        patients = []
        for i, label in enumerate(labels):
            if len(patients) == first_n_patients+1:
                pos.append(i-1)
                break
            grp, pat = label["group"], label["patient"]
            if len(patients) == 0  or patients[-1] != pat:
                patients.append(pat) 
    
    return ((beats_part1[:pos[0]], labels_part1[:pos[0]]),
            (beats_part2[:pos[1]], labels_part2[:pos[1]]),
            (beats_part1[pos[0]:], labels_part1[pos[0]:]))

class Data:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) is list:
                v = np.array(v)
            setattr(self, k, v)

def label(splitted_data, processors=dict()):
    datas = []
    for bts, lbs in splitted_data:
        # substruct mean
        bts[:] = bts - bts.mean(axis=-2, keepdims=True)
        data_kwargs = {}
        for name, processor_fun in processors.items():
            data_kwargs[name] = [processor_fun(x) for x in lbs]

        datas.append(Data(
            x = bts,
            y = [label_processor_to_label(x) for x in lbs],
            g = [label_processor_to_group(x) for x in lbs],
            p = [label_processor_to_patient(x) for x in lbs],
            patient_mapping = get_patient_mapping(),
            **data_kwargs
        ))
    return datas
