import numpy as np 
import pandas as pd
import wfdb
import os


def read_as_beats(directory, fixed_len=800):
    q, patient_name = os.path.split(directory)
    _, group_name = os.path.split(q)
    
    data, samples = read_patient(directory)
    data = data[..., 2:]
    samples["END"] = (samples.Time + samples.QT + 100).astype(int)
    samples["START"] = (samples.Time - 100).astype(int)
    beats, labels = [], []
    for row in samples.itertuples():
        beat = data[row.START:row.END]
        anot_dict = dict(row._asdict())
        anot_dict.pop("START")
        anot_dict.pop("END")
        anot_dict["original_size"] = len(beat)
        # pad beat to the fixed length
        beat = np.pad(beat[:fixed_len], 
                      pad_width=((0,max(0, fixed_len - beat.shape[0])), (0,0)), 
                      mode="edge")

        beats.append(beat)
        labels.append({"anotations": anot_dict,
                       "group": group_name,
                       "patient": patient_name})
    return np.array(beats), np.array(labels)

def create_header(record, header):
    directory, name = os.path.split(os.path.splitext(record)[0])
    
    header = open(header,"r").read()
    header_default_name = header.split()[0]
    new_header_fpath = os.path.join(directory, name+".hea")
    
    if not os.path.exists(new_header_fpath):
        with open(new_header_fpath, "w") as new_header:
            new_header.write(header.replace(header_default_name, name))

def read_csv(patient, drop_na=True):
    frames = []
    for x in sorted(os.listdir(patient)):
        if os.path.splitext(x)[1] == ".csv" and "QT" in x:
            df = pd.read_csv(
                os.path.join(patient, x), 
                index_col=None, header=0)
            frames.append(df)
    
    frame = pd.concat(frames, axis=0, ignore_index=True)
    if drop_na is True:
        nacols = frame.isna().all()
        frame = frame.drop(columns=nacols[nacols==True].index).dropna()
        frame = frame.reset_index(drop=True)
    return frame

def read_patient(directory, header=None, rate=1000, channels=None, start_hour=0, end_hour=24, 
                 save_gzip=None, dtype=32, drop_na=True):
    """ 
    Reads patients ECG data from the binnary files.
    
    Args:
        directory (str): path to the folder of the patient files 
        header (str): header file name.
        rate (int, optional): points per second, should be divisible to 1000. 
            Defaults to 1000.
        channels (list of int or int optional): returns specific channels. 
            Give name or index of the channel. Defaults to None, 
            i.e. returns al channels.
        start_hour (int, optional): start hour of the recording, 1..24. Defaults to 0.
        end_hour (int, optional): end hour inclusive, 1..24. Defaults to 24.
        save_gzip (str, optional): save data as gzip file int the specified directory,
            Defaults to None, not to save
        dtype (int, optional): Could be 8, 16, 32 or 64, Defaults to 32.
    """
    
    
    if channels is int:
        channels = [channels]
    bin_files = []
    for file in os.listdir(directory):
        name, ext = os.path.splitext(file)
        if ext == ".bin":
            bin_files.append(name)


    data = []
    for name in bin_files:
        
        # name e.g. "Hour10RawData"
        hour = 0
        if len(bin_files) > 1:
            try:
                hour = int(name.split("Hour")[1].split("RawData")[0])
            except Exception as e:
                print(e, name)
                continue
    
            if not start_hour <= hour <= end_hour:
                continue
        
        if header is not None:
            # create header file
            create_header(os.path.join(directory, name), header)
        
        # os.remove(new_header_fpath)
        rec = wfdb.rdrecord(os.path.join(directory, name), channels=channels,
                    return_res=dtype)
        values = rec.p_signal[::1000//rate]
        
        data.append((hour, values))
    
    data = sorted(data, key=lambda x: x[0])
    
    all_values = np.concatenate(
        [values for hour, values in data], axis=0)
    if save_gzip is not None:
        np.savez_compressed(save_gzip, all_values)
        
    # reading CSV files
    frame = read_csv(directory, drop_na)

    return all_values, frame

