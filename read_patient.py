import numpy as np 
import pandas as pd
import wfdb
import os


def read_as_beats(directory, fixed_len=800, start_hour=1, end_hour=24):
    q, patient_name = os.path.split(directory)
    _, group_name = os.path.split(q)
    
    data, samples = read_patient(directory, start_hour=1, end_hour=24)
#     data = data[..., 2:]
    samples["END"] = (samples.Time + samples.QT + 100).astype(int)
    samples["START"] = (samples.Time - 100).astype(int)
    beats, labels = [], []
    for row in samples.itertuples():
        if row.START <0:
            continue
        beat = data[row.START:row.END]
        
        if np.any(np.isnan(beat)) == True:
            continue

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

def read_csv(patient, start_hour=1, end_hour=24, drop_na=True):
    """
    Reads and concatenates csv files. 
    CSV file nams have form ****QT#.csv

    Parameters
    ----------
    patient : std, folder containing csv files.
    start_hour : int, starting hour of the record, by default 1
        this value will be substructed from the Time values in the frame 
        shuch that Time column will start from zero (nearly).
    end_hour : int, ending hour of the record, by default 24
        after substruction all values in the Time column will be 
        less then end_hour * 3600000.
    drop_na : bool, optional, Drom rows containing NaNs, by default True.
    
    Returns
    -------
    Pandas Dataframe.
    """
    frames = []
    for x in sorted(os.listdir(patient)):
        if os.path.splitext(x)[1] == ".csv" and "QT" in x:
            df = pd.read_csv(
                os.path.join(patient, x), 
                index_col=None, header=0)
            frames.append(df)
    
    frame = pd.concat(frames, axis=0, ignore_index=True)
    if drop_na is True:
        # nacols = frame.isna().all()
        # frame = frame.drop(columns=nacols[nacols==True].index).dropna()
        frame = frame[(frame.Annotation == 0) & (frame.QT.isna() == False)]
        frame = frame.reset_index(drop=True)
    
    start_hour = start_hour - 1
    ids = (start_hour*3600000<=frame.Time) & (frame.Time<end_hour*3600000)
    frame = frame[ids]
    frame.Time -= start_hour * 3600000
    frame = frame.reset_index(drop=True)

    return frame

def get_hour_from_filename(name):
    """
    Filename has form "Hour<number>RawData"
    e.g Hour10RawData.bin
    if got name of different form, returns None.
    """
     # name e.g. "Hour10RawData"
    try:
        hour = int(name.split("Hour")[1].split("RawData")[0])
    except Exception as e:
        print(e, name)
        return None
    return hour

def read_patient(directory, header=None, rate=1000, channels=None, start_hour=1, end_hour=24, 
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
        start_hour (int, optional): start hour of the recording, 1..24. Defaults to 1.
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
            hour = get_hour_from_filename(name)
            if hour is None:
                continue
            if not start_hour <= hour <= end_hour:
                continue
            bin_files.append((hour, name))

    bin_files = sorted(bin_files, key=lambda x: x[0])
    
    # allocate 
    data = np.zeros((len(bin_files)*3600*rate, 8), dtype="float" + str(dtype)); data+=1
    free_point = 0
    for hour, name in bin_files:      
        if header is not None:
            # create header file
            create_header(os.path.join(directory, name), header)
        
        # os.remove(new_header_fpath)
        rec = wfdb.rdrecord(os.path.join(directory, name), channels=channels,
                    return_res=dtype)
        values = rec.p_signal[::1000//rate]
        
        data[free_point:][:values.shape[0], :values.shape[1]] = values
        free_point += values.shape[0]
    
    
    all_values = data[:free_point]

    hours = [hour for hour, name in bin_files]
    if save_gzip is not None:
        np.savez_compressed(save_gzip, all_values)
        
    # reading CSV files
    frame = read_csv(directory, min(hours), max(hours), drop_na)
    return all_values, frame

