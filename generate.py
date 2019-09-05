import numpt as np
from read_patient import read_patient, create_header, read_csv


def generate(data_json,samples_per_patient, fixed_len=800):
    beats = []
    labels = []
    for group_name, patients in data_json.items():
        for patient_name, records in tqdm(patients.items()):
            data, anotations = read_patient(os.path.join(group_name, patient_name))
            samples = anotations.sample(n=samples_per_patient)

            samples["END"] = (samples.Time + samples.QT + 100).astype(int)
            samples["START"] = (samples.Time - 100).astype(int)

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
    return beats, labels