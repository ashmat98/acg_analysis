

_PATIENTS = {}
def label_processor_to_patient(label):
    patient = label["patient"]
    if not (patient in _PATIENTS):
        _PATIENTS[patient] = len(_PATIENTS)
    return _PATIENTS[patient]

def get_patient_mapping():
    return _PATIENTS.copy()

def label_processor_to_group(label):
    G = {"ctrls":0, "t1posajneg":1, "t1negajpos":2, "ICC":3}
    return G[label["group"]]

def label_processor_to_label(label):
    G = {"ctrls":0, "t1posajneg":1, "t1negajpos":1, "ICC":1}
    return G[label["group"]]