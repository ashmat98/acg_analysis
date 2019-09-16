from matplotlib import pyplot as plt 
from sklearn.metrics import accuracy_score, f1_score
import numpy as np 

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["figure.dpi"] = 120


def plot_patients(predictions, labels, background_label, patients, title, print_scores=True, save_prefix=None):
    labels = np.array(labels)
    patients = np.array(patients)
    background_label = np.array(background_label)
    pe = 2 # plot every 2 points
    L = len(predictions[::pe])

    plt.figure(figsize=(16, 3), dpi=200)
#     predictions_flat = predictions.reshape(-1)
    means = np.zeros_like(predictions)
    s = 0
    patient_ticks = ([], [])
    patient_level = ([], [])
    for i in range(len(predictions)):
        if i+1 == len(predictions) or patients[i+1] != patients[i]:
            means[s:i+1] = np.mean(predictions[s:i+1])
            patient_ticks[0].append((i+s)//2//pe)
            patient_ticks[1].append(patients[(i+s)//2])
            patient_level[0].append(labels[i]>0.5)
            patient_level[1].append(means[i]>0.5)
            s = i+1
    
    plt.xlim(0, L)
    
    plt.axhline(y=0.5, xmin=0, xmax=L, color="black", lw=1)
    
    plt.scatter(range(L),predictions[::pe], s=0.1)

    plt.scatter(range(L), means[::pe], color="red",s=0.1)
    ax = plt.gca()
    ax.pcolorfast((0, L), (-0.05,1.1), background_label[::pe][None], alpha=0.4)
    ax.pcolorfast((0, L), (-0.1, -0.05), patients[::pe][None], alpha=0.4, cmap="jet")
    plt.xticks(*patient_ticks, rotation=90)
    if print_scores:
        pat_text = "Patient level: %0.2f%% %0.4f" % (
            100*accuracy_score(*patient_level),
            f1_score(*patient_level))
        beat_text = "Beat level:    %0.2f%% %0.4f" % (
            100*accuracy_score(labels>0.5, predictions>0.5),
            f1_score(labels>0.5, predictions>0.5))
        title += "   " + pat_text + "   " + beat_text
        plt.title(title)
    if save_prefix is not None:
        plt.savefig(save_prefix + " " + title+".png")
    plt.show() 