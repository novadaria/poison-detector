# -- визуализация результатов
 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class Visualizer:

    def __init__(self, output_dir="results"):
        self.output_dir = output_dir

    def plot_results(self, df):

        plt.figure()
        plt.plot(df["poison_level"], df["p_total"], label="P_total")
        plt.plot(df["poison_level"], df["p_canary"], label="P_canary")
        plt.plot(df["poison_level"], df["p_kappa"], label="P_kappa")
        plt.legend()
        plt.title("Poisoning detection probabilities")
        plt.savefig(f"{self.output_dir}/probabilities.png")
        plt.close()

        plt.figure()
        plt.plot(df["poison_level"], df["accuracy"])
        plt.title("Model accuracy vs poisoning level")
        plt.savefig(f"{self.output_dir}/accuracy.png")
        plt.close()

        plt.figure()
        plt.plot(df["poison_level"], df["kappa"])
        plt.title("Cohen Kappa vs poisoning level")
        plt.savefig(f"{self.output_dir}/kappa.png")
        plt.close()

        y_true = df["poison_level"] > 0
        y_scores = df["p_total"]

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.legend()
        plt.title("ROC Curve")
        plt.savefig(f"{self.output_dir}/roc_curve.png")
        plt.close()