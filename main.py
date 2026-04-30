import argparse
import os

from dataset_loader import DatasetLoader
from canary_method import CanaryDetector
from experiment import ExperimentRunner
from visualize import Visualizer
from diagnose import DatasetDiagnoser
from attack_simulator import AttackSimulator
from kappa_method import KappaCalculator
from combined_detector import CombinedDetector

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", default=None)
    parser.add_argument("--poison_ratio", type=float, default=0.0)
    parser.add_argument("--use_kappa", action="store_true")
    parser.add_argument("--experiment", action="store_true")

    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    loader = DatasetLoader()
    X, y, df = loader.load(args.data, args.target)

    target = args.target if args.target else df.columns[-1]

    print("\nClass distribution:")
    print(df[target].value_counts())

    canary = CanaryDetector()
    canary_idx, true_labels = canary.inject_canaries(X, y)

    attack = AttackSimulator()
    y_observed, _ = attack.poison_dataset(y, args.poison_ratio)

    kappa_calc = KappaCalculator()

    annotator2 = None
    if args.use_kappa:
        _, annotator2 = kappa_calc.simulate_annotators(
            y,
            args.poison_ratio,
            args.poison_ratio
        )

    diagnoser = DatasetDiagnoser(
        canary,
        kappa_calc,
        CombinedDetector()
    )

    diagnosis = diagnoser.diagnose(
        y_observed,
        canary_idx,
        true_labels,
        annotator2
    )

    print("\n=== DATASET DIAGNOSIS ===")
    print(f"P_canary: {diagnosis['p_canary']:.3f}")

    if diagnosis["p_kappa"] is not None:
        print(f"P_kappa: {diagnosis['p_kappa']:.3f}")

    print(f"P_total: {diagnosis['p_total']:.3f}")

    if args.experiment:
        runner = ExperimentRunner(
            attack=attack,
            canary=canary,
            kappa_calc=kappa_calc,
            combiner=CombinedDetector()
        )

        results = runner.run(X, y, canary_idx, true_labels)

        print("\n=== RESULTS ===")
        print(results)

        results.to_csv("results/results.csv", index=False)

        visualizer = Visualizer()
        visualizer.plot_results(results)

if __name__ == "__main__":
    main()