import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.ticker as mticker

def bound(c: int) -> float:
    if c <= 1:
        return np.nan
    return np.log(1 + ((np.e ** 2) / (c - 1)))

def plot_nscl_dcl_bounds(batch_size: int, class_count: int = 10, output_dir: str = ".", both : bool =  True):
    """
    Plots NSCL, DCL, and theoretical upper bound for each corruption mode.
    Uses batch_size to dynamically build file paths.
    """
    # Define base path
    base_path = f"/scratch/user/aaryabookseller/Research/understanding-ssl/configs/mixed/results/loss_data/b1024/third_run"

    # Dynamically generate file paths
    corruption_modes = {
        "Clean": {
            "nscl": os.path.join(base_path, "losses_nscl_clean.csv"),
            "dcl":  os.path.join(base_path, "losses_dcl_clean.csv")
        },
        "50% Random": {
            "nscl": os.path.join(base_path, "losses_nscl_half_random.csv"),
            "dcl":  os.path.join(base_path, "losses_dcl_half_random.csv")
        },
        "100% Random": {
            "nscl": os.path.join(base_path, "losses_nscl_fully_random.csv"),
            "dcl":  os.path.join(base_path, "losses_dcl_fully_random.csv")
        }
    }

    for mode, paths in corruption_modes.items():
        df_nscl = pd.read_csv(paths["nscl"])
        df_dcl = pd.read_csv(paths["dcl"])
        
        plt.figure(figsize=(12, 8))
        
        #if we are plotting figure 2 (bottom), nscl loss using nscl and dcl weights
        if both:
            # This is NSCL loss using DCL weights
            plt.plot(df_dcl["epoch"], df_dcl["nscl_loss"], color="red", label="NSCL Loss on DCL Weights", marker="o", linewidth=2)
            
            # This is NSCL loss using NSCL weights
            plt.plot(df_nscl["epoch"], df_nscl["nscl_loss"], color="blue", label="NSCL Loss on NSCL Weights", marker="s", linewidth=2)

            # Test loss versions
            plt.plot(df_dcl["epoch"], df_dcl["nscl_loss_test"], color="red", linestyle="--", marker="o", linewidth=2)
            plt.plot(df_nscl["epoch"], df_nscl["nscl_loss_test"], color="blue", linestyle="--", marker="s", linewidth=2)

            
        #if we are plotting figure 2 (top), nscl loss, dcl loss, and, bound using dcl weights
        else:
           # NSCL loss using DCL weights
           plt.plot(df_dcl["epoch"], df_dcl["nscl_loss"], color="blue", label="NSCL Loss on DCL Weights", marker="s", linewidth=2)
           
           # DCL loss using DCL weights
           plt.plot(df_dcl["epoch"], df_dcl["dcl_loss"], color="red", label="DCL Loss on DCL Weights", marker="o", linewidth=2)
           
           # Test losses
           plt.plot(df_dcl["epoch"], df_dcl["nscl_loss_test"], color="blue", linestyle="--", marker="s", linewidth=2)
           plt.plot(df_dcl["epoch"], df_dcl["dcl_loss_test"], color="red", linestyle="--", marker="o", linewidth=2)
           
           # Upper bounds
           df_dcl["upper_bound"] = df_dcl["epoch"].apply(lambda e: bound(class_count)) + df_dcl["nscl_loss"]
           df_dcl["upper_bound_test"] = df_dcl["epoch"].apply(lambda e: bound(class_count)) + df_dcl["nscl_loss_test"]
           
           plt.plot(df_dcl["epoch"], df_dcl["upper_bound"], color="black", label="Upper Bound", marker="^", linewidth=2.5)
           plt.plot(df_dcl["epoch"], df_dcl["upper_bound_test"], color="black", linestyle="--", label="Upper Bound (Test)", linewidth=2.5)

        plt.xscale("log")
        plt.xlim(1, 100)
        plt.xlabel("Training Epochs (log scale)", fontsize=14)
        plt.ylabel("Contrastive Loss", fontsize=14)
        plt.title(f"{mode} Labels", fontsize=16)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

        # Add +1 to xtick labels
        xticks = plt.xticks()[0]
        xtick_labels = [str(int(x + 1)) if x == int(x) else f"{x + 1:.2f}" for x in xticks]
        plt.xticks(xticks, xtick_labels)

        plt.legend(fontsize=12, loc="upper right")
        plt.tight_layout()

        filename = f"plot_nscl_dcl_diff_{mode.replace('%', 'pct').replace(' ', '_').lower()}_bs{batch_size}.png"
        full_path = os.path.join(output_dir, filename)
        plt.savefig(full_path)
        plt.close()
        print(f"✅ Saved plot: {full_path}")

plot_nscl_dcl_bounds(batch_size = 1024, class_count=10, output_dir="/scratch/user/aaryabookseller/Research/understanding-ssl/configs/mixed/results/loss_data/b1024/figure_2_bottom",
                     both =  True)
