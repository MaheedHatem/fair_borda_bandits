import os
import sys
from plot_results import plot_results_from_folder
import argparse
def main(base_directory, out_directory):
    # Current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create Figures folder
    figures_dir = os.path.join(current_dir, out_directory)
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Figures will be saved to: {figures_dir}")
    
    # Loop over directories
    for item in os.listdir(base_directory):
        item_path = os.path.join(base_directory, item)

        # Check if it's a directory and not the Figures directory itself
        if os.path.isdir(item_path) and item != "Figures":
            # Check for params.txt to confirm it's an experiment folder
            if os.path.exists(os.path.join(item_path, "params.txt")):
                print(f"Processing folder: {item}", flush=True)
                # Use the folder name as the suffix (U_K_T_minimum_gap)
                plot_results_from_folder(item_path, output_dir=figures_dir, suffix=item)
    print("Done")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run Dueling Bandits Simulation")
    parser.add_argument("--base_directory", type=str, default="./results/", help="Base directory for saving results")
    parser.add_argument("--out_directory", type=str, default="./Figures/", help="Base directory for saving results")

    args = parser.parse_args()
    main(args.base_directory, args.out_directory)
