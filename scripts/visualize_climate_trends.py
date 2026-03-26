import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

def plot_variable_trends(data_dir: Path, variable: str, resolution: str = "1H", last_n_files: int = 10):
    """
    Plots the digitized data for a specific variable and resolution.
    """
    target_dir = data_dir / variable / resolution
    if not target_dir.exists():
        print(f"Error: {target_dir} not found. No data to plot.")
        return

    files = sorted(list(target_dir.glob("*.parquet")))
    if not files:
        print(f"No parquet files found in {target_dir}")
        return

    # Select recent files or all
    if last_n_files > 0:
        files = files[-last_n_files:]
        
    print(f"Loading {len(files)} files for {variable}...")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values("timestamp")
    
    # Plotting
    plt.figure(figsize=(15, 6))
    
    # 1. Plot the main trace
    plt.plot(df['timestamp'], df['value'], label=f'{variable.capitalize()} Trace', color='blue', alpha=0.7)
    
    # Plotting is now simplified to just the data trace.

    plt.title(f"{variable.upper()} Trends - Digitized Universal Data ({resolution})", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(f"Value ({variable})", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # Save the plot
    out_path = Path("output/visuals")
    out_path.mkdir(parents=True, exist_ok=True)
    filename = out_path / f"trend_{variable}_{resolution}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Plot saved to: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Digitized Climate Data")
    parser.add_argument("--variable", type=str, default="temp", help="Variable to plot (temp, humidity, precip, pressure)")
    parser.add_argument("--res", type=str, default="1H", help="Resolution (1S, 1T, 1H)")
    parser.add_argument("--limit", type=int, default=12, help="Limit number of recent files to plot (0 for all)")
    args = parser.parse_args()

    data_root = Path("/Users/yasinkaya/Hackhaton/output/universal_datasets")
    plot_variable_trends(data_root, args.variable, args.res, args.limit)
