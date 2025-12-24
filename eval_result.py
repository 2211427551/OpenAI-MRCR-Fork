import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import platform
import glob
import numpy as np

# ================= Configuration =================
LOG_PATTERN = "*.jsonl"
OUTPUT_IMAGE = "merged_heatmap_polished.png"

# Context Length Bins
BINS = [0, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1000000]
LABELS = ["≤4K", "4K-8K", "8K-16K", "16K-32K", "32K-64K", "64K-128K", "128K-256K", "256K-512K", ">512K"]
# ===============================================

def configure_plot_style():
    """
    Configure plot fonts and style settings.
    """
    system = platform.system()
    # Setting fonts that support unicode symbols just in case, though standard fonts work for English
    if system == "Windows":
        fonts = ['Arial', 'Microsoft YaHei', 'SimHei']
    elif system == "Darwin":
        fonts = ['Arial', 'Arial Unicode MS', 'PingFang SC']
    else:
        fonts = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    
    plt.rcParams['font.sans-serif'] = fonts + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False


def load_all_logs():
    """
    Load all matching JSONL files from the directory.
    """
    all_files = glob.glob(LOG_PATTERN)
    if not all_files:
        print(f"❌ No matching files found: {LOG_PATTERN}")
        return None

    df_list = []
    for filename in all_files:
        try:
            temp_df = pd.read_json(filename, lines=True)
            required_cols = {'model', 'score', 'token_count'}
            if required_cols.issubset(temp_df.columns):
                df_list.append(temp_df)
        except:
            pass

    if not df_list: return None
    return pd.concat(df_list, ignore_index=True)


def generate_heatmap():
    # 1. Load and Clean Data
    df = load_all_logs()
    if df is None or df.empty: return

    # Fill default needle count if missing
    if 'needle_count' not in df.columns:
        df['needle_count'] = 2
    df['needle_count'] = df['needle_count'].fillna(2).astype(int)

    # Binning context lengths
    df = df.dropna(subset=['token_count', 'score'])
    df['Length_Bin'] = pd.cut(df['token_count'], bins=BINS, labels=LABELS, right=False)

    # 2. Aggregation
    grouped = df.groupby(['model', 'needle_count', 'Length_Bin'], observed=False).agg(
        avg_score=('score', 'mean'),
        count=('score', 'count')
    )

    heatmap_data = grouped['avg_score'].unstack()
    counts_data = grouped['count'].unstack()

    # 3. Remove Empty Rows
    # Drop rows where all values are NaN (i.e., no data for that model/needle combo)
    rows_with_data = heatmap_data.dropna(how='all').index
    heatmap_data = heatmap_data.loc[rows_with_data]
    counts_data = counts_data.loc[rows_with_data]

    if heatmap_data.empty:
        print("❌ No valid data available to plot.")
        return

    # Sort indices
    heatmap_data = heatmap_data.sort_index(level=[0, 1])
    counts_data = counts_data.sort_index(level=[0, 1])

    # 4. Plot Setup
    configure_plot_style()
    num_rows = len(heatmap_data)
    # Adjust height dynamically based on number of rows
    fig_height = max(5, 1.8 + num_rows * 0.9)

    fig, ax = plt.subplots(figsize=(16, fig_height))
    # Adjust margins to fit model names on the left
    plt.subplots_adjust(left=0.22, bottom=0.15)

    # Set background color for "N/A" areas
    ax.set_facecolor('#f0f0f0') 

    # Color map: Red -> Orange -> Yellow -> Green
    colors = ["#ff6b6b", "#ff9f43", "#feca57", "#1dd1a1"]
    cmap = mcolors.LinearSegmentedColormap.from_list("CustomRdYlGn", colors)

    # 5. Draw Heatmap (Disable auto-annotation)
    sns.heatmap(
        heatmap_data,
        annot=False,
        cmap=cmap,
        vmin=0, vmax=1,
        linewidths=1.5,
        linecolor='white',
        cbar_kws={'label': 'Average Score', 'shrink': 0.6, 'aspect': 20},
        ax=ax
    )

    # 6. Manual Annotation Loop
    for y in range(heatmap_data.shape[0]):
        for x in range(heatmap_data.shape[1]):
            score = heatmap_data.iloc[y, x]
            count = counts_data.iloc[y, x]

            # Coordinate adjustment (x+0.5, y+0.5 is the center)
            if pd.isna(score) or count == 0:
                # Draw "N/A" for empty data
                ax.text(x + 0.5, y + 0.5, "N/A",
                        ha='center', va='center', color='#bbbbbb', fontsize=10)
            else:
                # Determine text color based on background intensity
                text_color = 'white' if score > 0.6 or score < 0.3 else '#333333'
                if 0.3 <= score <= 0.6: text_color = '#333333'

                # 6.1 Draw Percentage (Bold, Larger)
                ax.text(x + 0.5, y + 0.4, f"{score:.0%}",
                        ha='center', va='center',
                        color=text_color, fontsize=12, fontweight='bold')

                # 6.2 Draw Sample Count (Smaller, Transparent)
                ax.text(x + 0.5, y + 0.75, f"(n={int(count)})",
                        ha='center', va='center',
                        color=text_color, fontsize=9, alpha=0.8)

    # 7. Labels and Grouping Lines
    indices = heatmap_data.index.tolist()

    # Y-axis ticks (Needle Counts)
    yticks = np.arange(len(indices)) + 0.5
    # Simplified label: "2 Needles" instead of bilingual
    yticklabels = [f"{idx[1]} Needles" for idx in indices]
    
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, rotation=0, fontsize=10, color='#555555')
    ax.tick_params(axis='y', length=0) # Hide tick lines

    # Left-side Model Names + Separator Lines
    model_positions = {}
    for i, (model_name, needle) in enumerate(indices):
        if model_name not in model_positions: model_positions[model_name] = []
        model_positions[model_name].append(i)

    # Offset for model name label
    PAD_MODEL_NAME = -0.08

    for i, (model_name, rows) in enumerate(model_positions.items()):
        center_y = (rows[0] + rows[-1] + 1) / 2

        # Draw Model Name
        ax.text(
            PAD_MODEL_NAME, center_y,
            model_name,
            transform=ax.get_yaxis_transform(),
            ha='right', va='center',
            fontsize=13, fontweight='bold', color='#2c3e50'
        )

        # Draw Separator Line (Dark Grey)
        if rows[-1] < len(indices) - 1:
            line_y = rows[-1] + 1
            ax.axhline(y=line_y, color='#7f8c8d', linewidth=1.5, xmin=0, xmax=1)

    # 8. Global Decorations
    ax.set_xlabel("Context Length", fontsize=12, labelpad=15, fontweight='bold', color='#555555')
    ax.set_ylabel("")
    plt.xticks(rotation=0, fontsize=10, color='#555555')

    # Title
    plt.title("OpenAI MRCR Long Context Benchmark",
              fontsize=18, fontweight='bold', pad=30, color='#2c3e50')

    # Remove frame borders, keep only data area
    sns.despine(top=True, right=True, left=True, bottom=False)

    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f"✅ Polished heatmap saved to: {OUTPUT_IMAGE}")
    plt.show()


if __name__ == "__main__":
    generate_heatmap()
