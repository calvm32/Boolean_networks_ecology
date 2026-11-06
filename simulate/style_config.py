import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def apply_publication_style():
    # Applies global publication-grade rcParams for Ecological Modelling
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.edgecolor': '#444444',
        'axes.linewidth': 1.0,
        'grid.color': '#e0e0e0',
        'grid.linestyle': '--',
        'grid.linewidth': 0.6,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',  # Standard vector format for Elsevier / Ecological Modelling
    })

# Unified Ecological Palette
COLORS = {
    "Hi": "#2b5c8f",        # Deep steel blue (Hibernating)
    "Ot": "#e2AC2F",        # Yellow (Susceptible/Other)
    "In": "#cb181d",        # Rust crimson (Infected)
    "Im": "#1b9e77",        # Vibrant teal (Immune)
    "De": "#636363",        # Slate gray (Deceased)
    "Total": "#111111",     # Charcoal black
    "Sobol_S1": "#2b5c8f",  # Deep steel blue (First-order)
    "Sobol_ST": "#d95f02",  # Warm terracotta (Total-order) - academic, neutral contrast
}

WINTER_COLOR = "#c6dbef"    # ice blue for winter span highlights