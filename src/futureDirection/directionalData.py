import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Load the dataset
directional = pd.read_csv('./data/GTA Oil_Data_03Jan2018.csv')

# Create a directed graph
G = nx.DiGraph()

# Group and filter data
grouped = directional.groupby(['Exporting Country', 'Importing Country'])['Quantity in BBL'].sum().to_frame().reset_index().loc[lambda x: x['Quantity in BBL'] > 0]

# Identify all unique countries and their roles
all_exporters = set(grouped['Exporting Country'])
all_importers = set(grouped['Importing Country'])
all_countries = all_exporters.union(all_importers)

# Calculate total export and import volumes for each country
export_volumes = defaultdict(float)
import_volumes = defaultdict(float)

for _, row in grouped.iterrows():
    exporter = row['Exporting Country']
    importer = row['Importing Country']
    volume = row['Quantity in BBL']
    export_volumes[exporter] += volume
    import_volumes[importer] += volume

# Determine primary role for each country (exporter or importer)
# A country is classified based on whether it exports more than it imports
exporters = []
importers = []

for country in all_countries:
    if export_volumes[country] > import_volumes[country]:
        exporters.append((country, export_volumes[country]))
    else:
        importers.append((country, import_volumes[country]))

# Sort by volume
exporters.sort(key=lambda x: x[1], reverse=True)
importers.sort(key=lambda x: x[1], reverse=True)

# Take only the top 15 exporters and importers
top_exporters = exporters[:15]
top_importers = importers[:15]

# Create sets of the top countries for quick lookup
top_exporter_countries = {country for country, _ in top_exporters}
top_importer_countries = {country for country, _ in top_importers}

# Create a fresh graph with only the connections between top exporters and importers
G_filtered = nx.DiGraph()

# Add edges only between top exporters and top importers
for _, row in grouped.iterrows():
    exporter = row['Exporting Country']
    importer = row['Importing Country']
    volume = row['Quantity in BBL']
    
    if exporter in top_exporter_countries and importer in top_importer_countries:
        G_filtered.add_edge(exporter, importer, weight=volume)

# Create positions for a vertical bipartite layout
pos = {}
left_spacing = 1.0 / (len(top_exporters) + 1)
right_spacing = 1.0 / (len(top_importers) + 1)

# Position exporters on the left, from top to bottom by volume
for i, (country, _) in enumerate(top_exporters):
    pos[country] = (-1, 1 - left_spacing * (i + 1))

# Position importers on the right, from top to bottom by volume
for i, (country, _) in enumerate(top_importers):
    pos[country] = (1, 1 - right_spacing * (i + 1))

# Calculate node sizes based on volume
visible_countries = set([country for country, _ in top_exporters] + [country for country, _ in top_importers])
max_export = max(export_volumes[country] for country in visible_countries if export_volumes[country] > 0)
max_import = max(import_volumes[country] for country in visible_countries if import_volumes[country] > 0)
max_volume = max(max_export, max_import)

node_sizes = {}
for country in visible_countries:
    total_volume = export_volumes[country] + import_volumes[country]
    node_sizes[country] = 300 + 2000 * (total_volume / max_volume)

# Set up the figure
plt.figure(figsize=(16, 20), dpi=100)

# Extract edge weights for visualization
edge_weights = [G_filtered[u][v]['weight'] for u, v in G_filtered.edges()]
if edge_weights:  # Check if there are any edges
    max_weight = max(edge_weights)
    edge_widths = [0.5 + 3 * (w / max_weight) for w in edge_weights]
    edge_alphas = [0.2 + 0.6 * (w / max_weight) for w in edge_weights]
    edge_colors = [(0.5, 0.5, 0.5, alpha) for alpha in edge_alphas]

    # Draw edges with curved paths
    for i, (u, v) in enumerate(G_filtered.edges()):
        plt.annotate(
            "", xy=pos[v], xytext=pos[u],
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle=f"arc3,rad=0.1",
                color=edge_colors[i],
                lw=edge_widths[i],
                alpha=edge_alphas[i]
            )
        )

# Draw nodes
for role, color, label in [(top_exporters, 'lightcoral', 'Top 15 Exporters'), (top_importers, 'lightblue', 'Top 15 Importers')]:
    nx.draw_networkx_nodes(
        G_filtered, pos,
        nodelist=[country for country, _ in role],
        node_size=[node_sizes[country] for country, _ in role],
        node_color=color,
        alpha=0.8,
        edgecolors='white',
        linewidths=1.5,
        label=label
    )

# Draw labels with good visibility
label_offset = -0.06
# label_offset = 0.0
left_labels = {country: (pos[country][0] - label_offset, pos[country][1]) for country, _ in top_exporters}
right_labels = {country: (pos[country][0] + label_offset, pos[country][1]) for country, _ in top_importers}

nx.draw_networkx_labels(
    G_filtered, left_labels,
    labels={country: f"{country}\n({int(export_volumes[country]/1e6)}M BBL)" for country, _ in top_exporters},
    font_size=10,
    font_weight='bold',
    horizontalalignment='right'
)

nx.draw_networkx_labels(
    G_filtered, right_labels,
    labels={country: f"{country}\n({int(import_volumes[country]/1e6)}M BBL)" for country, _ in top_importers},
    font_size=10,
    font_weight='bold',
    horizontalalignment='left'
)

# Add title and legend, and shift down the title
plt.title('Oil Trade Network: Top 15 Exporters and Importers', fontsize=20, y=0.96)

# Add legend manually
plt.plot([], [], 'o', color='lightcoral', markersize=10, label='Top 15 Exporters')
plt.plot([], [], 'o', color='lightblue', markersize=10, label='Top 15 Importers')
# plt.legend(loc='upper center', fontsize=12)

# Remove axes
plt.axis('off')
plt.tight_layout()

# Show the plot
plt.show()