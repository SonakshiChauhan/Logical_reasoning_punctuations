
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def save_layer_accuracies(layer_accuracies, args, project_root):
    """Create and save accuracy vs layer plot."""
    accs = list(layer_accuracies.values())
    accs = [acc * 100 for acc in accs]
    sns.set(style="darkgrid")

    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(accs, marker='o', linestyle='-', linewidth=2, markersize=6,
            color='#E65100', label="Accuracy")  # Use the orange-reddish color
    plt.title(f'Accuracy vs. Layer for {args.intervention_type} Intervention – {args.model_name}', 
              fontsize=16, fontweight='bold')
    
    # Labels and titles
    plt.xlabel('Layers', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    plt.ylim(-5, 105)
    
    # Formatting axes
    plt.xticks(np.arange(0, args.num_layers, 1), fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper right')
    
    # Save the figure in high resolution
    output_dir = project_root / "intervention" / "plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/accuracy_vs_layer_{args.intervention_type}_{args.model_name}.png", 
                bbox_inches='tight', dpi=300)
    

def create_symmetric_heatmap(args,list_of_lists,project_root):
    # Determine the size of the square matrix
    n = len(list_of_lists)
    matrix = np.full((n, n), np.nan)
    for i in range(n):
        current_list = list_of_lists[i]
        for j in range(len(current_list)):
            if i + j < n:  
                matrix[i, i + j] = current_list[j]
    for i in range(n):
        for j in range(i+1, n):
            if not np.isnan(matrix[i, j]):
                matrix[j, i] = matrix[i, j]
    plt.figure(figsize=(12, 8))
    mask = np.isnan(matrix)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(matrix,
                mask=mask,
                cmap=cmap,
                annot=True,
                fmt=".2f",
                linewidths=.5,
                cbar_kws={"shrink": .5})

    plt.title(f'{args.model_type} {args.swap_type}layer swap heatmap')
    plt.xlabel('Layer Index')
    plt.ylabel('Layer Index')
    plt.tight_layout()
    plt.show()
    output_dir = project_root / "layer_swap" / "plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/layer_swap_{args.swap_type_type}_{args.model_name}.png", 
                bbox_inches='tight', dpi=300)

    return matrix