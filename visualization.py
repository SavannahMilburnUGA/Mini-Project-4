# Visualize ASL Neural Network Comparison Results
# Creates clean, presentation-ready charts comparing 3 network architectures

import matplotlib.pyplot as plt
import numpy as np

# Set modern academic styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'text.color': '#333333'
})

# Sample results - replace with your actual results after running training
# Format: {'network_name': {'accuracy': correct_predictions, 'time': seconds}}
results = {
    'Simple': {'accuracy': 4500, 'time': 120},        # Example: 62.7% accuracy
    'Improved': {'accuracy': 5200, 'time': 180},      # Example: 72.5% accuracy  
    'Deep': {'accuracy': 5800, 'time': 450}           # Example: 80.9% accuracy
}

# Total test samples (update with your actual test set size)
total_test_samples = 7172

def create_comparison_charts():
    """Create side-by-side comparison charts"""
    
    # Convert to percentages and minutes
    networks = list(results.keys())
    accuracies = [(results[net]['accuracy'] / total_test_samples) * 100 for net in networks]
    times = [results[net]['time'] / 60 for net in networks]  # Convert to minutes
    
    # Set up the plot with modern academic styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle('ASL Letter Recognition: Neural Network Comparison', 
                fontsize=18, fontweight='600', color='#2C3E50', y=0.95)
    
    # Modern academic color palette
    colors = ['#3498DB', '#E74C3C', '#2ECC71']  # Blue, Red, Green
    
    # Chart 1: Accuracy Comparison
    bars1 = ax1.bar(networks, accuracies, color=colors, alpha=0.85, 
                   edgecolor='white', linewidth=2, width=0.6)
    ax1.set_title('Classification Accuracy', fontsize=15, fontweight='600', 
                 color='#2C3E50')
    ax1.set_ylabel('Accuracy (%)', fontsize=13, color='#2C3E50')
    ax1.set_ylim(0, max(accuracies) * 1.15)
    
    # Add percentage labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.annotate(f'{acc:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='600', color='#2C3E50')
    
    # Chart 2: Training Time Comparison
    bars2 = ax2.bar(networks, times, color=colors, alpha=0.85, 
                   edgecolor='white', linewidth=2, width=0.6)
    ax2.set_title('Training Time', fontsize=15, fontweight='600', 
                 color='#2C3E50')
    ax2.set_ylabel('Time (minutes)', fontsize=13, color='#2C3E50')
    ax2.set_ylim(0, max(times) * 1.15)
    
    # Add time labels on bars
    for bar, time_val in zip(bars2, times):
        height = bar.get_height()
        ax2.annotate(f'{time_val:.1f}m',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='600', color='#2C3E50')
    
    # Style the x-axis labels
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', labelsize=11, colors='#2C3E50')
        ax.tick_params(axis='y', labelsize=10, colors='#2C3E50')
    
    plt.tight_layout()
    plt.savefig('asl_network_comparison.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

def create_summary_table():
    """Create a summary table of results"""
    
    print("\n" + "="*60)
    print("           ASL NEURAL NETWORK COMPARISON")
    print("="*60)
    print(f"{'Network':<12} {'Accuracy':<12} {'Time':<12} {'Parameters':<12}")
    print("-"*60)
    
    # Calculate parameters for each network (approximate)
    param_counts = {
        'Simple': 784*30 + 30*24 + 30 + 24,           # ~24K parameters
        'Improved': 784*30 + 30*24 + 30 + 24,         # ~24K parameters  
        'Deep': 784*100 + 100*50 + 50*25 + 25*24 + 100 + 50 + 25 + 24  # ~85K parameters
    }
    
    for network in results.keys():
        accuracy = (results[network]['accuracy'] / total_test_samples) * 100
        time_min = results[network]['time'] / 60
        params = param_counts[network]
        
        print(f"{network:<12} {accuracy:>8.1f}%    {time_min:>8.1f}m    {params/1000:>8.1f}K")

def create_architecture_diagram():
    """Create a simple diagram showing the three architectures"""
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Neural Network Architectures for ASL Recognition', 
                fontsize=18, fontweight='600', color='#2C3E50', y=0.95)
    
    # Architecture details
    architectures = {
        'Simple Network\n[784→30→24]': {'layers': [784, 30, 24], 'color': '#3498DB'},
        'Improved Network\n[784→30→24]': {'layers': [784, 30, 24], 'color': '#E74C3C'},
        'Deep Network\n[784→100→50→25→24]': {'layers': [784, 100, 50, 25, 24], 'color': '#2ECC71'}
    }
    
    axes = [ax1, ax2, ax3]
    
    for idx, (title, arch) in enumerate(architectures.items()):
        ax = axes[idx]
        layers = arch['layers']
        color = arch['color']
        
        # Draw layers as rectangles
        max_neurons = max(layers)
        for i, neurons in enumerate(layers):
            height = neurons / max_neurons * 0.7  # Scale height
            y_pos = (1 - height) / 2
            rect = plt.Rectangle((i*1.2, y_pos), 0.8, height, 
                               facecolor=color, alpha=0.8, edgecolor='white', linewidth=2)
            ax.add_patch(rect)
            
            # Add neuron count labels ABOVE the blocks
            ax.text(i*1.2 + 0.4, y_pos + height + 0.05, str(neurons), 
                   ha='center', va='bottom', fontsize=11, fontweight='600', color='#2C3E50')
        
        ax.set_xlim(-0.3, len(layers)*1.2)
        ax.set_ylim(0, 1.2)
        ax.set_title(title, fontsize=14, fontweight='600', color='#2C3E50')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')  # Remove all axes for cleaner look
    
    plt.tight_layout()
    plt.savefig('asl_architectures.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

if __name__ == "__main__":
    print("Creating ASL Neural Network Visualizations...")
    print("\n1. Comparison Charts")
    create_comparison_charts()
    
    print("\n2. Results Summary")
    create_summary_table()
    
    print("\n3. Architecture Diagrams")
    create_architecture_diagram()
    
    print("\nVisualization complete! Check the saved PNG files.")