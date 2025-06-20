# Combined ASL Neural Network Visualization
# Includes learning curves + comparison charts + architecture diagrams

import matplotlib.pyplot as plt
import numpy as np
import pickle

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

def load_training_data():
    """Load actual training results from pickle file"""
    try:
        with open('asl_training_results.pkl', 'rb') as f:
            results = pickle.load(f)
        return results
    except FileNotFoundError:
        print("Training results file not found! Using sample data for demo.")
        # Return sample data if no training results exist
        return {
            'simple': {'accuracy': 4500, 'time': 120, 'train_accuracy': [3000]*30, 'val_accuracy': [2500]*30, 'train_cost': [0]*30, 'val_cost': [0]*30},
            'improved': {'accuracy': 5200, 'time': 180, 'train_accuracy': [4000]*30, 'val_accuracy': [3500]*30, 'train_cost': [1.5]*30, 'val_cost': [1.8]*30},
            'deep': {'accuracy': 5800, 'time': 450, 'train_accuracy': [4500]*30, 'val_accuracy': [4000]*30, 'train_cost': [1.2]*30, 'val_cost': [1.4]*30}
        }

def convert_to_percentages(accuracy_list, total_samples):
    """Convert raw accuracy counts to percentages"""
    return [(acc / total_samples) * 100 for acc in accuracy_list]

def create_learning_curves(data, training_size=22000, validation_size=5000):
    """Create learning curves visualization"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('ASL Neural Network Learning Curves', 
                fontsize=18, fontweight='600', color='#2C3E50', y=0.98)
    
    # Colors for each network
    colors = {'simple': '#3498DB', 'improved': '#E74C3C', 'deep': '#2ECC71'}
    epochs = range(1, 31)  # 30 epochs
    
    # Top plot: Accuracy curves
    for network_name in ['simple', 'improved', 'deep']:
        if network_name in data:
            network_data = data[network_name]
            
            # Convert to percentages
            train_acc_pct = convert_to_percentages(network_data['train_accuracy'], training_size)
            val_acc_pct = convert_to_percentages(network_data['val_accuracy'], validation_size)
            
            ax1.plot(epochs, train_acc_pct, 
                   color=colors[network_name], linestyle='-', linewidth=2.5,
                   label=f'{network_name.title()} - Training', alpha=0.8)
            
            ax1.plot(epochs, val_acc_pct,
                   color=colors[network_name], linestyle='--', linewidth=2.5, 
                   label=f'{network_name.title()} - Validation', alpha=0.8)
    
    ax1.set_ylabel('Accuracy (%)', fontsize=12, color='#2C3E50')
    ax1.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='600', color='#2C3E50')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Bottom plot: Loss curves (only for improved and deep networks)
    for network_name in ['improved', 'deep']:
        if network_name in data and len(data[network_name]['train_cost']) > 0:
            network_data = data[network_name]
            
            # Only plot if we have actual cost data (not zeros)
            if max(network_data['train_cost']) > 0:
                ax2.plot(epochs, network_data['train_cost'], 
                       color=colors[network_name], linestyle='-', linewidth=2.5,
                       label=f'{network_name.title()} - Training Loss', alpha=0.8)
                
                ax2.plot(epochs, network_data['val_cost'],
                       color=colors[network_name], linestyle='--', linewidth=2.5,
                       label=f'{network_name.title()} - Validation Loss', alpha=0.8)
    
    ax2.set_xlabel('Epoch', fontsize=12, color='#2C3E50')
    ax2.set_ylabel('Loss', fontsize=12, color='#2C3E50')
    ax2.set_title('Training vs Validation Loss', fontsize=14, fontweight='600', color='#2C3E50')
    ax2.grid(True, alpha=0.3)
    
    # Single legend for both subplots
    handles1, labels1 = ax1.get_legend_handles_labels()
    fig.legend(handles1, labels1, loc='center right', 
              frameon=True, fancybox=True, shadow=True, fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('asl_learning_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def create_comparison_charts(data, total_test_samples=7172):
    """Create side-by-side comparison charts"""
    
    # Extract data for comparison
    networks = ['Simple', 'Improved', 'Deep']
    network_keys = ['simple', 'improved', 'deep']
    accuracies = [(data[key]['accuracy'] / total_test_samples) * 100 for key in network_keys]
    times = [data[key]['time'] / 60 for key in network_keys]  # Convert to minutes
    
    # Set up the plot
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

def create_summary_table(data, total_test_samples=7172):
    """Create a summary table of results"""
    
    print("\n" + "="*60)
    print("           ASL NEURAL NETWORK COMPARISON")
    print("="*60)
    print(f"{'Network':<12} {'Accuracy':<12} {'Time':<12} {'Parameters':<12}")
    print("-"*60)
    
    # Calculate parameters for each network (approximate)
    param_counts = {
        'simple': 784*30 + 30*24 + 30 + 24,           # ~24K parameters
        'improved': 784*30 + 30*24 + 30 + 24,         # ~24K parameters  
        'deep': 784*100 + 100*50 + 50*25 + 25*24 + 100 + 50 + 25 + 24  # ~85K parameters
    }
    
    network_names = ['Simple', 'Improved', 'Deep']
    network_keys = ['simple', 'improved', 'deep']
    
    for name, key in zip(network_names, network_keys):
        accuracy = (data[key]['accuracy'] / total_test_samples) * 100
        time_min = data[key]['time'] / 60
        params = param_counts[key]
        
        print(f"{name:<12} {accuracy:>8.1f}%    {time_min:>8.1f}m    {params/1000:>8.1f}K")

if __name__ == "__main__":
    print("Creating Combined ASL Neural Network Visualizations...")
    
    # Load training data (or use sample data if not available)
    data = load_training_data()
    
    print("\n1. Learning Curves")
    create_learning_curves(data)
    
    print("\n2. Comparison Charts")
    create_comparison_charts(data)
    
    print("\n3. Architecture Diagrams")
    create_architecture_diagram()
    
    print("\n4. Results Summary")
    create_summary_table(data)
    
    print("\nVisualization complete! Generated files:")
    print("• asl_learning_curves.png")
    print("• asl_network_comparison.png") 
    print("• asl_architectures.png")