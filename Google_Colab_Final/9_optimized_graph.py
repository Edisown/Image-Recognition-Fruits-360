import seaborn as sns # Import the seaborn library
import matplotlib.pyplot as plt

def plot_optimized_results(history):
    # Register Seaborn styles with Matplotlib
    sns.set()  # Apply default Seaborn style

    plt.figure(figsize=(15, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training', marker='o')
    plt.plot(history['val_acc'], label='Validation', marker='o')
    plt.title('Model Accuracy with Optimizations')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training', marker='o')
    plt.plot(history['val_loss'], label='Validation', marker='o')
    plt.title('Model Loss with Optimizations')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('optimized_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print best metrics
    best_train_acc = max(history['train_acc'])
    best_val_acc = max(history['val_acc'])
    print(f"\nBest Training Accuracy: {best_train_acc:.2f}%")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

# Plot results
plot_optimized_results(history)