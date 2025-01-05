from matplotlib import pyplot as plt


def print_plot(epochs, train_loss_values, test_loss_values, batch_size, lr, current_script_dir, layers_set, depth):
    plt.figure(figsize=(8, 5))
    plt.plot(list(range(1, epochs + 1)), train_loss_values, marker='o', linestyle='-', color='b', label='Train Loss')
    plt.plot(list(range(1, epochs + 1)), test_loss_values, marker='o', linestyle='--', color='r', label='Test Loss')
    plt.title('Loss durante le epoche')
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{current_script_dir}/loss_plot/grafico_loss_{batch_size}_{lr}_{layers_set}_{depth}.png", format='png', dpi=300)
    plt.close()