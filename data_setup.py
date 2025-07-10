import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader

def get_split_mnist_dataloaders(task_id, batch_size):
    """
    Returns the train and test DataLoaders for a specific Split MNIST task.
    For testing, it returns data for all tasks seen up to the current one.
    """
    if not (0 <= task_id < 5):
        raise ValueError("Task ID must be between 0 and 4")

    # Define a simple transformation to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the full datasets
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    full_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    tasks_classes = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    classes_for_task = tasks_classes[task_id]

    # Filter training dataset for the current task
    train_indices = [i for i, label in enumerate(full_train_dataset.targets) if label in classes_for_task]
    train_subset = Subset(full_train_dataset, train_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    
    # Filter testing dataset for all tasks seen so far
    classes_to_test = []
    for i in range(task_id + 1):
        classes_to_test.extend(tasks_classes[i])
        
    test_indices = [i for i, label in enumerate(full_test_dataset.targets) if label in classes_to_test]
    test_subset = Subset(full_test_dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader