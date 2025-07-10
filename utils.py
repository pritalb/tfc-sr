import torch
from torch.utils.data import Subset, DataLoader
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

def set_seed(seed):
    """Sets the seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_results(results, filepath):
    """Saves the results dictionary to a file using pickle."""
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {filepath}")

def load_results(filepath):
    """Loads a results dictionary from a file using pickle."""
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    print(f"Results loaded from {filepath}")
    return results

def plot_results(results_dict, title="Continual Learning Performance on Split MNIST"):
    """
    Plots the average accuracy curves for multiple experiments.
    
    Args:
    - results_dict (dict): A dictionary where keys are experiment names (e.g., 'Baseline')
                           and values are the list/array of accuracies per task.
    """
    plt.figure(figsize=(10, 6))
    
    for experiment_name, accuracies in results_dict.items():
        # The x-axis represents the number of tasks learned (1 through 5)
        tasks = range(1, len(accuracies) + 1)
        plt.plot(tasks, accuracies, marker='o', linestyle='-', label=experiment_name)
        
    plt.title(title)
    plt.xlabel("Number of Tasks Learned")
    plt.ylabel("Average Accuracy on All Seen Tasks (%)")
    plt.xticks(range(1, len(next(iter(results_dict.values()))) + 1))
    plt.ylim(0, 101) # Accuracy is between 0 and 100
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()


def create_buffer_validation_set(replay_buffer, val_split=0.1):
    """Splits the replay buffer into a training and validation set."""
    buffer_size = len(replay_buffer.buffer)
    val_size = int(buffer_size * val_split)
    
    # Ensure we don't try to take more than available for validation
    val_size = min(val_size, buffer_size)
    
    # Shuffle the buffer to get a random validation set
    shuffled_buffer = list(replay_buffer.buffer)
    random.shuffle(shuffled_buffer)
    
    val_samples = shuffled_buffer[:val_size]
    
    if not val_samples:
        return None

    data_batch, target_batch = zip(*val_samples)
    return torch.stack(data_batch), torch.stack(target_batch)

class ReservoirReplayBuffer:
    """A simple Reservoir Sampling Replay Buffer."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.num_items_seen = 0

    def add(self, data, target):
        """Adds an item (data, target) to the buffer."""
        item = (data.clone(), torch.tensor(target)) # Store a copy

        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            # Reservoir sampling logic
            idx_to_replace = random.randint(0, self.num_items_seen)
            if idx_to_replace < self.capacity:
                self.buffer[idx_to_replace] = item
        
        self.num_items_seen += 1

    def sample(self, batch_size):
        """Samples a batch of items from the buffer."""
        # Ensure we don't sample more than what's available
        actual_batch_size = min(batch_size, len(self.buffer))
        
        samples = random.sample(self.buffer, actual_batch_size)
        
        # Unzip the list of tuples into two tensors
        data_batch, target_batch = zip(*samples)
        
        return torch.stack(data_batch), torch.stack(target_batch)

    def __len__(self):
        return len(self.buffer)
        
def evaluate_on_seen_tasks(model, benchmark, task_id, device, batch_size):
    """
    Evaluates a model on all tasks seen so far.
    
    Args:
    - model: The model to evaluate.
    - benchmark: The Avalanche benchmark object (e.g., SplitMNIST).
    - task_id: The ID of the current task (0 to 4).
    - device: The torch device (e.g., 'cuda').
    - batch_size: The batch size for evaluation.
    
    Returns:
    - The average accuracy in percent.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        # Get the test stream for all tasks up to the current one
        test_stream_seen_so_far = benchmark.test_stream[:task_id + 1]
        
        for experience in test_stream_seen_so_far:
            test_loader = DataLoader(experience.dataset, batch_size=batch_size)
            for data, targets, _ in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
    return (100.0 * correct / total) if total > 0 else 0.0

def evaluate_replay_buffer(model, replay_buffer, device, batch_size=256):
    """
    Evaluates the model's performance on the replay buffer using
    a class-incremental, context-aware accuracy calculation.
    """
    model.eval()
    
    if len(replay_buffer.buffer) == 0:
        return 100.0 # If buffer is empty, memory is "perfect" by default

    # Get all unique classes currently in the buffer
    buffer_classes_cpu = torch.unique(torch.tensor([item[1] for item in replay_buffer.buffer]))
    
    # Move the indices tensor to the same device as the model
    buffer_classes = buffer_classes_cpu.to(device)
    
    # Sample a batch for validation
    val_data, val_targets = replay_buffer.sample(batch_size=min(len(replay_buffer.buffer), batch_size))
    val_data, val_targets = val_data.to(device), val_targets.to(device)
    
    correct_val = 0
    with torch.no_grad():
        outputs = model(val_data)
        outputs_for_buffer_classes = outputs[:, buffer_classes]
        
        _, predicted_indices = torch.max(outputs_for_buffer_classes, 1)
        predicted_classes = buffer_classes[predicted_indices]
        correct_val = (predicted_classes == val_targets).sum().item()

    return (100.0 * correct_val / len(val_targets)) if len(val_targets) > 0 else 0.0

from avalanche.training.plugins import SupervisedPlugin

def run_avalanche_strategy(strategy, benchmark, device):
    """
    Runs a complete continual learning experiment for a given Avalanche strategy.

    This function handles the training loop over all experiences and evaluates
    after each experience using our custom, unified evaluation function.

    Args:
    - strategy: The Avalanche strategy object (e.g., EWC, SI).
    - benchmark: The Avalanche benchmark object (e.g., SplitMNIST).
    - device: The torch device to run on.

    Returns:
    - A list of average accuracies on tasks seen so far, after each experience.
    """
    print(f"--- Starting run for strategy: {strategy.__class__.__name__} ---")
    
    accuracies = []
    
    # Main continual learning loop
    for task_id, experience in enumerate(benchmark.train_stream):
        print(f"--> Training on experience {task_id+1}/{len(benchmark.train_stream)}")
        
        # The train method just does the training on the current experience
        strategy.train(experience)
        
        print(f"--> Evaluating after experience {task_id+1}")
        accuracy = evaluate_on_seen_tasks(
            strategy.model, 
            benchmark, 
            task_id, 
            device, 
            strategy.train_mb_size # Use the batch size from the strategy
        )
        accuracies.append(accuracy)
        print(f"----- Avg Accuracy after Task {task_id+1}: {accuracy:.2f}% -----")
        
    return accuracies

class TFCSR_Plugin(SupervisedPlugin):
    """
    An Avalanche Plugin to implement Task-Focused Consolidation with Spaced Replay.
    Combines with any base strategy (like EWC, SI, or Naive).
    """
    def __init__(self, replay_buffer, mastery_threshold, initial_replay_gap, replay_gap_multiplier):
        super().__init__()
        self.replay_buffer = replay_buffer
        self.mastery_threshold = mastery_threshold
        self.initial_replay_gap = initial_replay_gap
        self.replay_gap_multiplier = replay_gap_multiplier
        
        # Internal state for scheduling
        self.current_replay_gap = float(self.initial_replay_gap)
        self.replay_timer = int(self.current_replay_gap)

    def before_training_exp(self, strategy, **kwargs):
        """Called before training on a new experience. Used to populate the buffer."""
        print(f"Populating replay buffer from Task {strategy.experience.current_experience}...")
        for data_point, target, _ in strategy.experience.dataset:
            self.replay_buffer.add(data_point, target)
        print(f"Replay buffer size: {len(self.replay_buffer)}")
        
        # Reset the schedule for the new task
        self.current_replay_gap = float(self.initial_replay_gap)
        self.replay_timer = int(self.current_replay_gap)

    def before_training_iteration(self, strategy, **kwargs):
        """Called before each training iteration. Used to create the mixed batch."""
        if len(self.replay_buffer) > strategy.train_mb_size // 2:
            replay_batch_size = strategy.train_mb_size // 2
            
            # Get old data from buffer
            old_data, old_targets = self.replay_buffer.sample(replay_batch_size)
            
            # Get new data from the current mini-batch
            new_data = strategy.mb_x[:replay_batch_size]
            new_targets = strategy.mb_y[:replay_batch_size]
            
            # Overwrite the strategy's current mini-batch with our mixed batch
            strategy.mb_x = torch.cat((new_data, old_data), dim=0).to(strategy.device)
            strategy.mb_y = torch.cat((new_targets, old_targets), dim=0).to(strategy.device)

    def after_training_epoch(self, strategy, **kwargs):
        """Called after each training epoch. Used for the memory check."""
        epoch = strategy.epoch
        print(f"Task {strategy.experience.current_experience+1}, Epoch {epoch+1}", end="")

        if (epoch + 1) == self.replay_timer and len(self.replay_buffer) > 1:
            print(" <-- Memory Check!", end="")
            strategy.model.eval()
            
            val_data, val_targets = self.replay_buffer.sample(batch_size=min(len(self.replay_buffer), 256))
            val_data, val_targets = val_data.to(strategy.device), val_targets.to(strategy.device)
            
            with torch.no_grad():
                outputs = strategy.model(val_data)
                _, predicted = torch.max(outputs.data, 1)
                correct_val = (predicted == val_targets).sum().item()
                replay_perf = 100 * correct_val / len(val_targets)
            
            print(f" | Replay Perf: {replay_perf:.2f}%", end="")

            if replay_perf >= self.mastery_threshold:
                self.current_replay_gap *= self.replay_gap_multiplier
                self.replay_timer += round(self.current_replay_gap)
                print(f" | Mastery OK. Next check @ epoch {self.replay_timer}.")
            else:
                self.replay_timer += 1
                print(f" | Mastery FAIL. Next check @ epoch {self.replay_timer}.")
            
            strategy.model.train()
        else:
            print()