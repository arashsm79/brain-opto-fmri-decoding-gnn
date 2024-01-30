import os
import numpy as np
import time
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.profile import count_parameters
from torch.optim import lr_scheduler
import pandas as pd
import argparse

from LCNAData import LCNAData
from torch_geometric.loader import DataLoader
from model import GNN
from loss import topk_loss, unit_loss, consist_loss
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score



def main():
    parser = argparse.ArgumentParser(description='Path to the project directory.')
    parser.add_argument('project_dir', type=str, help='Path to the project repository.',
                        default='/home/Arash-Sal-Moslehian/Playground/EPFL/epfl-ml4science/')
    args = parser.parse_args()
    current_dir = args.project_dir
    # Change this accordingly
    data_path = os.path.join(current_dir, 'data', 'gnn_data', 'preproc')
    model_path = os.path.join(current_dir, 'gnn', 'model')
    
    print(model_path)

    torch.manual_seed(7)
    np.random.seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Model Parameters
    ratio = 0.5
    n_roi = 68
    indim = n_roi

    # Adam Optimizer Parameters
    lr = 0.001
    weightdecay = 0.2

    # Learning Rate Parameters
    stepsize = 10
    gamma = 0.5

    # Setting up
    num_epoch = 50
    # Read up how batching is performed on graphs https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html
    batch_size = 32
    kfold = 5
    lamb_ce = 1
    lamb0 = 0.1
    lamb1 = 0.1
    lamb2 = 0.4
    n_class = 2

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Load dataset
    dataset = LCNAData(data_path)
    data_labels = pd.read_csv(os.path.join(data_path, 'data_labels.csv'))

    test_to_valid_ratio = 0.5

    # Since the labels for subjects are not distributed equally, we will create the fold manually.abs
    # In each fold, we make sure to have atleast two subjects that have both labels.
    # We will add whatever is left from subjects with one label to the training set in eahc fold.
    subjects_with_both_labels = [1619, 1623, 1633, 1634, 1635, 1646, 1674, 1675, 2108, 2123]
    subjects_with_one_label = [1620, 1644, 1663, 1669, 1670, 2073, 2081, 2084, 2109, 2110, 2129, 2130, 2174]
    all_subjects = subjects_with_both_labels + subjects_with_one_label
    
    subject_folds = []
    
    for _ in range(kfold):
        two_sub_with_both_labels = np.random.choice(subjects_with_both_labels, 2, replace=False)
        subjects_with_both_labels = list(set(subjects_with_both_labels) - set(two_sub_with_both_labels))
        two_sub_with_one_label = np.random.choice(subjects_with_one_label, 2, replace=False)
        subjects_with_one_label = list(set(subjects_with_one_label) - set(two_sub_with_one_label))
        fold_subjects = np.concatenate([two_sub_with_both_labels, two_sub_with_one_label])
        np.random.shuffle(fold_subjects)
        subject_folds.append(fold_subjects)
    
    
    for fold in range(kfold):

        subjects_in_test_valid = subject_folds[fold]
        subjects_in_train = list(set(all_subjects)-set(subjects_in_test_valid))

        # Get the indices for all the graphs for these subjects
        test_valid_indices = data_labels[data_labels['subject_id'].isin(subjects_in_test_valid)].index.tolist()
        train_indices = data_labels[data_labels['subject_id'].isin(subjects_in_train)].index.tolist()

        test_indices = np.random.choice(test_valid_indices, int(len(test_valid_indices)*test_to_valid_ratio), replace=False)
        valid_indices = list(set(test_valid_indices)-set(test_indices))

        # Use random_split to create train, validation, and test sets
        train_set, val_set, test_set = Subset(dataset, indices=train_indices), Subset(dataset, indices=valid_indices), Subset(dataset, indices=test_indices),

        # Create DataLoader for each set
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        # Initialize the GNN model and print its structure
        model = GNN(indim, ratio).to(device)
        print(model)
        print('Total parameters: ', count_parameters(model))

        # Set up Adam optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weightdecay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=gamma)
        
        # Record losses
        metric_recorder = {
            'epoch': [],
            'train_loss': [],
            'valid_loss': [],
            'train_accuracy': [],
            'valid_accuracy': [],
        }
        
        cross_val_recorder = {
            'test_accuracy': [],
            'test_loss': []
        }

        def train(epoch):
            """
            Train the model for one epoch.

            Parameters:
            - epoch (int): Current epoch number.

            Returns:
            - float: Average loss over the training dataset for the epoch.
            - np.ndarray: Concatenated array of s1 values.
            - np.ndarray: Concatenated array of s2 values.
            - torch.Tensor: Model's w1 parameters.
            - torch.Tensor: Model's w2 parameters.
            """
            for param_group in optimizer.param_groups:
                print("LR: ", param_group['lr'])
            # Set the model to training mode
            model.train()
            # Lists to store s1 and s2 values
            scores_dict = {
                's1': [],
                'p1': [],
            }
            # Variables to store the total loss and step count
            total_loss = 0
            step = 0

            # Iterate over the training data loader
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                # Forward pass through the model
                output, w1, s1, p1 = model(
                    data.x, data.edge_index, data.batch, data.edge_attr)
                # Append s1 and s2 values to the lists
                # s1, p1 have shape [|batch|, (|N|*ratio)], we add blocks (batches) of them
                # to a list and later on vstack them to get an array of shape [total_n_graphs, (|N|*ratio)]
                scores_dict['s1'].append(s1.detach().cpu().numpy())
                scores_dict['p1'].append(p1.detach().cpu().numpy())
                # Calculate the loss components
                loss_c = F.nll_loss(output, data.y)
                loss_p1 = unit_loss(w1)
                loss_tpk1 = topk_loss(s1, ratio)
                loss_consist = 0
                for c in range(n_class):
                    loss_consist += consist_loss(s1[data.y == c], device)
                # loss = classification loss + unit loss + topkpooling loss + Group-level consistency loss
                loss = (lamb_ce * loss_c) + (lamb0 * loss_p1) + (lamb1 * loss_tpk1) + (lamb2 * loss_consist)
                step = step + 1
                # Backward pass and optimization step
                loss.backward()
                total_loss += loss.item() * data.num_graphs
                optimizer.step()

            # Change LR
            scheduler.step()
            # vstack s1, s2, p1, p2 to get the scores and indices for all the graphs.
            # We get an array of shape [total_n_graphs, (|N|*ratio)] for each.
            scores_dict['s1'] = np.vstack(scores_dict['s1'])
            scores_dict['p1'] = np.vstack(scores_dict['p1'])
            # Return average loss and other logged values
            return total_loss / len(train_set), scores_dict, w1

        def evaluate_accuracy(loader):
            """
            Evaluate the model accuracy on a given data loader.

            Parameters:
            - loader: DataLoader for evaluation.

            Returns:
            - float: Accuracy on the evaluation dataset.
            """
            # Set the model to evaluation mode
            model.eval()
            # Variable to store the correct predictions count
            correct = 0
            # Iterate over the evaluation data loader
            for data in loader:
                data = data.to(device)
                # Forward pass through the model
                outputs = model(data.x, data.edge_index, data.batch,
                                data.edge_attr)
                # Get predicted labels
                pred = outputs[0].max(dim=1)[1]
                # Update correct predictions count
                correct += pred.eq(data.y).sum().item()
            # Compute and return accuracy
            return correct / len(loader.dataset)


        def evaluate_loss(loader, epoch):
            """
            Evaluate the model loss on a given data loader.

            Parameters:
            - loader: DataLoader for evaluation.
            - epoch (int): Current epoch number.

            Returns:
            - float: Average loss on the evaluation dataset.
            """
            # Set the model to evaluation mode
            model.eval()
            # Variable to store the total loss
            total_loss = 0
            # Iterate over the evaluation data loader
            for data in loader:
                data = data.to(device)
                # Forward pass through the model
                output, w1, s1, p1 = model(data.x, data.edge_index, data.batch, data.edge_attr)
                # Calculate the loss components
                loss_c = F.nll_loss(output, data.y)
                loss_p1 = unit_loss(w1)
                loss_tpk1 = topk_loss(s1, ratio)
                loss_consist = 0
                for c in range(n_class):
                    loss_consist += consist_loss(s1[data.y == c], device)
                # Combine the loss components with specified weights
                loss = (lamb_ce * loss_c) + (lamb0 * loss_p1) + (lamb1 * loss_tpk1) + (lamb2 * loss_consist)
                # Update total loss
                total_loss += loss.item() * data.num_graphs
            # Return average loss
            return total_loss / len(loader.dataset)

        # Initialize variables for tracking the best model weights and loss
        best_model_weights = copy.deepcopy(model.state_dict())
        best_model_scores = {}
        best_loss = np.inf

        # Iterate through training epochs
        for epoch in range(0, num_epoch):
            # Record the start time of the epoch
            since = time.time()

            # Train the model and retrieve training metrics
            training_loss, scores_dict, w1 = train(epoch)

            # Evaluate training and validation accuracy
            training_accuracy = evaluate_accuracy(train_loader)
            validation_accuracy = evaluate_accuracy(val_loader)

            # Evaluate validation loss
            validation_loss = evaluate_loss(val_loader, epoch)

            # Calculate the time elapsed for the current epoch
            time_elapsed = time.time() - since
            
            # Print epoch summary
            print('---')
            print(f'{time_elapsed // 60}m {time_elapsed % 60}s')
            print(f'Epoch: {epoch}, Train Loss: {training_loss}, Train Acc: {training_accuracy}, Valid Loss: {validation_loss}, Test Valid: {validation_accuracy}')
            print('---')

            # Log metrics

            metric_recorder['epoch'].append(epoch)
            metric_recorder['train_loss'].append(training_loss)
            metric_recorder['valid_loss'].append(validation_loss)
            metric_recorder['train_accuracy'].append(training_accuracy)
            metric_recorder['valid_accuracy'].append(validation_accuracy)

            # Save the best model along with the scores if the validation loss improves
            if validation_loss < best_loss and epoch > 2:
                print("New best model.")
                best_loss = validation_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                best_model_scores = copy.deepcopy(scores_dict)


        # Save the model, scores, and metric on disk.
        torch.save(best_model_weights, os.path.join(model_path, f'best-model-{str(fold)}.pth'))
        np.savez_compressed(os.path.join(model_path, f'scores-{str(fold)}.npz'), **best_model_scores)
        pd.DataFrame(metric_recorder).to_pickle(os.path.join(model_path, f'metrics-{str(fold)}.pkl'))

        # Use the best model weights obtained during training
        model.load_state_dict(best_model_weights)
        model.eval()

        # Evaluate the model on the testing set
        test_accuracy  = evaluate_accuracy(test_loader)
        test_loss = evaluate_loss(test_loader, 0)
        
        cross_val_recorder['test_accuracy'].append(test_accuracy)
        cross_val_recorder['test_loss'].append(test_loss)

        print("---")
        print(f"Test Acc: {test_accuracy}, Test Loss: {test_loss}")

    print("---")
    print(f"Cross-validation: Test Acc: {np.mean(cross_val_recorder['test_accuracy'])}, Test Loss: {np.mean(cross_val_recorder['test_loss'])}")


if __name__ == "__main__":
    main()
