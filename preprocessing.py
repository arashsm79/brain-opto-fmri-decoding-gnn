import nibabel as nib
import nilearn as nil
from nilearn import plotting, connectome, image, maskers
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
from networkx.convert_matrix import from_numpy_array
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
from tqdm import tqdm
import torch as torch
import re
from LCNAData import LCNAData
import argparse

parser = argparse.ArgumentParser(description='Path to the project directory.')
parser.add_argument('project_dir', type=str, help='Path to the project repository.',
                    default='/home/Arash-Sal-Moslehian/Playground/EPFL/epfl-ml4science/')
args = parser.parse_args()
current_dir = args.project_dir

data_path = os.path.join(current_dir, 'data', 'dataset')
stuff_path = os.path.join(current_dir, 'data', 'dataset', 'OTHER_STUFF')
preproc_path = os.path.join(current_dir, 'data', 'gnn_data', 'preproc')

# Create directories if they don't exist
if not os.path.exists(preproc_path):
    os.makedirs(preproc_path)

stim_start_ts = [480, 540, 600, 660, 720, 780, 840, 900]

# We extract the time series for each stimulation period (60s)
stim_duration = 60
protocols = ['3Hz', '15Hz']

# Sample each slice this many times
num_samples = 30

# The mean time series for each node is derived from a random subset comprising one-third of the voxels within the ROI. This random sampling is performed 30 times, resulting in 30 graphs for each stimulation instance.
fraction_to_sample = 1/3

epi_template = nib.load(os.path.join(stuff_path, 'EPI_template.nii.gz'))

# Brain image parcellation into Regions of Interest (ROIs) is accomplished using an atlas.
atlas_img = nib.load(os.path.join(stuff_path, 'EPI_template_2021_200um_parcellation_VZe_RL_thr.nii.gz'))
atlas_img_arr = nib.load(os.path.join(stuff_path, 'EPI_template_2021_200um_parcellation_VZe_RL_thr.nii.gz')).get_fdata()

# Remove all the labels with less than min_voxel voxels
min_label_voxel = 43
unique_values, counts = np.unique(atlas_img_arr, return_counts=True)
filtered_values = unique_values[counts <= min_label_voxel]
atlas_img_arr[np.isin(atlas_img_arr, filtered_values)] = 0
np.savez_compressed(os.path.join(preproc_path, 'EPI_label_to_node_mapping.npz'), epi_labels=np.unique(atlas_img_arr))


metadata_dict = {
    'subject_id': [],
    'filename': [],
    'stim_id': [],
    'sample_id': [],
    'label': [],
    'timeseries': []
}

# Go through each protocol's folder
for protocol_label, protocol in enumerate(protocols):
    # Go through each file in that folder (a Nifti image)
    for filename in tqdm(os.listdir(os.path.join(data_path, protocol))):
        if filename.endswith('.nii.gz'):
            fmri_img = nib.load(os.path.join(data_path, protocol, filename))
            fmri_img_arr = fmri_img.get_fdata()

            # Get the subject id form the filename
            match = re.search(r'sub_(\d+)', filename)
            subject_id = None
            if match:
                subject_id = match.group(1)
            else:
                print("Not a valid filename. Filename must be like: filtered_func_data_EPI_sub_1634_15Hz_ds_BP")
                continue
            

            # Cut out each stimulation period and derive the timeseries from it
            for stim_id, stim_start in enumerate(stim_start_ts):
                stim_slice = fmri_img_arr[:, :, :, stim_start:stim_start+stim_duration]
                
                # Generate num_samples graphs
                for sample_id in range(num_samples):
                    random_masks = np.zeros_like(atlas_img_arr, dtype=bool)
                    # Iterate over each unique label in the ROI
                    for label in np.unique(atlas_img_arr):
                        if label == 0:
                            continue
                        # Create a random mask for the current label
                        label_mask = (atlas_img_arr == label)
                        num_voxels_in_label = np.sum(label_mask)
                        # Calculate the number of voxels to sample (1/3 of the voxels)
                        num_voxels_to_sample = int(fraction_to_sample * num_voxels_in_label)
                        # Get flattened indices of the current label's voxels
                        flat_indices = np.where(label_mask.flatten())[0]
                        # Randomly sample 1/3 of the voxels within the current label
                        sampled_indices = np.random.choice(flat_indices, size=num_voxels_to_sample, replace=False)
                        # Convert flat indices back to 3D indices
                        sampled_voxels = np.unravel_index(sampled_indices, label_mask.shape)
                        # Update the random masks
                        random_masks[sampled_voxels] = True

                    # Apply the random masks to the labels image
                    masked_roi_labels_img = nib.Nifti1Image(np.where(random_masks, atlas_img_arr, 0), affine=atlas_img.affine)
                    # Update the masker with the new labels
                    masker = maskers.NiftiLabelsMasker(labels_img=masked_roi_labels_img)
                    # Extract the mean time series
                    stim_slice_img = nib.Nifti1Image(stim_slice, affine=fmri_img.affine)
                    mean_time_series = masker.fit_transform(stim_slice_img)
                    # plotting.plot_img(nil.image.mean_img(stim_slice_img), cmap='gray').add_overlay(masked_roi_labels_img, alpha=0.5)
                    
                    metadata_dict['subject_id'].append(subject_id)
                    metadata_dict['filename'].append(filename)
                    metadata_dict['stim_id'].append(stim_id)
                    metadata_dict['sample_id'].append(sample_id)
                    metadata_dict['label'].append(protocol_label)
                    metadata_dict['timeseries'].append(mean_time_series)

# Save everything into a pickle file for connectivity generation
preproc_df = pd.DataFrame(metadata_dict)
preproc_df.to_pickle(os.path.join(preproc_path, 'timeseries.pkl'))

# Save the labels for the Dataloader
preproc_df.drop(columns=['timeseries']).to_csv(os.path.join(preproc_path, 'data_labels.csv'), index=False)

preproc_df = pd.read_pickle(os.path.join(preproc_path, 'timeseries.pkl'))
all_sample_timeseries = preproc_df['timeseries']
corr_list = connectome.ConnectivityMeasure(kind='correlation', standardize='zscore_sample').fit_transform(all_sample_timeseries)
parcorr_list = connectome.ConnectivityMeasure(kind='partial correlation', standardize='zscore_sample').fit_transform(all_sample_timeseries)

# Create the graphs and save them to disk
# https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs
# Iterate over rows and their indices in the preprocessed dataframe
for preproc_row_idx, preproc_row in tqdm(enumerate(preproc_df.iterrows())):
    
    preproc_row = preproc_row[1]

    parcorr = parcorr_list[preproc_row_idx]
    corr = corr_list[preproc_row_idx]
    
    # So we don't get infinity in arctanh
    np.fill_diagonal(parcorr, 0.9999)
    np.fill_diagonal(corr, 0.9999)

    # arctanh is a kind of normalization (z-Fisher) for connectivity matrices
    parcorr = np.arctanh(parcorr)
    corr = np.arctanh(corr)

    # Get the number of nodes from the matrix 'parcorr' at the current index
    num_nodes = parcorr.shape[0]

    # Convert a NumPy array to a NetworkX graph
    G = from_numpy_array(parcorr)

    # Calculate the threshold for the top 10 percent positive partial correlations
    threshold = np.percentile(np.abs(parcorr), 90)
    # Get edges with weights below the threshold
    edges_to_remove = [(u, v) for u, v, data in G.edges(data=True) if abs(data['weight']) < threshold]
    # Remove edges from the graph
    G.remove_edges_from(edges_to_remove)
    # Add the largest edge to each isolated node
    isolated_nodes = [node for node, degree in G.degree() if degree == 0]
    if len(isolated_nodes) > 0:
        print('Found isolated nodes after prunning. Adding back the largest edge.')
    for node in isolated_nodes:
        # Find the index of the largest edge in the original parcorr matrix for the isolated node
        max_edge_index = np.unravel_index(np.argmax(parcorr[node]), parcorr.shape)
        # Add the largest edge to the mask
        G.add_edge(max_edge_index[0], max_edge_index[1], weight=parcorr[max_edge_index[0], max_edge_index[1]])


    # Convert the graph to a scipy sparse matrix
    A = nx.to_scipy_sparse_array(G)
    # Convert the sparse matrix to a COO (coordinate) format. This is NOT a typical adjacency matrix.
    adj = A.tocoo()
    
    # Initialize an array for edge attributes
    edge_att = np.zeros(len(adj.row))
    # Populate edge_att with values from 'parcorr' matrix
    for sample_id in range(len(adj.row)):
        edge_att[sample_id] = parcorr[adj.row[sample_id], adj.col[sample_id]]
    # Stack row and column indices to create edge_index
    edge_index = np.stack([adj.row, adj.col])
    # Remove self-loops from edge_index and edge_att
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    # Convert edge_index to long type
    edge_index = edge_index.long()
    # Coalesce duplicate entries in edge_index and edge_att
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes, num_nodes)

    # Convert correlation values to a PyTorch tensor
    att_torch = torch.from_numpy(corr_list[preproc_row_idx]).float()
    # Convert label to a PyTorch tensor for classification
    y_torch = torch.from_numpy(np.array(preproc_row['label'])).long()

    # Create a PyTorch Data object with node features, edge indices, labels, and edge attributes
    data = Data(x=att_torch, edge_index=edge_index.long(), y=y_torch, edge_attr=edge_att)
    
    # Extract metadata from the row
    subject_id = preproc_row['subject_id']
    sample_id = preproc_row['sample_id']
    stim_id = preproc_row['stim_id']
    protocol_label = preproc_row['label']
    
    # Create directories if they don't exist
    if not os.path.exists(os.path.join(preproc_path, subject_id)):
        os.makedirs(os.path.join(preproc_path, subject_id))
    
    # Save the PyTorch Data object to a file
    torch.save(data, os.path.join(preproc_path, subject_id, f'sub{subject_id}_prot{protocol_label}_stim{stim_id}_sample{sample_id}.pt'))


                        
# Validate the data
dataset = LCNAData(preproc_path)
for data in dataset:
    data.validate(raise_on_error=True)
    print(data)
