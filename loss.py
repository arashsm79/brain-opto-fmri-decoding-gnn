
import torch

EPS = 1e-15

def unit_loss(w):
    """
    Calculate the unit loss for a given vector 'w'.

    Parameters:
    - w (torch.Tensor): Input vector for which the unit loss is computed.

    Returns:
    - torch.Tensor: The unit loss, which is the L2 norm (Euclidean norm) of the input vector
      minus 1. The unit loss measures the deviation of the vector from being a unit vector
      (having a norm of 1). It penalizes vectors that do not have a norm of 1.
    """
    return (torch.norm(w, p=2) - 1)


def topk_loss(s, ratio):
    """
    Compute the top-k loss for a given tensor s.

    Parameters:
    - s (torch.Tensor): Input tensor.
    - ratio (float): Ratio used for top-k pooling.

    Returns:
    - torch.Tensor: Computed top-k loss.
    """
    # Adjust the ratio if it exceeds 0.5
    if ratio > 0.5:
        ratio = 1 - ratio
    
    # Sort the input tensor along dimension 1
    s = s.sort(dim=1).values
    
    # Compute the top-k loss using negative logarithms
    res = -torch.log(s[:, -int(s.size(1) * ratio):] + EPS).mean() - \
          torch.log(1 - s[:, :int(s.size(1) * ratio)] + EPS).mean()
    
    return res


def consist_loss(s, device):
    """
    Compute the consistency loss for a given tensor s.

    Parameters:
    - s (torch.Tensor): Input tensor.

    Returns:
    - torch.Tensor: Computed consistency loss.
    """
    # Return 0 if the input tensor is empty
    if len(s) == 0:
        return 0
    
    # Apply sigmoid activation to the input tensor
    s = torch.sigmoid(s)
    
    # Create a weight matrix W and a diagonal matrix D
    W = torch.ones(s.shape[0], s.shape[0])
    D = torch.eye(s.shape[0]) * torch.sum(W, dim=1)
    
    # Compute the Laplacian matrix L
    L = D - W
    L = L.to(device)
    
    # Compute the consistency loss using the Laplacian matrix
    res = torch.trace(torch.transpose(s, 0, 1) @ L @ s) / (s.shape[0] * s.shape[0])
    
    return res