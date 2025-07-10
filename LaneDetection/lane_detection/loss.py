import math
import torch
import torch.nn as nn

def euclidean_distance(p1, p2):
    return torch.norm(p1 - p2)

def gps_pairwise_distance(P, Q):
    """
    Pairwise GPS distance between P and Q using equirectangular approximation.

    Args:
        P: (m, 2) tensor of [lat, lon]
        Q: (n, 2) tensor of [lat, lon]

    Returns:
        D: (m, n) tensor of distances in meters
    """
    lat1 = P[:, 0].unsqueeze(1)  # (m, 1)
    lon1 = P[:, 1].unsqueeze(1)  # (m, 1)
    lat2 = Q[:, 0].unsqueeze(0)  # (1, n)
    lon2 = Q[:, 1].unsqueeze(0)  # (1, n)

    avg_lat = ((lat1 + lat2) / 2) * math.pi / 180
    dx = (lon2 - lon1) * 111_320 * torch.cos(avg_lat)
    dy = (lat2 - lat1) * 111_320
    dist = torch.sqrt(dx ** 2 + dy ** 2)
    return dist  # (m, n)


def frechet_distance(P, Q):
    """
    Differentiable approximation of discrete Frechet distance between two curves P and Q.
    Args:
        P: (m, 2) torch.Tensor
        Q: (n, 2) torch.Tensor
    Returns:
        Scalar tensor: approximate Frechet distance
    """
    m, n = P.size(0), Q.size(0)
    # print(f"Frechet P: {P}, Q: {Q}")
    device = P.device

    # Compute full pairwise distance matrix
    # D = torch.cdist(P.unsqueeze(0), Q.unsqueeze(0), p=2).squeeze(0)  # (m, n)
    D = gps_pairwise_distance(P, Q)  # (m, n)

    # Dynamic programming: fill a table of same shape
    path = torch.zeros_like(D)
    path[0, 0] = D[0, 0]

    for i in range(1, m):
        path[i, 0] = torch.max(path[i - 1, 0], D[i, 0])
    for j in range(1, n):
        path[0, j] = torch.max(path[0, j - 1], D[0, j])

    for i in range(1, m):  # i from 1 to m-1
        for j in range(1, n):  # j from 1 to n-1
            prev = torch.stack([
                path[i - 1, j],
                path[i - 1, j - 1],
                path[i, j - 1]
            ])
            path[i, j] = torch.max(torch.min(prev), D[i, j])

    return path[m - 1, n - 1]


class LaneTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(LaneTripletLoss, self).__init__()
        self.margin = margin

    def gps_distance(self, A, B):
        """
        Computes equirectangular approximation distance between two (N, 2) lat/lon tensors.
        """
        lat1, lon1 = A[:, 0], A[:, 1]
        lat2, lon2 = B[:, 0], B[:, 1]
        avg_lat = torch.deg2rad((lat1 + lat2) / 2)
        dx = (lon2 - lon1) * 111_320 * torch.cos(avg_lat)
        dy = (lat2 - lat1) * 111_320
        return torch.sqrt(dx ** 2 + dy ** 2) # shape: (N,)
    
    def forward(self, anchor, positive, negative):
        # anchor, positive: (N, D)
        # negative: (N, D) or (N, K, D)

        # Ensure inputs are tensors with gradients
        if not isinstance(anchor, torch.Tensor):
            anchor = torch.tensor(anchor, dtype=torch.float32)
        if not isinstance(positive, torch.Tensor):
            positive = torch.tensor(positive, dtype=torch.float32)
        if not isinstance(negative, torch.Tensor):
            negative = torch.tensor(negative, dtype=torch.float32)

        # Euclidean distances
        # pos_dist = torch.norm(anchor - positive, dim=1) # (N,)
        pos_dist = self.gps_distance(anchor, positive)

        # Pairwise distances to all negatives
        if negative.dim() == 2:
            # neg_dist = torch.norm(anchor - negative, dim=1) # (N,)
            neg_dist = self.gps_distance(anchor, negative)
        else:
            # (N, K, D) --> (N, K)
            # neg_dist = torch.norm(anchor.unsqueeze(1) - negative, dim=2)
            # neg_dist = torch.min(neg_dist, dim=1).values # (N,)
            anchor_exp = anchor.expand_as(negative)  # (N, K, 2)
            lat1, lon1 = anchor_exp[:, :, 0], anchor_exp[:, :, 1]
            lat2, lon2 = negative[:, :, 0], negative[:, :, 1]
            avg_lat = torch.deg2rad((lat1 + lat2) / 2)
            dx = (lon2 - lon1) * 111_320 * torch.cos(avg_lat)
            dy = (lat2 - lat1) * 111_320
            dist = torch.sqrt(dx ** 2 + dy ** 2)  # (N, K)
            neg_dist = torch.min(dist, dim=1).values

        losses = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return torch.mean(losses)

def ReLU(x):
    return x * (x > 0)

