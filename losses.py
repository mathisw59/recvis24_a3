import torch
import torch.nn.functional as F
from torch import nn

from yaml_parser import CriterionConfig

class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    def forward(self, output, target=None):
        batch_size = output.size(0)
        z_i, z_j = output[:, 0], output[:, 1]

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        z = torch.cat([z_i, z_j], dim=0)

        # Compute similarity matrix
        similarity_matrix = torch.mm(z, z.T) / self.temperature

        # Positive pairs
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji])

        # Create mask for negative pairs (excluding self-similarities and positive pairs)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        # Add positive pairs to mask
        mask[range(batch_size), range(batch_size, 2*batch_size)] = True
        mask[range(batch_size, 2*batch_size), range(batch_size)] = True
        
        # Apply mask to similarity matrix
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # For each sample, gather all its negative pairs
        negatives = similarity_matrix.view(2 * batch_size, -1)

        # Combine positive and negative similarities
        logits = torch.cat([positives.view(2 * batch_size, 1), negatives], dim=1)

        # Labels: positive pair is at index 0
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)

        # Compute cross entropy loss
        loss = F.cross_entropy(logits, labels)

        acc = (logits.argmax(dim=1) == labels).float().mean()

        return acc, loss # Self supervised losses should also return self supervised accuracy
    
def get_criterion(config: CriterionConfig):
    if config.name == "simclr":
        return SimCLRLoss(config.params['temperature'])
    elif config.name == "crossentropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f'Criterion {config.name} not supported')