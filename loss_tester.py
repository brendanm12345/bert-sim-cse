import torch
import torch.nn.functional as F

class SimCSEModel:
    def __init__(self, temperature=0.05):
        self.temperature = temperature

    def forward(self, embeddings):
        # Normalize the embeddings to have unit length
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Calculate cosine similarity matrix (size: batch_size x batch_size)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # For numerical stability, subtract the maximum value in each row
        max_vals = torch.max(similarity_matrix, dim=1, keepdim=True)[0]
        similarity_matrix = similarity_matrix - max_vals.detach()
        
        # The diagonal entries are the similarities between each embedding and itself
        # We want to exclude these when calculating the denominator in softmax
        logits_mask = torch.eye(similarity_matrix.size(0)).bool().to(embeddings.device)
        similarity_matrix.masked_fill_(logits_mask, float('-inf'))
        
        # Calculate softmax along each row, but exclude the diagonal (self-similarity)
        softmax_scores = F.softmax(similarity_matrix, dim=1)
        
        # The targets are the positions of the positive examples in the softmax_scores matrix
        # For unsupervised SimCSE, the positive example for each anchor is its pair (next in sequence)
        targets = torch.arange(embeddings.size(0)).to(embeddings.device)
        if embeddings.size(0) % 2 == 0:  # Assuming even batch size for simplicity
            targets = (targets + 1) - 2 * (targets % 2)
        
        # Calculate the log likelihood of the positive examples
        log_probs = torch.log(torch.gather(softmax_scores, 1, targets.unsqueeze(1)).squeeze(1))
        
        # The loss is the negative log likelihood averaged across the batch
        loss = -log_probs.mean()
        
        return loss

# Stack all embeddings into a single tensor
embeddings = torch.stack([
    torch.tensor([1, 2, 3], dtype=torch.float), 
    torch.tensor([1, 2, 3], dtype=torch.float), 
    torch.tensor([4, 5, 6], dtype=torch.float), 
    torch.tensor([4, 5, 6], dtype=torch.float), 
    torch.tensor([7, 8, 9], dtype=torch.float), 
    torch.tensor([7, 8, 9], dtype=torch.float)
])

# Create an instance of the model
model = SimCSEModel()

# Call the forward function with the embeddings
loss = model.forward(embeddings)
print(loss)
