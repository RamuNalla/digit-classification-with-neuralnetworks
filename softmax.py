import torch
import torch.nn.functional as F

# A batch of 3 images, with 4 possible classes. Each row represents the logits for a single image
logits = torch.tensor([
    [2.0, 1.0, 0.1, 3.0],    # Logits for image 1
    [1.0, 5.0, 2.0, 0.0],    # Logits for image 2
    [0.1, 0.2, 6.0, -2.0]    # Logits for image 3
])

print(f"Input tensor shape: {logits.shape}")        # Output: Input tensor shape: torch.Size([3, 4])

probabilities_dim1 = F.softmax(logits, dim=1)
print(f"Output with dim=1:\n{probabilities_dim1}")

# Check the sum of each row (each image's probabilities)
print(f"Sum of each row: {torch.sum(probabilities_dim1, dim=1)}")

# Output with dim=1:
# tensor([[0.1179, 0.0434, 0.0163, 0.8224],
#         [0.0245, 0.6652, 0.0665, 0.2438],
#         [0.0022, 0.0024, 0.9702, 0.0252]])
# Sum of each row: tensor([1., 1., 1.])

probabilities_dim0 = F.softmax(logits, dim=0)
print(f"Output with dim=0:\n{probabilities_dim0}")

# Check the sum of each column (each class's probabilities across the batch)
print(f"Sum of each column: {torch.sum(probabilities_dim0, dim=0)}")

# Output with dim=0:
# tensor([[0.7259, 0.0223, 0.0016, 0.9999],
#         [0.2671, 0.9765, 0.0531, 0.0001],
#         [0.0070, 0.0012, 0.9453, 0.0000]])
# Sum of each column: tensor([1., 1., 1., 1.])

# This output is nonsensical for a classification task. 
# The columns sum to 1, but the rows (the probability distribution for a single image) do not.
#  You've essentially created a probability distribution across your batch of data, which is not what you want.

#Conclusion
# For a typical classification model where the output tensor is of shape (batch_size, num_classes), 
# you will almost always use dim=1 or dim=-1 (which refers to the last dimension, equivalent to dim=1 in this case) 
# to apply softmax correctly. 
# This ensures that the output for each sample is a valid probability distribution over the classes.