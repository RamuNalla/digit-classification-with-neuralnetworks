import torch
import torch.nn as nn
import torch.optim as optim
from model import simpleNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

INPUT_SIZE = 784
HIDDEN_SIZE = 128
NUM_CLASSES = 10
NUM_EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


model = simpleNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES)

adam = optim.Adam(model.parameters(), lr = LEARNING_RATE)

loss_func = nn.CrossEntropyLoss()           # This combines LogSoftmax and NLLLoss (negative log likelihood loss)

for epoch in range(NUM_EPOCHS):
    model.train()                       # set the model to training mode

    for i, (images, labels) in enumerate(train_loader):     # labels has (batch_size x num_classes), images - (BATCH_SIZE, 1, 28, 28) 1: color channels (part of MNIST dataset)

        outputs = model(images)         # forward pass
        loss = loss_func(outputs, labels)       # calculates loss by crossentropy loss (softmax with dim=1 and then negative log likelihood)

        adam.zero_grad()                # cleans old gradients from previous steps (set all gradients of model's parameters to zero)
        loss.backward()                 # computes the gradients (for each parameter, it populates a .grad attribute with the computed gradient.  For example, model.fc1.weight.grad will now contain the gradient for the weights of the first linear layer)                 
        adam.step()                     # updates the model parameters using computed gradients (using the Adam algorithm): In simple terms: w = w - learning_rate * (updated_gradient)

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        

model.eval()                            # sets in evaluation mode
with torch.no_grad():                     # saves memory and speeds up the process as no need for backprop during evaluation
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)         # batch_size x num_classes

        _, predicted = torch.max(outputs, 1)        # dim = 1 means maximum of all columns in each row        
        # outputs = tensor([
        #     [ -2.3, 1.5, -0.1, 4.2, -0.8, ... ],  # Scores for image 1
        #     [  1.0, 5.0, 2.0, 0.0, -1.2, ... ],  # Scores for image 2
        #     [  0.1, 0.2, 6.0, -2.0, 0.5, ... ]   # Scores for image 3
        # ])

        # (values=tensor([4.2, 5.0, 6.0]), indices=tensor([3, 1, 2])) this is the output of torch.max, we need only indices

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100*correct/total
    print(f'Test Accuracy of the model on the 10000 test images: {accuracy:.2f}%')

torch.save(model.state_dict(), 'simple_nn_baseline.pth')
print("Model saved to simple_nn_baseline.pth")







