import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# hyper parameter
batch_size = 128
lr = 0.01
epoch = 40

# dataset
train_data = datasets.CIFAR10('./data/cifar10', train=True, transform=train_transform, download=True)
train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = datasets.CIFAR10('./data/cifar10', train=False, transform=test_transform, download=True)
test = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# model
model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=False)
model.classifier._modules['6'] = torch.nn.Linear(in_features=4096, out_features=10)

if torch.cuda.is_available():
    model = model.cuda()

summary(model, (3, 32, 32))

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()

valid_loss_min = np.Inf

H = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

for i in range(epoch):
    print('*' * 10, 'epoch {}'.format(i + 1), '*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    model.train()
    for data, label in tqdm(train):
        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, label)
        running_loss += loss.item() * label.size(0)
        _, pred = out.max(1)
        running_acc += (pred == label).sum().item()
        loss.backward()
        optimizer.step()
    valid_loss = 0.0
    valid_acc = 0.0
    model.eval()
    for x, y in tqdm(test):
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        out = model(x)
        loss = loss_fn(out, y)
        valid_loss += loss.item() * y.size(0)
        _, pred = out.max(1)
        valid_acc += (pred == y).sum().item()
    running_loss = running_loss / len(train.dataset)
    running_acc = running_acc / len(train.dataset) * 100
    valid_loss = valid_loss / len(test.dataset)
    valid_acc = valid_acc / len(test.dataset) * 100
    H['train_loss'].append(running_loss)
    H['train_acc'].append(running_acc)
    H['val_loss'].append(valid_loss)
    H['val_acc'].append(valid_acc)
    print('Finish {} epoch, loss={:.6f}, acc={:.6f}'.format(i + 1, running_loss, running_acc))
    print('Validation, loss={:.6f}, acc={:.6f}'.format(valid_loss, valid_acc))
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(valid_loss_min, valid_loss))
        torch.save(model.state_dict(), 'vgg16_cifar10.pth')
        valid_loss_min = valid_loss

print('Finished Training')
print('==> Saving model..')
torch.save(model, 'final.pt')
print('Finished Saving')
print('==> Make Graph..')
plt.style.use("ggplot")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, sharey = False)
ax1.plot(H["train_loss"], label="train_loss")
ax1.plot(H["val_loss"], label="val_loss")
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend(loc='right')
ax2.plot(H["train_acc"], label="train_acc")
ax2.plot(H["val_acc"], label="val_acc")
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend(loc='right')
fig.savefig('curve.png')
print('Finished Making')
