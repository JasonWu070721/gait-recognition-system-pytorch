from torchvision.datasets import MNIST, CIFAR100
from torchvision import transforms
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from datasets import TripletMNIST, TripletCIFAR100
from trainer import fit
from networks import EmbeddingNet, TripletNet
from losses import TripletLoss
from torchsummary import summary
from torchvision import models


def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10, 10))
    for i in range(10):
        inds = np.where(targets == i)[0]
        plt.scatter(
            embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[i]
        )
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k : k + len(images)] = (
                model.get_embedding(images).data.cpu().numpy()
            )
            labels[k : k + len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels


if __name__ == "__main__":
    print(torch.__version__)

    mean, std = 0.1307, 0.3081

    train_dataset = CIFAR100(
        "../data/CIFAR100",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((mean,), (std,))]
        ),
    )
    test_dataset = CIFAR100(
        "../data/CIFAR100",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((mean,), (std,))]
        ),
    )
    n_classes = 10

    cuda = torch.cuda.is_available()
    print(cuda)

    triplet_train_dataset = TripletCIFAR100(train_dataset)  # Returns triplets of images
    triplet_test_dataset = TripletCIFAR100(test_dataset)
    batch_size = 128
    kwargs = {"num_workers": 1, "pin_memory": True} if cuda else {}
    triplet_train_loader = torch.utils.data.DataLoader(
        triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    triplet_test_loader = torch.utils.data.DataLoader(
        triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs
    )

    margin = 1.0
    embedding_net = EmbeddingNet()
    summary(embedding_net.cuda(), (3, 32, 32))

    model = TripletNet(embedding_net)
    if cuda:
        model.cuda()
    loss_fn = TripletLoss(margin)
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 5
    log_interval = 100

    fit(
        triplet_train_loader,
        triplet_test_loader,
        model,
        loss_fn,
        optimizer,
        scheduler,
        n_epochs,
        cuda,
        log_interval,
    )
