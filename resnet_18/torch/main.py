import argparse
import time
from torchvision import models
from torch.optim import Adam
import torch
import torch.nn as nn
from dataset import get_cifar10

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--seed", type=int, default=0, help="random seed")
args = parser.parse_args()

torch.manual_seed(args.seed)

device = torch.device("cpu")

def eval_fn(model, inp, tgt):
    with torch.no_grad():
        outputs = model(inp)
        _, preds = torch.max(outputs, dim=1)
        return torch.mean((preds == tgt).float()).item()

def train_epoch(model, train_loader, optimizer, criterion, epoch):
    model.train()
    losses = []
    accs = []
    throughput = []

    for batch_counter, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        start_time = time.perf_counter()
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        end_time = time.perf_counter()

        _, preds = torch.max(outputs, dim=1)
        acc = torch.mean((preds == y).float()).item()
        losses.append(loss.item())
        accs.append(acc)

        throughput.append(x.size(0) / (end_time - start_time))

        if batch_counter % 10 == 0:
            print(
                " | ".join(
                    [
                        f"Epoch {epoch:02d} [{batch_counter:03d}]",
                        f"Train loss: {loss.item():.3f}",
                        f"Train acc: {acc:.3f}",
                        f"Throughput: {throughput[-1]:.2f} images/sec",
                    ]
                )
            )

    mean_loss = sum(losses) / len(losses)
    mean_acc = sum(accs) / len(accs)
    mean_throughput = sum(throughput) / len(throughput)
    return mean_loss, mean_acc, mean_throughput

def test_epoch(model, test_loader):
    model.eval()
    accs = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            acc = eval_fn(model, x, y)
            accs.append(acc)
    mean_acc = sum(accs) / len(accs)
    return mean_acc

def main(args):
    
    model = models.resnet18(pretrained=False, num_classes=10).to(device)
    print(f"Number of params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.4f} M")

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = get_cifar10(args.batch_size)

    start_time = time.time()
    print(f"Start time: {start_time}")

    for epoch in range(args.epochs):
        tr_loss, tr_acc, throughput = train_epoch(model, train_loader, optimizer, criterion, epoch)
        print(
            " | ".join(
                [
                    f"Epoch: {epoch}",
                    f"avg. Train loss: {tr_loss:.3f}",
                    f"avg. Train acc: {tr_acc:.3f}",
                    f"Throughput: {throughput:.2f} images/sec",
                ]
            )
        )

        test_acc = test_epoch(model, test_loader)
        print(f"Epoch: {epoch} | Test acc: {test_acc:.3f}")

    end_time = time.time()
    print(f"End time: {end_time}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main(args)