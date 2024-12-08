import torch
import torch.optim as optim
from torchsummary import summary
from torchvision import datasets, transforms
import torch.nn.functional as F
from tqdm import tqdm
from models import MNIST_CNN_M2

class train_M2():
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []

    def count_parameters(self, model):
        """
        Calculate the total number of parameters in a PyTorch model.

        Args:
            model (torch.nn.Module): The PyTorch model.

        Returns:
            int: The total number of parameters.
            int: The number of trainable parameters.
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def show_data_stats(self, train, train_loader):
        # We'd need to convert it into Numpy! Remember above we have converted it into tensors already
        train_data = train.data
        train_data = train.transform(train_data.numpy())

        print('[Train]')
        print(' - Numpy Shape:', train.train_data.cpu().numpy().shape)
        print(' - Tensor Shape:', train.train_data.size())
        print(' - min:', torch.min(train_data))
        print(' - max:', torch.max(train_data))
        print(' - mean:', torch.mean(train_data))
        print(' - std:', torch.std(train_data))
        print(' - var:', torch.var(train_data))

        dataiter = iter(train_loader)
        images, labels = next(dataiter)

        print('\n[Batch Stats]')
        print(images.shape)
        print(labels.shape)
    

    def load_data(self):
        # Train Phase transformations
        train_transforms = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                            ])

        # Test Phase transformations
        test_transforms = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                            ])
        
        ## MNIST - Train and Test Datasets  
        train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
        test = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)

        SEED = 1

        # CUDA?
        cuda = torch.cuda.is_available()
        print("CUDA Available?", cuda)

        # For reproducibility
        torch.manual_seed(SEED)

        if cuda:
            torch.cuda.manual_seed(SEED)

        # dataloader arguments - something you'll fetch these from cmdprmt
        dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

        # train dataloader
        train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

        # test dataloader
        test_loader = torch.utils.data.DataLoader(test, **dataloader_args)

        return train, train_loader, test_loader


    def train(self, model, device, train_loader, optimizer):
        model.train()
        pbar = tqdm(train_loader)
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            # get samples
            data, target = data.to(device), target.to(device)

            # Init
            optimizer.zero_grad()
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            y_pred = model(data)

            # Calculate loss
            loss = F.nll_loss(y_pred, target)
            self.train_losses.append(loss)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update pbar-tqdm
            
            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)

    def test(self, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        self.test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
        self.test_acc.append(100. * correct / len(test_loader.dataset))

    
    def run(self):
        ## Data Loading
        train, train_loader, test_loader = self.load_data()
        self.show_data_stats(train, train_loader)

        ## Device Configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        ## Model description
        model = MNIST_CNN_M2().to(device)
        summary(model, input_size=(1, 28, 28))
        total_params, trainable_params = self.count_parameters(model)
        print(f"Total Parameters: {total_params}, Trainable Parameters: {trainable_params}")
        
        ## Optimizer
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # ## Training
        EPOCHS = 15
        for epoch in range(EPOCHS):
            print("EPOCH:", epoch)
            self.train(model, device, train_loader, optimizer)
            self.test(model, device, test_loader)

        print("Training complete")


if __name__ == "__main__":
    trainer = train_M2()
    trainer.run()
