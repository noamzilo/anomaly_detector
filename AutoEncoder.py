from torch import nn
import torch
from torch import optim
import torchvision
import os


class AutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128)
        self.encoder_output_layer = nn.Linear(in_features=128, out_features=128)
        self.decoder_hidden_layer = nn.Linear(in_features=128, out_features=128)
        self.decoder_output_layer = nn.Linear(in_features=128, out_features=kwargs["input_shape"])

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed

    def check_mnist(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        model = AutoEncoder(input_shape=784).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        train_dataset = torchvision.datasets.MNIST(root="~/torch_datasets", train=True, transform=transform, download=True)

        test_dataset = torchvision.datasets.MNIST(root="~/torch_datasets", train=False, transform=transform, download=True)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

        n_epochs = 20
        generic_path_to_epoch_weights = "./weights"
        assert os.path.isdir(generic_path_to_epoch_weights)
        for epoch in range(n_epochs):
            path_to_weights_file = os.path.join(generic_path_to_epoch_weights, str(epoch) + ".weights")
            loss = 0
            for batch_features, _ in train_loader:
                loss = self._train(batch_features, criterion, device, loss, model, optimizer)


            # compute the epoch training loss
            loss = loss / len(train_loader)
            torch.save(model.state_dict(), path_to_weights_file)

            # display the epoch training loss
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, n_epochs, loss))

    def _train(self, batch_features, criterion, device, loss, model, optimizer):
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        batch_features = batch_features.view(-1, 784).to(device)
        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()
        # compute reconstructions
        outputs = model(batch_features)
        # compute training reconstruction loss
        train_loss = criterion(outputs, batch_features)
        # compute accumulated gradients
        train_loss.backward()
        # perform parameter update based on current gradients
        optimizer.step()
        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()
        return loss


if __name__ == "__main__":
    ae = AutoEncoder()
    ae.check_mnist()
