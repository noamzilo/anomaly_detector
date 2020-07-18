from torch import nn
import torch
from torch import optim
import torchvision
import os


class AutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self._device)
        self._input_shape = kwargs["input_shape"]
        self.encoder_hidden_layer = nn.Linear(
            in_features=self._input_shape, out_features=128)
        self.encoder_output_layer = nn.Linear(in_features=128, out_features=128)
        self.decoder_hidden_layer = nn.Linear(in_features=128, out_features=128)
        self.decoder_output_layer = nn.Linear(in_features=128, out_features=kwargs["input_shape"])

        self._criterion = nn.MSELoss()

        self._epoch_loss = 0
        self._path_to_snapshots = "./weights"
        assert os.path.isdir(self._path_to_snapshots)

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
        model = self.to(self._device)
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        train_dataset = torchvision.datasets.MNIST(root="~/torch_datasets", train=True, transform=transform, download=True)

        test_dataset = torchvision.datasets.MNIST(root="~/torch_datasets", train=False, transform=transform, download=True)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

        n_epochs = 20
        for epoch in range(n_epochs):
            self._train(model, epoch, train_loader, optimizer)
            eval_loss = self._eval(model, epoch, test_loader)

            # display the epoch training loss
            print(f"epoch : {epoch + 1}/{n_epochs}, train_loss = {self._epoch_loss:.6f} "
                  f"validation_loss = {eval_loss:.6f}")

    def _train(self, model, epoch, train_loader, optimizer):
        model.train()
        path_to_weights_file = os.path.join(self._path_to_snapshots, str(epoch) + ".weights")
        self._epoch_loss = 0
        for batch_features, _ in train_loader:
            batch_features = batch_features.view(-1, 784).to(self._device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            batch_loss = self._criterion(outputs, batch_features)
            batch_loss.backward()
            optimizer.step()
            self._epoch_loss += batch_loss.item()
        self._epoch_loss = self._epoch_loss / len(train_loader)

        torch.save(model.state_dict(), path_to_weights_file)

    def _eval(self, model, epoch, test_loader):
        model.eval()
        eval_loss = 0
        for batch_features, _ in test_loader:
            batch_features = batch_features.view(-1, 784).to(self._device)
            outputs = model(batch_features)
            batch_loss = self._criterion(outputs, batch_features)
            eval_loss += batch_loss.item()
        eval_loss = eval_loss / len(test_loader)
        return eval_loss

if __name__ == "__main__":
    ae = AutoEncoder(input_shape=784)
    ae.check_mnist()
