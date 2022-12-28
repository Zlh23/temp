import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from read_image import ImageNetData

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(46656, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 45)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 46656)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_info(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x


class PCA:
    def __init__(self, output_dim) -> None:
        self.output_dim = output_dim

    def fit(self, X_data):
        N = len(X_data)
        H = torch.eye(n=N,device="cuda") - 1 / N * (torch.matmul(torch.ones(size=(N, 1),device="cuda"), torch.ones(size=(1, N),device="cuda")))
        X_data = torch.matmul(H, X_data)
        _, _, v = torch.svd(X_data)
        self.base = v[:, :self.output_dim]

    def fit_transform(self, X_data):
        self.fit(X_data)
        return self.transform(X_data)

    def transform(self, X_data):
        return torch.matmul(X_data, self.base)

    def inverse_transform(self, X_data):
        return torch.matmul(X_data, self.base.T)


if __name__ == '__main__':

    data_dir = "./dataset"
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # read data
    dataloders, dataset_sizes = ImageNetData(data_dir)
    print(dataset_sizes)

    model = torch.load(r".\best_model\saved.pth")
    model.cuda()
    print(model)

    best = 0
    correct = 0
    total = 0

    nn_outdatas = []
    predicts = []
    truelabels = []

    with torch.no_grad():
        for data in dataloders["train"]:
            images, labels = data

            info = model.get_info(images.cuda())
            nn_outdatas.append(torch.flatten(info))
            predicts.append(model(images.cuda()))
            truelabels.append(labels.item())

            total = total + 1

    print(nn_outdatas[0])
    pca = PCA(2)
    X_train_pca = pca.fit_transform(torch.stack(nn_outdatas))
    x = X_train_pca.cpu()[:, 0]
    y = X_train_pca.cpu()[:, 1]
    print(x)
    print(y)
    print(truelabels)
    plt.scatter(x, y, c=truelabels)
    plt.show()
