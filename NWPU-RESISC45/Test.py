import torch.nn as nn
import torch.optim as optim
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


if __name__ == '__main__':

    data_dir = "./dataset"
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # read data
    dataloders, dataset_sizes = ImageNetData(data_dir)
    print(dataset_sizes)
    model = Model()
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    best = 0
    for epoch in range(100):
        running_loss = 0.0
        for i, data in enumerate(dataloders["train"]):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs.cuda())
            print(outputs)
            print(labels.cuda())
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 step
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        correct = 0
        total = 0
        with torch.no_grad():
            for data in dataloders["train"]:
                images, labels = data
                labels = labels.cuda()
                outputs = model(images.cuda())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Accuracy in train: %d %%' % (100 * correct / total))
            correct = 0
            total = 0

            valnum = 5000
            for data in dataloders["val"]:
                images, labels = data
                labels = labels.cuda()
                outputs = model(images.cuda())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if total > valnum:
                    break
            print('Accuracy in val: %d %%' % (100 * correct / total))
            if correct/total >= best:
                best = correct/total
                torch.save(model, r".\best_model\best.pth")
                print("save model")
            print('Accuracy: %d %%' % (100 * correct / total))
    print('Finished Training')
