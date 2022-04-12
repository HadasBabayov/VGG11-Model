import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from gcommand_loader import GCommandLoader

NUM_OF_CLASSES = 30
BATCH_SIZE = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = self.define_layers()
        self.fc0 = nn.Linear(7680, 512)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 30)

    def define_layers(self):
        return nn.Sequential(nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
                             nn.BatchNorm2d(64),
                             nn.ReLU(inplace=True),
                             nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                             nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
                             nn.BatchNorm2d(128),
                             nn.ReLU(inplace=True),
                             nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                             nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
                             nn.BatchNorm2d(256),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
                             nn.BatchNorm2d(256),
                             nn.ReLU(inplace=True),
                             nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                             nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
                             nn.BatchNorm2d(512),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
                             nn.BatchNorm2d(512),
                             nn.ReLU(inplace=True),
                             nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                             nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
                             nn.BatchNorm2d(512),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
                             nn.BatchNorm2d(512),
                             nn.ReLU(inplace=True),
                             nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                             nn.AvgPool2d(kernel_size=1, stride=1))

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def validation_accuracy(validation, model):
    print('validation...')
    model.eval()
    with torch.no_grad():
        sum_correct = 0
        total = 0
        for x, y in validation:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = model(x)
            _, pred = torch.max(output.data, 1)
            total += y.size(0)
            sum_correct += (pred == y).sum().item()
        # print('valid Accuracy: {} %'.format((sum_correct / total) * 100))


def train_model(model, optimizer, train_loader):
    epoch = 10
    # Train the model
    # loss_list = []
    # accuracy_list = []
    model.train()
    for e in range(epoch):
        # loss = 0
        # correct = 0
        # total = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            # Forward
            outputs = model(x)
            loss = F.nll_loss(outputs, y)
            # loss_list.append(loss)

            # Bprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #     total = y.size(0)
        #     _, pred = torch.max(outputs.data, 1)
        #     correct = (pred == y).sum().item()
        #     accuracy_list.append(correct / total)
        # 
        # print("Epoch:{}, loss : {}, acc :{} ".format(e + 1, loss.item(), (correct / total) * 100))


def predict(loader, model, test_dataset):
    y_hats = []
    for i, (x, y) in enumerate(loader):
        n = test_dataset.spects[i][0]
        x = x.to(DEVICE)
        output = model(x)
        y_hats.append([os.path.basename(n), output.argmax(1)])
    return y_hats


def write_to_file(y_hats, classes_names):
    size_of_test = len(y_hats)
    with open("test_y", "w") as f:
        for i in range(size_of_test - 1):
            row = y_hats[i][0] + "," + classes_names[y_hats[i][1].item()] + "\n"
            f.write(row)
        row = y_hats[size_of_test - 1][0] + "," + classes_names[y_hats[size_of_test - 1][1].item()]
        f.write(row)


if __name__ == '__main__':
    train = GCommandLoader('./gcommands/train')
    validation = GCommandLoader('./gcommands/valid')
    test = GCommandLoader('./gcommands/test', search_subdir=False)

    test_loader = DataLoader(test, batch_size=1, shuffle=False, pin_memory=True, num_workers=2)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
    valid_loader = DataLoader(validation, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
    # Train the model, predict test's targets and write the results to the file.
    model = Model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train_model(model, optimizer, train_loader)
    # validation_accuracy(valid_loader, model)
    y_hats = predict(test_loader, model, test)
    write_to_file(y_hats, train.classes)
