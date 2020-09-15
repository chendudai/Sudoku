# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import time
import pandas as pd
# -----------------------------------------------------------------------------
# constants:
train_num_classes = 10
solution_num_classes = 9

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    # TODO: maybe we can force the input to be with 4 dimensions (the first one can be 1)
    #  and then we will get cleaner function without the "if" part.
    if y.shape.__len__() < 3:
        return np.transpose(np.eye(num_classes, dtype='uint8')[y], (2, 0, 1))
    else:
        return np.transpose(np.eye(num_classes, dtype='uint8')[y], (0, 3, 1, 2))

def split_data(x, y, ratio_train = 0.8, ratio_test = 0.1, random_state = 42):
    # Defines ratios, w.r.t. whole dataset.
    ratio_val = 1 - ratio_test - ratio_train

    # Produces test split.
    x_remaining, x_test, y_remaining, y_test = train_test_split(
        x, y, test_size=ratio_test, random_state=random_state)

    # Adjusts val ratio, w.r.t. remaining dataset.
    ratio_remaining = 1 - ratio_test
    ratio_val_adjusted = ratio_val / ratio_remaining

    # Produces train and val splits.
    x_train, x_val, y_train, y_val = train_test_split(
        x_remaining, y_remaining, test_size=ratio_val_adjusted, random_state=random_state)

    return x_train, x_val, x_test, y_train, y_val, y_test

# dataset object
class MyDataset(Dataset):
    def __init__(self, data, solution, transform=None):
        self.data = torch.from_numpy(data).float()
        self.solution = torch.from_numpy(solution).float()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.solution[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


# def fillBlank(self, net, quizzes):
#     preds = net(quizzes)
#     zeros = np.where()
#     best_probs =


def loss_func(quizzes, solutions):
    # Loss function
    loss = F.binary_cross_entropy(quizzes, solutions, reduction='sum')
    return loss, quizzes[0, :, :, :], solutions[0, :, :, :]

def delete_cells(grids, n_delete):
    boards = grids.argmax(1) + 1
    for board in boards:
        board.view(-1)[np.random.randint(0, 81, n_delete)] = 0  # generate blanks (replace = True)

    return to_categorical(boards, train_num_classes)

def testNet(net):
    test_hits = 0
    test_samples_checked = 0
    net.eval()
    with torch.no_grad():
        for data in test_loader:
            quizzes, solutions = data
            outputs = net(quizzes)
            test_hits += (outputs.argmax(1) == solutions.argmax(1)).sum().double()
            test_samples_checked += len(solutions)
            # plot_CM_AUX(np.array(labels), np.array(predicted), classes_name)
    print('Accuracy of the network on the ' + str(test_samples_checked) + ' test images: %d %%' %
          (100 * test_hits / (test_samples_checked*9*9)))
# -----------------------------------------------------------------------------
# model
class CNN_2(torch.nn.Module):
    def __init__(self):
        super(CNN_2, self).__init__()
        self.kenels_num = 16
        self.conv1_9 = torch.nn.Conv2d(train_num_classes, self.kenels_num, kernel_size=(1, 9), stride=1, padding=0)
        self.conv9_1 = torch.nn.Conv2d(train_num_classes, self.kenels_num, kernel_size=(9, 1), stride=1, padding=0)
        self.conv3_3 = torch.nn.Conv2d(train_num_classes, self.kenels_num, kernel_size=3, stride=3, padding=0)
        # self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=1)

        # 4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(2 * 9 * self.kenels_num, 9 ** 2)
        self.bn1 = torch.nn.BatchNorm1d(9 ** 2)

        self.fc2 = torch.nn.Linear(9 ** 2, 9 ** 3)
        self.bn2 = torch.nn.BatchNorm1d(9 ** 3)
        # 64 input features, 10 output features for our 10 defined classes
        self.fc3 = torch.nn.Linear(9 ** 3, solution_num_classes ** 3)

        self.soft = torch.nn.Softmax(dim=1)

    def forward(self, x):
        y1 = self.conv1_9(x)
        # y2 = self.conv3_3(x)
        x = self.conv9_1(x)

        x = torch.cat((y1.view(-1, 9 * self.kenels_num), x.view(-1, 9 * self.kenels_num)), dim=1)
        # x = torch.cat((y1.view(-1, 9*kenels_num), y2.view(-1, 9*kenels_num), x.view(-1, 9*kenels_num)), dim=1)
        # x = x.view(-1, 9*kenels_num)
        # Computes the activation of the first fully connected layer
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 64) to (1, 10)
        x = self.fc3(x).view(-1, solution_num_classes, solution_num_classes, solution_num_classes)
        x = self.soft(x)
        return x
# -----------------------------------------------------------------------------
# convert to one-hot matrices

base_data_path = r'C:\Users\ברק\Desktop\Sudoku\Sudoku\dataset\1M_kaggle\\'
with open(base_data_path + 'quizzes.pkl', 'rb') as input:
    quizzes = pickle.load(input)
with open(base_data_path + 'solutions.pkl', 'rb') as input:
    solutions = pickle.load(input)

# convert to one-hot matrices
quizzes_1h = to_categorical(quizzes[:10**2, :, :], 10)
solutions_1h = to_categorical(solutions[:10**2, :, :] - 1, 9)

# split data
X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(quizzes_1h, solutions_1h)

train_set = MyDataset(X_train, Y_train)
val_set = MyDataset(X_val, Y_val)
test_set = MyDataset(X_test, Y_test)

batch_size = 8

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False,
                          pin_memory=torch.cuda.is_available())
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

# -----------------------------------------------------------------------------
if torch.cuda.is_available():
    torch.cuda.current_device()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("running calculations on: ", device)


def trainNet(net, batch_size, n_epochs, learning_rate):
    # Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    # Get training data
    n_batches = len(train_loader)

    # Create our optimizer functions
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Time for printing
    training_start_time = time.time()

    # Statistics:
    numDeleted = []
    train_acc_total = []
    val_acc_total = []
    train_loss_total = []
    val_loss_total = []

    for n_epochs, n_delete in zip([1, 1, 2, 1, 2, 1, 2, 3, 3, 2, 2, 2, 2],
                                  [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20]):
        # Loop for n_epochs
        print(f'Delete: {n_delete}. Epochs: {n_epochs}')
        for epoch in range(n_epochs):

            running_loss = 0.0
            running_hits = 0.0
            samples_checked = 0
            print_every = n_batches // 10
            start_time = time.time()
            train_running_loss_epoch = 0

            train_samples_checked_epoch = 0
            train_running_hits_epoch = 0

            for batch_idx, data in enumerate(train_loader, 0):

                # Get inputs
                _, solutions = data
                solutions = solutions.to(device)
                quizzes = torch.from_numpy(delete_cells(solutions, n_delete)).float().to(device)
                # Set the parameter gradients to zero
                optimizer.zero_grad()

                # Forward pass, backward pass, optimize
                outputs = net(quizzes)
                # outputs = fillBlank(net, quizzes)
                loss_size, quizz_example, sol_example = loss_func(outputs, solutions)
                loss_size.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss_size.data
                train_running_loss_epoch += loss_size.data
                running_hits += (outputs.argmax(1) == solutions.argmax(1)).sum().double()
                samples_checked += len(outputs)

                train_running_hits_epoch += (outputs.argmax(1) == solutions.argmax(1)).sum().double()
                train_samples_checked_epoch += len(outputs)
                # Print every 10th batch of an epoch
                if (batch_idx + 1) % (print_every + 1) == 0:
                    print("Epoch {}, {:d}% \t train_loss: {:.6f} Acc: {:.2f} took: {:.2f}s".format(
                        epoch + 1, int(100 * (batch_idx + 1) / n_batches), running_loss / samples_checked,
                        running_hits / (samples_checked * 9 * 9), time.time() - start_time))
                    # Reset running loss and time
                    running_loss = 0.0
                    running_hits = 0.0
                    samples_checked = 0
                    start_time = time.time()

            train_acc_total.append(train_running_hits_epoch / (train_samples_checked_epoch * 9 * 9))
            train_loss_total.append(train_running_loss_epoch / train_samples_checked_epoch)
            # At the end of the epoch, do a pass on the validation set
            total_val_loss = 0
            val_hits = 0
            val_samples_checked = 0
            net.eval()
            with torch.no_grad():
                for data in val_loader:
                    # Wrap tensors in Variables
                    _, val_solutions = data
                    val_solutions = val_solutions.to(device)
                    val_quizzes = torch.from_numpy(delete_cells(val_solutions, n_delete)).float().to(device)

                    # Forward pass
                    val_outputs = net(val_quizzes)
                    val_loss_size, _, _ = loss_func(val_outputs, val_solutions)
                    total_val_loss += val_loss_size.data
                    val_hits += (val_outputs.argmax(1) == val_solutions.argmax(1)).sum().double()
                    val_samples_checked += len(val_solutions)

                val_acc_total.append(val_hits / (9 * 9 * val_samples_checked))
                val_loss_total.append(total_val_loss / val_samples_checked)
                numDeleted.append(n_delete)
            print("Validation: loss = {:.6f} Acc = {:.2f}"
                  .format(total_val_loss / val_samples_checked, val_hits / (9 * 9 * val_samples_checked)))

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    return numDeleted, train_acc_total, train_loss_total, val_acc_total, val_loss_total
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    CNN = CNN_2().to(device)
    numDeleted, train_acc, train_loss, val_acc, val_loss = trainNet(CNN, batch_size=batch_size, n_epochs=15, learning_rate=0.0001)
    # testNet(CNN)
# -----------------------------------------------------------------------------
