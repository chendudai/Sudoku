import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import time

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

# load data
base_data_path = r'C:\Users\ברק\Desktop\Sudoku\Sudoku\dataset\1M_kaggle\\'
with open(base_data_path + 'quizzes.pkl', 'rb') as input:
    quizzes = pickle.load(input)
with open(base_data_path + 'solutions.pkl', 'rb') as input:
    solutions = pickle.load(input)

# convert to one-hot matrices
quizzes_1h = to_categorical(quizzes[:10**4, :, :], 10)
solutions_1h = to_categorical(solutions[:10**4, :, :] - 1, 9)

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

# model
class CNN_1(torch.nn.Module):

    # Our batch shape for input x is (3, 32, 32)

    def __init__(self):
        super(CNN_1, self).__init__()

        # Input channels = 10x9x9 (one hot vector of 0-9), output = 32x10x10
        self.conv1 = torch.nn.Conv2d(train_num_classes, 32, kernel_size=2, stride=1, padding=1)
        # from 32x10x10 to 32x11x11
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=1)

        # from 32x10x10 to 32x11x11
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=1)

        # 4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(10 * 9 * 9, 9**3)
        self.bnf1 = torch.nn.BatchNorm1d(9**3)

        self.fc2 = torch.nn.Linear(9 ** 3, 9 ** 3)
        self.bnf2 = torch.nn.BatchNorm1d(9 ** 3)
        # 64 input features, 10 output features for our 10 defined classes
        self.fc3 = torch.nn.Linear(9**3, solution_num_classes**3)

        self.soft = torch.nn.Softmax(dim=1)
    def forward(self, x):
        # Computes the activation of the first convolution
        # Size changes from (10, 9, 9) to (18, 9, 9)
        # x = F.relu(self.conv1(x))

        # Size changes from (18, 9, 9) to (18, 10, 10)
        # x = self.pool(x)

        # Computes the activation of the second convolution
        # Size changes from (10, 9, 9) to (18, 9, 9)
        # x = F.relu(self.conv2(x))

        # Reshape data to input to the input layer of the neural net
        # Size changes from (18, 16, 16) to (1, 4608)
        # Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 10*9*9)

        # Computes the activation of the first fully connected layer
        # Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.bnf1(self.fc1(x)))
        x = F.relu(self.bnf2(self.fc2(x)))
        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 64) to (1, 10)
        x = self.fc3(x).view(-1, solution_num_classes, solution_num_classes, solution_num_classes)
        x = self.soft(x)
        return x


# def fillBlank(self, net, quizzes):
#     preds = net(quizzes)
#     zeros = np.where()
#     best_probs =



# training function:

def loss_func(quizzes, solutions):
    # Loss function
    loss = F.binary_cross_entropy(quizzes, solutions, reduction='sum')
    return loss, quizzes[0, :, :, :], solutions[0, :, :, :]

def delete_cells(grids, n_delete):
    boards = grids.argmax(1) + 1
    for board in boards:
        board.view(-1)[np.random.randint(0, 81, n_delete)] = 0  # generate blanks (replace = True)

    return to_categorical(boards, train_num_classes)



def trainNet(net, batch_size, n_epochs, learning_rate):
    # Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    # Get training data
    n_batches = len(train_loader)

    # Create our optimizer functions
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Time for printing
    training_start_time = time.time()

    for n_epochs, n_delete in zip([1, 2, 2, 3, 7, 12], [0, 1, 2, 3, 4, 5]):
        # Loop for n_epochs
        for epoch in range(n_epochs):

            running_loss = 0.0
            running_hits = 0.0
            samples_checked = 0
            print_every = n_batches // 10
            start_time = time.time()
            total_train_loss = 0

            for batch_idx, data in enumerate(train_loader, 0):

                # Get inputs
                _, solutions = data
                quizzes = torch.from_numpy(delete_cells(solutions, n_delete)).float()
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
                total_train_loss += loss_size.data
                running_hits += (outputs.argmax(1) == solutions.argmax(1)).sum().double()
                samples_checked += len(outputs)
                # Print every 10th batch of an epoch
                if (batch_idx + 1) % (print_every + 1) == 0:
                    print("Epoch {}, {:d}% \t train_loss: {:.2f} Acc: {:.2f} took: {:.2f}s".format(
                        epoch + 1, int(100 * (batch_idx + 1) / n_batches), running_loss / print_every,
                        running_hits / (samples_checked*9*9), time.time() - start_time))
                    # Reset running loss and time
                    running_loss = 0.0
                    running_hits = 0.0
                    samples_checked = 0
                    start_time = time.time()

            # At the end of the epoch, do a pass on the validation set
            total_val_loss = 0
            val_hits = 0
            val_samples_checked = 0
            for data in val_loader:
                # Wrap tensors in Variables
                _, val_solutions = data
                val_quizzes = torch.from_numpy(delete_cells(val_solutions, n_delete)).float()

                # Forward pass
                val_outputs = net(val_quizzes)
                val_loss_size, _, _ = loss_func(val_outputs, val_solutions)
                total_val_loss += val_loss_size.data
                val_hits += (val_outputs.argmax(1) == val_solutions.argmax(1)).sum().double()
                val_samples_checked += len(val_solutions)

            print("Validation: loss = {:.2f} Acc = {:.2f}"
                  .format(total_val_loss / len(val_loader), val_hits/(9*9*val_samples_checked)))

        print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


def testNet(net):
    test_hits = 0
    test_samples_checked = 0
    with torch.no_grad():
        for data in test_loader:
            quizzes, solutions = data
            outputs = net(quizzes)
            test_hits += (outputs.argmax(1) == solutions.argmax(1)).sum().double()
            test_samples_checked += len(solutions)
            # plot_CM_AUX(np.array(labels), np.array(predicted), classes_name)
    print('Accuracy of the network on the ' + str(test_samples_checked) + ' test images: %d %%' %
          (100 * test_hits / (test_samples_checked*9*9)))


X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(quizzes_1h, solutions_1h)

train_set = MyDataset(X_train, Y_train)
val_set = MyDataset(X_val, Y_val)
test_set = MyDataset(X_test, Y_test)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, drop_last=True,
                          pin_memory=torch.cuda.is_available())
test_loader = DataLoader(test_set, batch_size=4, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_set, batch_size=64, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

if __name__ == "__main__":
    CNN = CNN_1()
    trainNet(CNN, batch_size=128, n_epochs=15, learning_rate=0.001)
    # testNet(CNN)
