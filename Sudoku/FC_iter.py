# -----------------------------------------------------------------------------
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import time
import pandas as pd

def checkDevice():
    if torch.cuda.is_available():
        torch.cuda.current_device()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("running calculations on: ", device)
    return device


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

def torch_categorical(y, num_classes, device=torch.device("cpu")):
    """ 1-hot encodes a tensor """
    # TODO: maybe we can force the input to be with 4 dimensions (the first one can be 1)
    #  and then we will get cleaner function without the "if" part.
    if y.shape.__len__() < 3:
        return torch.transpose(torch.transpose(torch.eye(num_classes)[y], 0, 1), 0, 2).to(device)
    else:
        return torch.transpose(torch.transpose(torch.eye(num_classes)[y], 1, 2), 1, 3).to(device)

def split_data(x, y, ratio_train=0.8, ratio_test=0.1, random_state=42):
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
        # self.data = torch.from_numpy(data).float()
        # self.solution = torch.from_numpy(solution).float()
        self.data = data
        self.solution = solution
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.solution[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)

def fillBlank(net, quizzes, n_delete,  device):
    boards = quizzes.argmax(1).to(device)  # numbers between 0-9.
    for _ in range(n_delete):
        preds = net(torch_categorical(boards, train_num_classes, device))
        # preds = net(torch.from_numpy(to_categorical(boards, train_num_classes)).float().to(device))
        probs = preds.max(1)[0]
        values = preds.argmax(1) + 1
        zeros = (boards == 0).reshape(-1, 81)

        for board, prob, value, zero in zip(boards, probs, values, zeros):
            if any(zero):
                whereZeros = torch.nonzero(zero).view(-1)
                confidence_position = whereZeros[prob.view(-1)[zero].argmax()]
                confidence_value = value.view(-1)[confidence_position]
                board.view(-1)[confidence_position] = confidence_value

    if (boards == 0).any():
        print("There is still a zero in boards!! Chen fault")
    # return torch_categorical(boards - 1, solution_num_classes, device)  # numbers between 0-8.
    return boards  # numbers between 1-9.

def fillBlank_imporved_complexity(net, quizzes, n_delete,  device):
    boards = quizzes.argmax(1).to(device)  # numbers between 0-9.
    for _ in range(n_delete):
        preds = net(torch_categorical(boards, train_num_classes, device))
        # preds = net(torch.from_numpy(to_categorical(boards, train_num_classes)).float().to(device))
        probs = preds.max(1)[0]
        values = preds.argmax(1) + 1
        zeros = (boards == 0)

        probs[~zeros] = 0  # Try ~zeros instead of 1 - zeros
        probs = probs.reshape(-1, 81)
        confidence_position = torch.zeros_like(probs, dtype=torch.bool)
        confidence_position[torch.arange(len(probs)), probs.argmax(1)] = True
        confidence_position = confidence_position.reshape(boards.shape)
        confidence_position[~zeros] = False  # In case there is no zeros in board, we won't change the numbers there.
        boards[confidence_position] = values[confidence_position]


    if (boards == 0).any():
        print("There is still a zero in boards!! Chen fault")
    # return torch_categorical(boards - 1, solution_num_classes, device)  # numbers between 0-8.
    return boards  # numbers between 1-9.

def loss_func(quizzes, solutions):
    # Loss function
    # loss = -torch.sum(torch.log(quizzes).mul(solutions), dim=1)
    loss = F.cross_entropy(quizzes, solutions.argmax(1), reduction='none')
    return loss, quizzes[0, :, :, :], solutions[0, :, :, :]

def delete_cells(grids, n_delete, device=torch.device("cpu")):
    boards = grids.argmax(1) + 1
    for board in boards:
        board.view(-1)[np.random.randint(0, 81, n_delete)] = 0  # generate blanks (replace = True)
    return torch_categorical(boards, train_num_classes, device)

def delete_cells_improved_complexity(grids, n_delete, device=torch.device("cpu")):
    boards = grids.argmax(1) + 1
    boards.view(-1, 81)[torch.arange(len(boards)).repeat(n_delete),
                        torch.randint(0, 81, (n_delete * len(boards), ))] = 0
    return torch_categorical(boards, train_num_classes, device)

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
class CNN_1(torch.nn.Module):
    # model

    def __init__(self):
        super(CNN_1, self).__init__()

        # Input channels = 10x9x9 (one hot vector of 0-9), output = 32x10x10
        self.conv1 = torch.nn.Conv2d(train_num_classes, 32, kernel_size=3, stride=3, padding=0)
        # from 32x10x10 to 32x11x11
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=1)

        # from 32x11x11 to 32x12x12
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=1)

        # 4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(10 * 9 * 9, 9 ** 3)
        self.bn1 = torch.nn.BatchNorm1d(9 ** 3)

        self.fc2 = torch.nn.Linear(9 ** 3, 9 ** 3)
        self.bn2 = torch.nn.BatchNorm1d(9 ** 3)
        # 64 input features, 10 output features for our 10 defined classes
        self.fc3 = torch.nn.Linear(9 ** 3, solution_num_classes ** 3)

        self.soft = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # x = x.view(-1, 10*9*9)
        x = x.reshape(-1, 10 * 9 * 9)
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.fc3(x).view(-1, solution_num_classes, solution_num_classes, solution_num_classes)
        x = self.soft(x)
        return x
# -----------------------------------------------------------------------------
def getDataLoaders(batch_size, device):
    if device == torch.device("cuda:0"):
        # if we run it through Colab:
        a = pd.read_csv('sudoku.csv')
        quizzes = torch.from_numpy(np.asarray([[int(s) for s in str] for str in a.quizzes]).reshape((-1, 9, 9)))
        solutions = torch.from_numpy(np.asarray([[int(s) for s in str] for str in a.solutions]).reshape((-1, 9, 9)))
        samplesNum = 10**6
    else:
        # If run through my CPU:
        base_data_path = r'C:\Users\ברק\Desktop\Sudoku\Sudoku\dataset\1M_kaggle\\'
        with open(base_data_path + 'quizzes.pkl', 'rb') as input:
            quizzes = pickle.load(input)
        with open(base_data_path + 'solutions.pkl', 'rb') as input:
            solutions = pickle.load(input)
        samplesNum = 10**5

    # convert to one-hot matrices
    quizzes_1h = to_categorical(quizzes[:samplesNum, :, :], 10)
    solutions_1h = to_categorical(solutions[:samplesNum, :, :] - 1, 9)

    # split data
    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(quizzes_1h, solutions_1h)

    train_set = MyDataset(X_train, Y_train)
    val_set = MyDataset(X_val, Y_val)
    test_set = MyDataset(X_test, Y_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader, test_loader, val_loader
# -----------------------------------------------------------------------------
def trainNet(net, batch_size, learning_rate, step, patience, train_loader, val_loader):
    # Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("learning_rate=", learning_rate)
    print("step=", step)
    print("patience=", patience)
    print("=" * 30)

    # Get training data
    n_batches = len(train_loader)
    print_every = n_batches // 10

    # Time for printing
    training_start_time = time.time()

    # Statistics:
    numDeleted = np.array([])
    train_acc_total = np.array([])
    val_acc_total = np.array([0])
    train_loss_total = np.array([])
    val_loss_total = np.array([])

    a = np.arange(12, 62, step)
    for n_delete, n_epochs in zip(np.append([2, 3, 4, 6, 8, 10], a),
                                  np.append([3, 3, 4, 5, 5,  5], 10 * np.ones(a.shape))):
        print(f'Delete: {n_delete}.')

        # for epoch in range(n_epochs):
        epoch = 0
        p = 0
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        while p < patience:
            epoch += 1
            running_loss = 0.0
            running_hits = 0.0
            runningDeletedNumber = 0
            samples_checked = 0
            start_time = time.time()
            train_running_loss_epoch = 0

            train_samples_checked_epoch = 0
            train_running_hits_epoch = 0
            net.train()
            for batch_idx, data in enumerate(train_loader, 0):

                # Get inputs
                _, solutions = data
                solutions = solutions.float().to(device)
                quizzes = delete_cells_improved_complexity(solutions, n_delete, device)
                mask_of_deleted_cells = (quizzes.argmax(1) == 0).float()
                # Set the parameter gradients to zero
                optimizer.zero_grad()

                # Forward pass, backward pass, optimize
                outputs = net(quizzes)
                # solved_boards = fillBlank_imporved_complexity(net, quizzes, n_delete, device)  # 1-9

                loss_matrix, quizz_example, sol_example = loss_func(outputs, solutions)
                loss_size = (mask_of_deleted_cells * loss_matrix).sum()
                loss_size.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss_size.data
                train_running_loss_epoch += loss_size.data

                running_hits += (
                            (outputs.argmax(1) == solutions.argmax(1)).float() * mask_of_deleted_cells).sum().double()
                # running_hits += ((solved_boards == solutions.argmax(1) + 1).float() * mask_of_deleted_cells).sum().double()
                samples_checked += len(outputs)
                runningDeletedNumber += mask_of_deleted_cells.sum()
                train_running_hits_epoch += (outputs.argmax(1) == solutions.argmax(1)).sum().double()
                train_samples_checked_epoch += len(outputs)
                # Print every 10th batch of an epoch
                if (batch_idx + 1) % (print_every + 1) == 0:
                    print("Epoch {}, {:d}% \t train_loss: {:.3f} Acc: {:.3f} took: {:.2f}s".format(
                        epoch, int(100 * (batch_idx + 1) / n_batches), running_loss / samples_checked,
                                                                       running_hits / runningDeletedNumber,
                                                                       time.time() - start_time))
                    # Reset running loss and time
                    running_loss = 0.0
                    running_hits = 0.0
                    runningDeletedNumber = 0
                    samples_checked = 0
                    start_time = time.time()

            train_acc_total = np.append(train_acc_total,
                                        (train_running_hits_epoch / (
                                                    train_samples_checked_epoch * n_delete)).cpu().numpy())
            train_loss_total = np.append(train_loss_total,
                                         (train_running_loss_epoch / train_samples_checked_epoch).cpu().numpy())

            if epoch % 5 == 0:
                # At the end of the epoch, do a pass on the validation set
                total_val_loss = 0
                val_hits = 0
                val_runningDeletedNumber = 0
                val_samples_checked = 0
                net.eval()
                with torch.no_grad():
                    for data in val_loader:
                        # Wrap tensors in Variables
                        _, val_solutions = data
                        val_solutions = val_solutions.float().to(device)
                        val_quizzes = delete_cells_improved_complexity(val_solutions, n_delete, device=device)
                        val_mask_of_deleted_cells = (val_quizzes.argmax(1) == 0).float()
                        # Forward pass
                        # val_outputs = net(val_quizzes)
                        val_solved_boards = fillBlank_imporved_complexity(net, val_quizzes, n_delete, device)
                        val_iterative_outputs = torch_categorical(val_solved_boards - 1, 9, device)
                        val_loss_matix, _, _ = loss_func(val_iterative_outputs, val_solutions)
                        total_val_loss += (val_mask_of_deleted_cells * val_loss_matix).sum().data
                        # val_hits += ((val_outputs.argmax(1) == val_solutions.argmax(1)) * val_mask_of_deleted_cells).sum().double()
                        val_hits += ((val_solved_boards == val_solutions.argmax(
                            1) + 1).float() * val_mask_of_deleted_cells).sum().double()
                        val_runningDeletedNumber += val_mask_of_deleted_cells.sum()
                        val_samples_checked += len(val_solutions)

                val_acc_total = np.append(val_acc_total, (val_hits / val_runningDeletedNumber).cpu().numpy())
                val_loss_total = np.append(val_loss_total, (total_val_loss / val_samples_checked).cpu().numpy())
                numDeleted = np.append(numDeleted, (val_runningDeletedNumber / val_samples_checked).cpu().numpy())
                print("Validation: loss = {:.3f} Acc = {:.3f}"
                      .format(total_val_loss / val_samples_checked, val_hits / val_runningDeletedNumber))
                if val_acc_total[-1] <= val_acc_total[-2]:
                    p += 1
                else:
                    p = 0
                if epoch == n_epochs:
                    break

    print("Training finished, took {:.3f}s".format(time.time() - training_start_time))
    return numDeleted, train_acc_total, train_loss_total, val_acc_total[1:], val_loss_total
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    device = checkDevice()
    CNN = CNN_1().to(device)
    batch_size = 128
    LR = 0.0001
    step = 1
    patience = 10
    train_loader, test_loader, val_loader = getDataLoaders(batch_size, device)
    numDeleted, train_acc, train_loss, val_acc, val_loss = \
        trainNet(CNN, batch_size, LR, step, patience, train_loader, val_loader)
# -----------------------------------------------------------------------------

