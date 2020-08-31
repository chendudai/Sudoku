import torch
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import time
import pandas as pd
from plot_confusion_matrix import plot_CM_AUX
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from solveBacktracking import solve_sudoku as solve_backtracking

import turtle
# import pygame
# pygame.font.init()


# constants:
train_num_classes = 10
solution_num_classes = 9

def checkDevice():
    print("torch version: " + torch.__version__)
    if torch.cuda.is_available():
        torch.cuda.current_device()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("running calculations on: ", device)
    return device

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    # TODO: maybe we can force the input to be with 4 dimensions (the first one can be 1)
    #  and then we will get cleaner function without the "if" part.
    if y.dim() < 3:
        return np.transpose(np.eye(num_classes, dtype='uint8')[y], (2, 0, 1))
    else:
        return np.transpose(np.eye(num_classes, dtype='uint8')[y], (0, 3, 1, 2))

def torch_categorical(y, num_classes, device=torch.device("cpu")):
    """ 1-hot encodes a tensor """
    # TODO: maybe we can force the input to be with 4 dimensions (the first one can be 1)
    #  and then we will get cleaner function without the "if" part.
    if y.dim() < 3:
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

def fillBlank_imporved_complexity(net, quizzes,  device, trackIdx=None, maxIter=None):
    boards = quizzes.argmax(1).to(device)  # numbers between 0-9.
    if trackIdx is not None:
        predsTrack = quizzes[[trackIdx], 1:]  # take the one-hot corresponds to 1-9 only (without the place for 0)
        boardTrack = boards[[trackIdx]]
    i = 1
    while torch.any(boards == 0):
        preds = net(torch_categorical(boards, train_num_classes, device))
        # preds = net(torch.from_numpy(to_categorical(boards, train_num_classes)).float().to(device))
        probs = preds.max(1)[0]
        values = preds.argmax(1) + 1
        zeros = (boards == 0)

        probs[~zeros] = 0
        probs = probs.reshape(-1, 81)
        confidence_position = torch.zeros_like(probs, dtype=torch.bool)
        confidence_position[torch.arange(len(probs)), probs.argmax(1)] = True
        confidence_position = confidence_position.reshape(boards.shape)
        confidence_position[~zeros] = False  # In case there is no zeros in board, we won't change the numbers there.
        boards[confidence_position] = values[confidence_position]
        if maxIter is not None and i >= maxIter:
            boards[zeros] = values[zeros]
            break

        i += 1
        if trackIdx is not None:
            predsTrack = torch.cat((predsTrack, preds[[trackIdx]]), dim=0)
            boardTrack = torch.cat((boardTrack, boards[[trackIdx]]), dim=0)


    if (boards == 0).any():
        print("There is still a zero in boards!! Chen fault")
    # return torch_categorical(boards - 1, solution_num_classes, device)  # numbers between 0-8.
    if trackIdx is not None:
        return boards, predsTrack, boardTrack
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

def delete_cells_improved_complexity(grids, n_delete, device=torch.device("cpu"), deleteType=0, quizzesBoards=None):

    boards = grids.argmax(1) + 1
    # if n_delete == 0:
    #     return torch_categorical(boards, train_num_classes, device)
    if deleteType == 0:
        # delete unproperly with repetitions:
        boards.view(-1, 81)[torch.arange(len(boards)).repeat(n_delete),
                            torch.randint(0, 81, (n_delete * len(boards),))] = 0
    elif deleteType == 1:
        # delete unproperly without repetitions
        boards.view(-1, 81)[torch.arange(len(boards)).repeat(n_delete).reshape(n_delete, len(boards)).T.reshape(-1),
                            torch.tensor(random.sample(range(81), n_delete)).repeat(len(boards))] = 0
    elif deleteType == 2:
        # To delete the proper elements as we know from the quizzes (without repetitions)
        if quizzesBoards is not None:
            boards = boards.view(-1, 81)
            quizzesBoards = quizzesBoards.view(-1, 81)
            zeroInds = torch.where(quizzesBoards == 0)
            for i in range(len(boards)):
                zerosCurBoard = torch.where(zeroInds[0] == i)[0]
                k = min(n_delete, len(zerosCurBoard))
                indsToZero = zerosCurBoard[torch.tensor(random.sample(range(len(zerosCurBoard)), k))]
                boards[i, zeroInds[1][indsToZero]] = 0
            boards = boards.view(-1, 9, 9)
        else:
            print("delete_cells_improved_complexity - insert argument quizzesBoards plz")
    elif deleteType == 3:
        # To delete the proper elements as we know from the quizzes (with repetitions)
        if quizzesBoards is not None:
            boards = boards.view(-1, 81)
            quizzesBoards = quizzesBoards.view(-1, 81)
            zeroInds = torch.where(quizzesBoards == 0)
            for i in range(len(boards)):
                zerosCurBoard = torch.where(zeroInds[0] == i)[0]
                indsToZero = zerosCurBoard[torch.randint(low=0, high=len(zerosCurBoard), size=(n_delete, ))]
                boards[i, zeroInds[1][indsToZero]] = 0
            boards = boards.view(-1, 9, 9)
        else:
            print("delete_cells_improved_complexity - insert argument quizzesBoards plz")

    else:
        print("delete_cells_improved_complexity - deleteType unresolved!")

    return torch_categorical(boards, train_num_classes, device)

def getDataLoaders(batch_size, device):
    if device == torch.device("cuda:0"):
        # if we run it through Colab:
        a = pd.read_csv('dataset/1M_kaggle/sudoku.csv')
        quizzes = torch.from_numpy(np.asarray([[int(s) for s in str] for str in a.quizzes], dtype='uint8').reshape((-1, 9, 9)))
        solutions = torch.from_numpy(np.asarray([[int(s) for s in str] for str in a.solutions], dtype='uint8').reshape((-1, 9, 9)))
        samplesNum = len(quizzes)
    else:
        # If run through my CPU:
        base_data_path = 'dataset/1M_kaggle/'

        a = pd.read_csv(base_data_path + 'sudoku.csv')
        quizzes = torch.from_numpy(np.asarray([[int(s) for s in str] for str in a.quizzes]).reshape((-1, 9, 9)))
        solutions = torch.from_numpy(np.asarray([[int(s) for s in str] for str in a.solutions]).reshape((-1, 9, 9)))

        # with open(base_data_path + 'quizzes.pkl', 'rb') as input:
        #     quizzes = pickle.load(input)
        # with open(base_data_path + 'solutions.pkl', 'rb') as input:
        #     solutions = pickle.load(input)
        samplesNum = 10**6

    # convert to one-hot matrices
    quizzes_1h = to_categorical(quizzes[:samplesNum, :, :], 10)
    solutions_1h = to_categorical(solutions[:samplesNum, :, :] - 1, 9)

    # split data
    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(quizzes_1h, solutions_1h)

    train_set = MyDataset(X_train, Y_train)
    val_set = MyDataset(X_val, Y_val)
    test_set = MyDataset(X_test, Y_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    return train_loader, test_loader, val_loader

def trainNet(net, batch_size, learning_rate, step, patience, maxDelete, valCalcFreq, train_loader, val_loader, device):
    # Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("learning_rate=", learning_rate)
    print("step=", step)
    print("patience=", patience)
    print("=" * 30)

    # Time for printing
    training_start_time = time.time()

    # Statistics:
    numDeleted = np.array([])
    val_numDeleted = np.array([])
    train_acc_total = np.array([])
    val_acc_total = np.array([0])
    train_loss_total = np.array([])
    val_loss_total = np.array([])

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    delJumps = np.arange(12, maxDelete, step)
    n_epochJumps = 100 * np.ones(delJumps.shape)
    epochsOfDelete = []   # from all epochs
    epochesValCalc = []
    epoch = 0
    for n_delete, n_epochs in zip(np.append([1, 2, 3, 4, 6, 8, 10], delJumps), np.append(np.array([2, 2, 3, 4, 5, 8, 10]), n_epochJumps)):
        print('Delete: ', n_delete)
        # for epoch in range(n_epochs):
        relEpoch = 0
        p = 0
        epochsOfDelete.append(epoch)
        while p < patience:
            epoch += 1
            relEpoch += 1
            train_running_loss = 0.0
            train_running_hits = 0.0
            train_samples_checked = 0
            runningDeletedNumber = 0
            start_time = time.time()

            net.train()
            for batch_idx, data in enumerate(train_loader, 0):

                # Get inputs
                Originquizzes, solutions = data
                solutions = solutions.float().to(device)
                Originquizzes = Originquizzes.float().to(device)
                quizzes = delete_cells_improved_complexity(solutions, n_delete, device, deleteType=3, quizzesBoards=Originquizzes.argmax(1))
                mask_of_deleted_cells = (quizzes.argmax(1) == 0).float()
                # Set the parameter gradients to zero
                optimizer.zero_grad()

                # Forward pass, backward pass, optimize
                outputs = net(quizzes)
                loss_matrix, quizz_example, sol_example = loss_func(outputs, solutions)
                loss_size = (mask_of_deleted_cells * loss_matrix).sum()/(mask_of_deleted_cells.sum() + 1e-14)
                loss_size.backward()
                optimizer.step()

                # Update statistics:
                train_running_loss += loss_size.data
                train_running_hits += (
                            (outputs.argmax(1) == solutions.argmax(1)).float() * mask_of_deleted_cells).sum().double()
                # running_hits += ((solved_boards == solutions.argmax(1) + 1).float() * mask_of_deleted_cells).sum().double()
                runningDeletedNumber += mask_of_deleted_cells.sum()
                train_samples_checked += len(outputs)
            train_acc_total = np.append(train_acc_total,
                                        (train_running_hits / (1e-17 + runningDeletedNumber)).cpu().numpy())
            train_loss_total = np.append(train_loss_total, (train_running_loss / len(train_loader)).cpu().numpy())
            numDeleted = np.append(numDeleted, (runningDeletedNumber / train_samples_checked).cpu().numpy())
            print("Delete on average {:.2f}  Epoch {}:\tTook {:.2f}s. \t Train: loss = {:.3f} Acc = {:.3f}"
                  .format(numDeleted[-1], relEpoch, time.time() - start_time, train_loss_total[-1], train_acc_total[-1]))

            if relEpoch % valCalcFreq == 0:
                # At the end of the relEpoch, do a pass on the validation set
                epochesValCalc.append(epoch)
                val_start_time = time.time()
                total_val_loss = 0
                val_hits = 0
                val_runningDeletedNumber = 0
                val_samples_checked = 0
                net.eval()
                with torch.no_grad():
                    for data in val_loader:
                        # Wrap tensors in Variables
                        val_quizzesOrigin, val_solutions = data
                        val_solutions = val_solutions.float().to(device)
                        val_quizzes = delete_cells_improved_complexity(val_solutions, n_delete, device=device, deleteType=3, quizzesBoards=val_quizzesOrigin.argmax(1))
                        val_mask_of_deleted_cells = (val_quizzes.argmax(1) == 0).float()
                        # Forward pass
                        # val_outputs = net(val_quizzes)
                        val_solved_boards = fillBlank_imporved_complexity(net, val_quizzes, device)
                        val_iterative_outputs = torch_categorical(val_solved_boards - 1, 9, device)
                        val_loss_matix, _, _ = loss_func(val_iterative_outputs, val_solutions)
                        total_val_loss += (val_mask_of_deleted_cells * val_loss_matix).sum()/val_mask_of_deleted_cells.sum()
                        # val_hits += ((val_outputs.argmax(1) == val_solutions.argmax(1)) * val_mask_of_deleted_cells).sum().double()
                        val_hits += ((val_solved_boards == val_solutions.argmax(
                            1) + 1).float() * val_mask_of_deleted_cells).sum().double()
                        val_runningDeletedNumber += val_mask_of_deleted_cells.sum()
                        val_samples_checked += len(val_solutions)

                val_acc_total = np.append(val_acc_total, (val_hits / val_runningDeletedNumber).cpu().numpy())
                val_loss_total = np.append(val_loss_total, (total_val_loss / len(val_loader)).cpu().numpy())
                val_numDeleted = np.append(val_numDeleted, (val_runningDeletedNumber / val_samples_checked).cpu().numpy())
                print("Delete {}  Epoch {}:\tTook {:.2f}s. \t Validation: loss = {:.3f} Acc = {:.3f}"
                      .format(val_numDeleted[-1], relEpoch, time.time() - val_start_time, val_loss_total[-1], val_acc_total[-1]))
                if val_acc_total[-1] <= val_acc_total[-2]:
                    p += 1
                else:
                    p = 0
            if relEpoch >= n_epochs:
                break


    print("Training finished, took {:.3f}s".format(time.time() - training_start_time))
    return net, numDeleted, val_numDeleted, epochsOfDelete, epochesValCalc, train_acc_total, train_loss_total, val_acc_total[1:], val_loss_total

def testNet(net, n_delete, test_loader, device):
    hits = 0
    deltas = torch.tensor([], dtype=torch.int64)
    runningDeletedNumber = 0
    samples_checked = 0
    true_cell_total = torch.tensor([], dtype=torch.int64)
    solved_cell_total = torch.tensor([], dtype=torch.int64)
    net.eval()
    quizzesBoards = torch.zeros((test_loader.dataset.data.shape[0], 81))
    time_solved_net = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            quizzes, solutions = data
            solutions = solutions.float().to(device)
            quizzes = quizzes.float().to(device)
            # quizzes = torch_categorical(quizzes.type(torch.bool), 10, device)
            # quizzes = delete_cells_improved_complexity(solutions, n_delete, device=device, deleteType=2, quizzesBoards=quizzes.argmax(1))
            quizzesBoards[samples_checked:samples_checked + len(quizzes)] = (quizzes.argmax(1) + 1).reshape(-1, 81)
            mask_of_deleted_cells = (quizzes.argmax(1) == 0)
            # Forward pass
            t = time.time()
            solved_boards = fillBlank_imporved_complexity(net, quizzes, device)
            time_solved_net += (time.time() - t)
            # statistics:
            solutions_boards = solutions.argmax(1) + 1
            deltas = torch.cat((deltas, diff(solutions_boards, solved_boards).cpu()))  # get number of errors on each quizz
            runningDeletedNumber += mask_of_deleted_cells.sum()
            hits += ((solved_boards == solutions_boards).float() * mask_of_deleted_cells.float()).sum().double()
            samples_checked += len(solutions)
            true_cell_total = torch.cat((true_cell_total, solutions_boards[mask_of_deleted_cells].cpu()))
            solved_cell_total = torch.cat((solved_cell_total, solved_boards[mask_of_deleted_cells].cpu()))
    boardAcc = (deltas == 0).float().mean()  # portion of correct solved quizzes
    plotConfusionMatrix(true_cell_total, solved_cell_total)
    print("test images number: " + str(samples_checked) + '. Cell Acc:  %f, Board Acc:  %f. Deleted on average: %f. average time for a board: %f seconds'
          % (hits / runningDeletedNumber, boardAcc, runningDeletedNumber.__float__()/samples_checked, time_solved_net/samples_checked))
    return quizzesBoards, deltas

def trackBoard(net, test_loader, device):
    trackIdx = 0
    trackPreds = []
    trackBoards = []
    trackBoardsSolutions = []
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            quizzes, solutions = data
            solutions = solutions.float().to(device)
            quizzes = quizzes.float().to(device)
            # Forward pass
            solved_boards, trackPred, trackBorad = fillBlank_imporved_complexity(net, quizzes, device, trackIdx=trackIdx)
            trackPreds.append(trackPred)
            trackBoards.append(trackBorad)
            solutions_boards = solutions.argmax(1) + 1
            trackBoardsSolutions.append(solutions_boards[trackIdx])
    plotTrackBoard(trackPreds, trackBoards, trackBoardsSolutions, 9)
    return trackPreds, trackBoards, trackBoardsSolutions

def plotTrackBoard(trackPreds, trackBoards, trackBoardsSolutions, digit=[]):
    fillx = -1
    filly = -1
    iterationsNum = 20
    boardIdx = 0
    # plot solution:
    plotBoard(trackBoardsSolutions[boardIdx], digit, trackPreds[boardIdx][0][0], 'end')
    for i, partialBoard in enumerate(trackBoards[boardIdx][:iterationsNum]):
        # draw(partialBoard)
        if i < len(trackBoards[boardIdx]):
            fillx, filly = torch.where(trackBoards[boardIdx][i+1] != partialBoard)
        plotBoard(partialBoard, digit, trackPreds[boardIdx][i + 1][digit-1], i, fillx, filly)

def plotBoard(board, digit, predsOfDigit, stage, fillx=-1, filly=-1):
    nx = 9
    ny = 9
    # data = np.random.randint(0, 10, size=(ny, nx))
    boardToPlot = board.cpu().numpy().astype('str')
    boardToPlot[boardToPlot == '0'] = ''
    plt.figure()
    plt.title('sudoku fill with confidence of digit: ' + str(digit) + '. iteration: ' + str(stage))
    tb = plt.table(cellText=boardToPlot, loc=(0, 0), cellLoc='center', fontsize=20)
    for i in range(9):
        for j in range(9):
            if boardToPlot[i, j] == '':
                # draw confidence
                t = tb[(i, j)].get_text().get_text() + ' ${^{' + '{:.0f}'.format(100*predsOfDigit[i, j]) + '\%}}$'
                tb[(i, j)].set_text_props(text=t, fontsize=20, color='red', fontweight=100)
            else:
                tb[(i, j)].set_text_props(fontsize=15)
            if (3 <= i < 6 and j < 3) or (3 <= i < 6 and 6 <= j) or (i < 3 and 3 <= j < 6) or (6 <= i and 3 <= j < 6):
                tb[(i, j)].set_facecolor("gainsboro")
            if (i == fillx) and (j == filly):
                tb[(i, j)].set_facecolor("aquamarine")


    tc = tb.properties()['child_artists']
    for cell in tc:
        cell.set_height(1 / ny)
        cell.set_width(1 / nx)

    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

def diff(grids_true, grids_pred):
    # get number of errors on each quizz
    return (grids_true != grids_pred).sum((1, 2))

def plotConfusionMatrix(truthBoards, predictedBoards):
    truthBoards = (truthBoards - 1)
    predictedBoards = (predictedBoards - 1)
    plot_CM_AUX(truthBoards, predictedBoards, normalize=True, class_names=np.arange(1, 10).astype(str))

def runTSNE(test_loader, quizzesBoards, deltas):
    # correctIndsOfBorads = np.where(deltas == 0)[0]
    random_state = 42
    np.random.seed(random_state)
    # quizzes = (test_loader.dataset.data.argmax(1) + 1).reshape(-1, 81)
    # solutions = (test_loader.dataset.solution.argmax(1) + 1).reshape(-1, 81)
    allData = quizzesBoards
    y = (deltas == 0)
    # allData = np.concatenate((quizzes, solutions), axis=0)
    scaler = StandardScaler()
    allData_scaled = scaler.fit_transform(allData)
    # y = np.append(np.zeros(quizzes.shape[0]), np.ones(solutions.shape[0]))
    pca = PCA(3)
    allData_embedded = pca.fit_transform(allData_scaled)
    # for perp in np.arange(12, 120, 10):
    allData_embedded = TSNE(n_components=3, perplexity=60, random_state=random_state).fit_transform(allData_scaled)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(allData_embedded[y == 0, 0],
                allData_embedded[y == 0, 1],
                allData_embedded[y == 0, 2], c='tab:blue', label='quizzes solved UNcorrectly')
    ax.scatter(allData_embedded[y == 1, 0],
                allData_embedded[y == 1, 1],
                allData_embedded[y == 1, 2], c='tab:orange', label='quizzes solved correctly')
    plt.xlabel("first component")
    plt.ylabel("second component")
    plt.title("PCA projection")
    plt.legend()
    plt.show()

def compareToBacktracking(net, quizzes, solutions, device, maxIterNet=None):
    if np.ndim(quizzes) == 2:
        quizzes = quizzes[np.newaxis, :, :]
        solutions = solutions[np.newaxis, :, :]
    time_net = []
    time_backtracking = []
    BoardCorrectIdx_net = []
    BoardCorrectIdx_backtracking = []
    numDelete = []
    net.eval()
    with torch.no_grad():
        for i, (quizz, sol) in enumerate(zip(quizzes, solutions)):
            numDelete.append((quizz == 0).sum())

            t = time.time()
            quizz_to_net = torch_categorical(quizz, 10, device)
            quizz_to_net = quizz_to_net.reshape(1, 10, 9, 9)
            solved_net = fillBlank_imporved_complexity(net, quizz_to_net, device, maxIter=maxIterNet)
            time_net.append(time.time() - t)
            if (solved_net[0].cpu() == sol).all():
                BoardCorrectIdx_net.append(True)
            else:
                BoardCorrectIdx_net.append(False)

            t = time.time()
            out, solved_back = solve_backtracking(quizz.cpu())
            time_backtracking.append(time.time() - t)
            if (solved_back == sol).all():
                BoardCorrectIdx_backtracking.append(True)
            else:
                BoardCorrectIdx_backtracking.append(False)

    BoardCorrectIdx_backtracking = np.array(BoardCorrectIdx_backtracking)
    BoardCorrectIdx_net = np.array(BoardCorrectIdx_net)
    numDelete = np.array(numDelete)
    time_backtracking = np.array(time_backtracking)
    time_net = np.array(time_net)


    plt.figure()
    plt.scatter(numDelete[BoardCorrectIdx_net & BoardCorrectIdx_backtracking] + 0.1,
                time_net[BoardCorrectIdx_net & BoardCorrectIdx_backtracking], label='Net', marker='x')
    plt.scatter(numDelete[BoardCorrectIdx_net & BoardCorrectIdx_backtracking] - 0.1,
                time_backtracking[BoardCorrectIdx_net & BoardCorrectIdx_backtracking], label='Backtracking')
    plt.title('Compare Running Time: Neural Net vs Backtracking', fontsize=15)
    plt.xlabel('Number of deleted elements', fontsize=15)
    plt.ylabel('Running Time', fontsize=15)
    plt.legend(fontsize=15)
    plt.tick_params(labelsize=15)
    plt.grid()
    plt.annotate('#Boards: ' + str(len(quizzes)) + \
                 '\nNet: Solved: ' + str(sum(BoardCorrectIdx_net)) + '. Average Time: {:.2f}'.format(time_net.mean()) + \
                 '\nBactracking: Solved: ' + str(sum(BoardCorrectIdx_backtracking)) + '. Average Time: {:.2f}'.format(time_backtracking.mean()),
                 xy=(0.3, 0.8), xycoords='axes fraction', backgroundcolor='palegreen', fontsize=12)
    plt.show()



def simulateDeleting(pullNum, rangeNum):
    deletedNum = np.zeros(rangeNum + 1)
    iterNum = 10000

    for i in range(iterNum):
        tmp = len(torch.unique(torch.randint(low=0, high=rangeNum, size=(pullNum,))))
        deletedNum[tmp] += 1
    plt.bar(range(rangeNum + 1), deletedNum/deletedNum.sum())
    plt.title(str(pullNum) + ' pulls. ' + str(rangeNum) + ' Different numbers in the bucket', fontsize=15)
    plt.suptitle('Probability to delete X different numbers with repeatitions', fontsize=20)
    plt.xticks(range(rangeNum + 1), range(rangeNum + 1))
    plt.xlabel('X', fontsize=15)
    plt.ylabel('Probability', fontsize=15)
    plt.show()




