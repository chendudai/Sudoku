from misc_functions import *
from Nets import *
import pickle
import matplotlib.pyplot as plt
import numpy as np


# if device == torch.device("cuda:0"):
#     with open("net_dicts/dict and stats FC_1/FC_stats_LR0.0001_batchSize512_step2_patience10.txt", "rb") as fp:  # Unpickling
#         b = pickle.load(fp)
# else:
#     with open(r"net_dicts\dict and stats FC_1\FC_stats_LR0.0001_batchSize512_step2_patience10.txt", "rb") as fp:
#         b = pickle.load(fp)  # Unpickling
# net, numDeleted, train_acc, train_loss, val_acc, val_loss = b
# net = net.to(device)
# net.load_state_dict(dict)

def plotStats(stats):#, numDeleted, epochsOfDelete, epochesValCalc, train_acc, train_loss, val_acc, val_loss, batch_size, LR, step, patience, valCalcFreq):
    # plot learning process:
    epochs = np.arange(len(stats['train_acc']))
    fig, ax = plt.subplots(2, 1)
    ax[0].set_title('Accuracy')
    ax[0].plot(epochs, stats['train_acc'], label='Train')
    ax[0].plot(stats['epochesValCalc'], stats['val_acc'], label='Validation')
    ax[0].legend()
    ax[1].set_title('Loss')
    ax[1].plot(epochs, stats['train_loss'], label='Train')
    ax[1].plot(stats['epochesValCalc'], stats['val_loss'], label='Validation')
    ax[1].legend()
    ax[1].set_xlabel('Epochs')
    # add vertical line in epochs that we deleted more elements:
    for xc in stats['epochsOfDelete'][::2]:
        ax[0].axvline(x=xc, color='palegreen')
        ax[0].text(x=xc, y=0.8, s=str(int(stats['numDeleted'][xc])), fontsize=7, color='green')
        ax[1].axvline(x=xc, color='palegreen')
        ax[1].text(x=xc, y=1.5, s=str(int(stats['numDeleted'][xc])), fontsize=7, color='green')
    plt.show()

if __name__ == "__main__":
    device = checkDevice()
    # dict = torch.load("net_dicts/CNN_1_dict", map_location=device)

    # #
    # net = FC_1().to(device)
    # batch_size = 512
    # LR = 0.001
    # step = 1
    # patience = 100
    # valCalcFreq = 5
    # maxDelete = 150
    # # 0 - unproper + repetetions
    # deleteType = 3
    # # train_loader, test_loader, val_loader = getNewDataLoaders(batch_size=batch_size, device=device)
    # train_loader, test_loader, val_loader = getDataLoaders(batch_size=batch_size, device=device)
    # net, numDeleted, val_numDeleted, epochsOfDelete, epochesValCalc, train_acc, train_loss, val_acc, val_loss = \
    #     trainNet(net, batch_size, LR, step, patience, maxDelete, deleteType, valCalcFreq, train_loader, val_loader, device)
    # #
    # #
    # stats = {'net': net, 'numDeleted': numDeleted, 'val_numDeleted': val_numDeleted, 'epochsOfDelete': epochsOfDelete, 'epochesValCalc': epochesValCalc,
    #               'train_acc': train_acc, 'train_loss': train_loss, 'val_acc': val_acc, 'val_loss': val_loss,
    #            'batch_size': batch_size, 'LR': LR, 'step': step, 'patience': patience, 'maxDelete': maxDelete, 'deleteType': deleteType,
    #          'valCalcFreq': valCalcFreq}

    # with open('net_dicts/dict and stats FC_1/stats_FC_1_deleteProperlyWithRept_NoCurriculum_Del100_100epoch_2.txt', 'wb') as handle:
    #     pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print('Stats saved!')

    with open('net_dicts/dict and stats FC_1/stats_FC_1_deleteProperlyWithRept_maxDel100.txt', "rb") as fp:
        stats = pickle.load(fp)  # Unpickling
    train_loader, test_loader, val_loader = getDataLoaders(batch_size=stats['batch_size'], device=device)

    # 1. stats and test:
    # plotStats(stats)
    quizzesBoards, deltas = testNet(stats['net'], test_loader, device)

    # 2. compare to backtracking
    it = iter(test_loader)
    quizzes_1hot, solutions_1hot = it.next()
    compareToBacktracking(stats['net'], quizzes_1hot.argmax(1), solutions_1hot.argmax(1) + 1, device)


    # trackBoard(stats['net'], test_loader, device)
    # with open('track_data/trackDict2.txt', "rb") as fp:
    #     track_data = pickle.load(fp)  # Unpickling
    #

    # 3. boards and cell track
    # plotTrackBoard(track_data['trackPreds'], track_data['trackBoards'], track_data['trackBoardsSolutions'], digit=7)
    # plotTrackCell(track_data['trackPreds'], track_data['trackBoards'], track_data['trackBoardsSolutions'], x=1, y=7)


    # simulateDeleting(pullNum=20, rangeNum=50)


















