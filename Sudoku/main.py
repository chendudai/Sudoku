from misc_functions import *
from Nets import *
import pickle
import matplotlib.pyplot as plt
import numpy as np

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
    # add vertical line in epochs that we deleted more elements:
    for xc in stats['epochsOfDelete']:
        ax[0].axvline(x=xc, color='green')
        ax[0].text(x=xc, y=0.8, s=str(int(stats['numDeleted'][xc])), fontsize=8, color='green')
        ax[1].axvline(x=xc, color='green')
        ax[1].text(x=xc, y=1.5, s=str(int(stats['numDeleted'][xc])), fontsize=8, color='green')
    plt.show()

if __name__ == "__main__":
    device = checkDevice()
    # net = FC_1().to(device)
    # dict = torch.load("net_dicts/CNN_1_dict", map_location=device)

    # if device == torch.device("cuda:0"):
    #     with open("net_dicts/dict and stats FC_1/FC_stats_LR0.0001_batchSize512_step2_patience10.txt", "rb") as fp:  # Unpickling
    #         b = pickle.load(fp)
    # else:
    #     with open(r"net_dicts\dict and stats FC_1\FC_stats_LR0.0001_batchSize512_step2_patience10.txt", "rb") as fp:
    #         b = pickle.load(fp)  # Unpickling
    # net, numDeleted, train_acc, train_loss, val_acc, val_loss = b
    # net = net.to(device)
    # net.load_state_dict(dict)
    # #
    # batch_size = 512
    # LR = 0.0001
    # step = 1
    # patience = 3
    # valCalcFreq = 5
    # maxDelete = 100
    # train_loader, test_loader, val_loader = getDataLoaders(batch_size=batch_size, device=device)
    # net, numDeleted, val_numDeleted, epochsOfDelete, epochesValCalc, train_acc, train_loss, val_acc, val_loss = \
    #     trainNet(net, batch_size, LR, step, patience, maxDelete, valCalcFreq, train_loader, val_loader, device)

    # quizzesBoards, deltas = testNet(net, 1, test_loader, device)
    # runTSNE(test_loader, quizzesBoards, deltas)


    # stats = {'net': net, 'numDeleted': numDeleted, 'val_numDeleted': val_numDeleted, 'epochsOfDelete': epochsOfDelete, 'epochesValCalc': epochesValCalc,
    #               'train_acc': train_acc, 'train_loss': train_loss, 'val_acc': val_acc, 'val_loss': val_loss,
    #            'batch_size': batch_size, 'LR': LR, 'step': step, 'patience': patience, 'maxDelete': maxDelete,
    #          'valCalcFreq': valCalcFreq}
    #
    # with open('net_dicts/dict and stats FC_1/stats_FC_1_deleteProperlyWithRept_maxDel100_100epochEachDel.txt', 'wb') as handle:
    #     pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print('Stats saved!')

    with open('net_dicts/dict and stats FC_1/stats_FC_1_deleteProperlyWithRept_maxDel100.txt', "rb") as fp:
        stats = pickle.load(fp)  # Unpickling

    train_loader, test_loader, val_loader = getDataLoaders(batch_size=stats['batch_size'], device=device)
    # # quizzesBoards, deltas = testNet(stats['net'], 55, test_loader, device)
    # # plotStats(stats)
    #
    it = iter(test_loader)
    quizzes_1hot, solutions_1hot = it.next()
    compareToBacktracking(stats['net'], quizzes_1hot.argmax(1), solutions_1hot.argmax(1) + 1, device)


    # trackBoard(stats['net'], test_loader, device)
    # with open('track_data/trackDict1.txt', "rb") as fp:
    #     track_data = pickle.load(fp)  # Unpickling
    #
    # plotTrackBoard(track_data['trackPreds'], track_data['trackBoards'], track_data['trackBoardsSolutions'], digit=3)

    # simulateDeleting(pullNum=110, rangeNum=47)


















