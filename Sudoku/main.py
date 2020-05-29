from misc_functions import *
from Nets import *
import pickle

if __name__ == "__main__":
    device = checkDevice()
    # net = FC_1().to(device)
    # dict = torch.load("net_dicts/CNN_1_dict", map_location=device)
    with open(r"net_dicts\dict and stats FC_1\FC_stats_LR0.0001_batchSize512_step2_patience10.txt", "rb") as fp:  # Unpickling
        b = pickle.load(fp)

    net, numDeleted, train_acc, train_loss, val_acc, val_loss = b
    # net.load_state_dict(dict)
    batch_size = 128
    LR = 0.0001
    step = 1
    patience = 10
    valCalcFreq = 5
    train_loader, test_loader, val_loader = getDataLoaders(batch_size, device)
    # net, numDeleted, train_acc, train_loss, val_acc, val_loss = \
    #     trainNet(net, batch_size, LR, step, patience, valCalcFreq, train_loader, val_loader, device)
    testNet(net, 50, test_loader, device)
