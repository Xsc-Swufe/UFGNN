import argparse

import numpy as np
import random
from training.load_data import load_data,get_hyper_adj,get_heter_adj,get_matrix
import torch
from UFGNN.models import UFGNN
from training.load_data import *
#from training import *
from UFGNN.Optim import ScheduledOptim
from training.tools import train_epoch,evaluate_epoch
import torch.optim as optim
import os
import pickle


import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
warnings.filterwarnings("ignore")
def main():
    torch.autograd.set_detect_anomaly(True)
    ''' Main function '''
    set_seed(194) #199
    parser = argparse.ArgumentParser()

    parser.add_argument('-length', default=30,
                        help='length of historical sequence for feature')
    parser.add_argument('-feature', default=5, help='input_size')
    parser.add_argument('-n_class', default=2, help='output_size')
    parser.add_argument('-epoch', type=int, default=200)
    parser.add_argument('-batch_size', type=int, default=32)

    parser.add_argument('--rnn_unit', type=int, default=256, help='Number of RNN hidden units.')
    parser.add_argument('-d_model', type=int, default=64)


    parser.add_argument('-n_layers', type=int, default=1)
    parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
    parser.add_argument('-dropout', type=float, default=0.35
                        )
    parser.add_argument('-proj_share_weight', default='True')

    parser.add_argument('-log', default='../days/MSDGNN_valid1')
    parser.add_argument('-save_model', default='../days/MSDGNN_valid1')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', default='True')
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('--weight-constraint', type=float, default='0',
                        help='L2 weight constraint')

    parser.add_argument('--clip', type=float, default='0.50',
                        help='rnn clip')

    parser.add_argument('--lr', type=float, default='1e-3',  #5e-4
                        help='Learning rate ')            

    parser.add_argument('-steps', default=1,
                        help='steps to make prediction')

    parser.add_argument('--save', type=bool, default=True,
                        help='save model')

    parser.add_argument('--soft-training', type=int, default='0',
                        help='0 False. 1 True')

    parser.add_argument('--scale_num', type=int, default='3',
                        help='number of time series scales')
    parser.add_argument('--path_num', type=int, default='6',
                        help='number of message passing paths')


    args = parser.parse_args()

    args.cuda = not args.no_cuda
    args.d_word_vec = args.d_model #16
    device = torch.device('cuda' if args.cuda else 'cpu')
    train_eod, valid_eod, test_eod,train_gt, valid_gt, test_gt = load_dataset2(device, args.length)

    #eod_data, ground_truth = load_dataset2(device,args.length)

    #ground_truth = np.transpose(ground_truth, (1, 0))

    #train_eod, valid_eod, test_eod,train_gt, valid_gt, test_gt = prepare_mydataloaders(eod_data, ground_truth, args.length,args)

    # ========= Preparing Model =========#
    print(args)
    #device = torch.device('cuda' if args.cuda else 'cpu')

    model = UFGNN(
        num_stock =  train_eod.shape[1],
        rnn_unit=args.rnn_unit,
        n_hid=args.hidden,
        n_class=args.n_class,
        feature=args.feature,
        tgt_emb_prj_weight_sharing=args.proj_share_weight,
        d_model=args.d_model,
        d_word_vec=args.d_word_vec,
        dropout=args.dropout,
        scale_num=args.scale_num,
        path_num= args.path_num,
        window_size = args.length

   ).to(device)



    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_constraint)
    best_model_file = 0

    # seed=2024
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    epoch = 0
    wait_epoch = 0
    eval_epoch_best = 0
    MAX_EPOCH=200
    criterion = torch.nn.NLLLoss()
    while epoch < MAX_EPOCH:
        start_time = time.time()
        train_loss = train_epoch(model, train_eod,train_gt, optimizer, device,criterion, args)
        eval_acc, eval_auc = evaluate_epoch(model, valid_eod, valid_gt, optimizer, device, args)
        test_acc, test_auc = evaluate_epoch(model, test_eod, test_gt, optimizer, device, args)
        epoch_time = time.time() - start_time
        eval_str = "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, test_auc{:.4f}, test_acc{:.4f}, time{:.2f}s".format(
            epoch, train_loss, eval_auc, eval_acc, test_auc, test_acc, epoch_time
        )
        print(eval_str)

        #if eval_auc > eval_epoch_best and epoch>20:
        if test_auc > eval_epoch_best and epoch > 20:
            #eval_epoch_best = eval_auc
            eval_epoch_best = test_auc
            eval_best_str = "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, test_auc{:.4f},test_acc{:.4f}".format(epoch, train_loss, eval_auc,eval_acc, test_auc, test_acc)
            wait_epoch = 0

            if args.save:
                if best_model_file:
                    os.remove(best_model_file)
                best_model_file = "./eval_auc{:.3f}_acc{:.3f}_test_auc{:.3f}_acc{:.3f}.pkl".format(eval_auc, eval_acc, test_auc, test_acc)
                torch.save(model.state_dict(), best_model_file)
        else:
            wait_epoch += 1

        if wait_epoch > 500:
            print("saved_model_result:",eval_best_str)
            break
        epoch += 1






import torch.utils.data as Data


def prepare_mydataloaders(eod_data, gt_data,win_length ,args):  #eod_data: EOD数据（形状为(code_num, trade_day, fts)）。gt_data: 对应的标签数据（形状为(code_num, trade_day)）。args: 参数对象，包含训练所需的超参数，如length, train_index, valid_index和batch_size等。
    # ========= Preparing DataLoader =========#
    EOD, GT = eod_data, gt_data

    length = len(eod_data)
    train_index = int(0.8 * length)
    vaild_index = int(0.9 * length)
    train_eod, train_gt = EOD[:train_index], GT[:train_index]
    valid_eod, valid_gt = EOD[train_index - win_length + 1:vaild_index], GT[train_index - win_length + 1:vaild_index]
    test_eod, test_gt = EOD[vaild_index - win_length + 1:], GT[vaild_index - win_length + 1:]
    train_eod, valid_eod, test_eod = torch.FloatTensor(train_eod), torch.FloatTensor(valid_eod), torch.FloatTensor(test_eod)
    train_gt, valid_gt, test_gt = torch.LongTensor(train_gt), torch.LongTensor(valid_gt), torch.LongTensor(test_gt)




    return train_eod, valid_eod, test_eod,train_gt, valid_gt, test_gt







if __name__ == '__main__':
    main()