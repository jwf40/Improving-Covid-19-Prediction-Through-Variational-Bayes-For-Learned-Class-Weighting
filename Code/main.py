from datetime import datetime
from os import makedirs
import random
import numpy as np
import argparse
from dataset_load import *
from vabal import Vabal
import models.vae as vae_model
import models.resnet as resnet_model
from torch.utils.tensorboard import SummaryWriter
import pickle

if __name__ == '__main__':
    ##Times Epoch Values by 10 to get back to origin
    parser = argparse.ArgumentParser(description='Va-bal Training')

    parser.add_argument('--encode_size', default=128, type=int, help='vae encoding size')
    parser.add_argument('--fc_size', default=128, type=int, help='vae fc size')
    parser.add_argument('--class_latent_size', default=10, type=int, help='vae latent size per class')


    parser.add_argument('--lambda_val', default=0.005, type=float, help='vae loss lambda factor')
    parser.add_argument('--sampling_num', default=100, type=int, help='number of sampling for probabilistic inference')

    parser.add_argument('--epoch', default=3, type=int, help='entire epoch number')#200 ##5
    parser.add_argument('--lr_change_epoch', default=160, type=int, help='epoch when the learning rate changes')    
    parser.add_argument('--lr', default=0.0002, type=float, help='initial learning rate')
    parser.add_argument('--last_lr', default=0.00002, type=float, help='last learning rate')

    parser.add_argument('--vae_epoch', default=3, type=int, help='vae entire epoch number')
    parser.add_argument('--vae_lr_change_epoch', default=25, type=int, help='vae epoch when the learning rate changes')
    parser.add_argument('--vae_init_lr', default=0.00000035, type=float, help='vae initial learning rate')
    parser.add_argument('--vae_last_lr', default=0.000000035, type=float, help='vae last learning rate')

    parser.add_argument('--sample_size', default=13892, type=int, help='sampling size for every round')#2000 ##500
    parser.add_argument('--rounds', default=3, type=int, help='maximum sampling rounds')#5
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def loss_to_tensorboard(writer, loss, name, x):
        #tensorboard --logdir logs
        try:
            writer.add_scalar(name, loss.item(),x)
        except AttributeError:
            writer.add_scalar(name, loss, x)
        writer.close()

    timestamp = str(datetime.now()).replace('.','_')
    timestamp = timestamp.replace(':','_')
    save_path = './Logs/'+timestamp+"BOTH_WEIGHT"
    makedirs(save_path)
    writer = SummaryWriter('tensorboard_logs/'+timestamp+"BOTH_WEIGHTS")
    
    random.seed(555)

    train_length = 13892  # TODO
    num_classes = 3
    sample_idx_all = list(range(train_length))
    random.shuffle(sample_idx_all)
    sample_idx = sample_idx_all[:args.sample_size]
    #spare_idx = [x for x in range(train_length) if x not in sample_idx]
    spare_idx = [x for x in range(train_length)]

    resnet = resnet_model.ResNet50()
    resnet = resnet.to(device)
    vae = vae_model.vae(resnet.input_chs, args.fc_size, args.encode_size, num_classes, args.class_latent_size, args.lambda_val)
    vae = vae.to(device)
    scores=None
    total_resnet_loss = []
    total_vae_loss = []
    total_covid_ppv = []
    total_covid_sensitivity = []
    total_val_acc = []
    for i_rounds in range(args.rounds):

        print('{}-th round'.format(i_rounds))

        # trainer build-up
        classification_module = Vabal(args, resnet, vae, scores)

        # trainer training
        (resnet_loss, vae_loss) = classification_module.train()
        total_resnet_loss.extend(resnet_loss)
        total_vae_loss.extend(vae_loss)
        # trainer saving
        #folder_name = classification_module.save(args.ckpt_folder_name + '_{:02d}'.format(i_rounds))
        #np.save(folder_name + '/sample_idx.npy', sample_idx)

        # trainer testing
        (ppv,sens,acc)= classification_module.test()
        total_covid_ppv.append(ppv)
        total_covid_sensitivity.append(sens)
        total_val_acc.append(acc)
        # validation set scoring & logging
        scores = classification_module.val()
        #np.save(folder_name + '/scores.npy', scores)
        torch.save(resnet, save_path+"/resnet_"+timestamp+"_"+str(i_rounds))
        torch.save(vae, save_path+"/vae_"+timestamp+"_"+str(i_rounds))

    torch.save(resnet, save_path+"/FINAL_resnet_"+timestamp)
    torch.save(vae, save_path+"/FINAL_vae_"+timestamp)
    with open(save_path+"/total_resnet_loss_"+timestamp, 'wb') as f:
        pickle.dump(total_resnet_loss, f)
    with open(save_path+"/total_vae_loss_"+timestamp, 'wb') as f:
        pickle.dump(total_vae_loss, f)
    with open(save_path+"/total_covid_ppv_"+timestamp, 'wb') as f:
        pickle.dump(total_covid_ppv, f)
    with open(save_path+"/total_covid_sensitivity_"+timestamp, "wb") as f:
        pickle.dump(total_covid_sensitivity, f)
    with open(save_path+"/total_val_acc_"+timestamp, 'wb') as f:
        pickle.dump(total_val_acc, f)
