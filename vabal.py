import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from dataset_load import dataset_loader
import models.vae as vae_model
import models.resnet as resnet_model
import CustomDataLoader as cdl
import os

class Vabal:
    def __init__(self, args, resnet, vae, weights,sample_idx=[]):
        self.epoch = args.epoch
        self.vae_epoch = args.vae_epoch
        self.sampling_num = args.sampling_num
        self.lr_change_epoch = args.lr_change_epoch
        self.vae_lr_change_epoch = args.vae_lr_change_epoch
        self.sampling_num = args.sampling_num

        np.random.seed(555)
        torch.manual_seed(555)

        
        batch_size = 8
        workers = 2
        # dataset loading...
        self.trainset = cdl.CovidDataSet(weights=weights)
        sample_weights = torch.tensor(self.trainset.weights, dtype=torch.float)
        train_sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        self.trainloader = torch.utils.data.DataLoader(self.trainset,batch_size=batch_size,sampler=train_sampler, num_workers=workers)
        #self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True,num_workers=workers)
        (self.testloader,self.classes) = dataset_loader()
        # model loading...

        net = resnet
        net_vae = vae
        # CUDA & cudnn checking...
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.resnet = net.to(self.device)
        self.vae = net_vae.to(self.device)
        if self.device == 'cuda':
            self.resnet = torch.nn.DataParallel(self.resnet)
            self.vae = torch.nn.DataParallel(self.vae)
            cudnn.deterministic = True
            cudnn.benchmark = False

        # loss loading...
        self.criterion = nn.CrossEntropyLoss(reduction='none')


        #self.resnet_optim = optim.Adam(self.resnet.parameters(), lr=lr)
        #self.resnet_last_optim = optim.Adam(self.resnet.parameters(), lr=last_lr)
        self.resnet_optim = optim.Adam(net.parameters(), lr=0.0002)
        self.vae_optim = optim.Adam(vae.parameters(), lr=args.vae_init_lr)


    def train(self):
        rloss=  self.train_resnet()
        vloss = self.train_vae()
        return (rloss, vloss)

    def train_resnet(self):
        #set to train mode
        self.resnet.train()
        optimiser = self.resnet_optim
        total_loss = []
        for epoch in range(self.epoch):
            train_loss = 0
            correct = 0
            total = 0           
            for batch_idx, sample in enumerate(self.trainloader):
                # set zero
                optimiser.zero_grad()

                # forward
                inputs, targets,weights = sample['img'].to(self.device), sample['target'].to(self.device), sample['weight'].to(self.device)
                outputs = self.resnet(inputs)

                # backward, apply weights
                loss = self.criterion(outputs, targets)
                loss *= weights
                loss = sum(loss)/len(loss)
                loss.backward()

                # optimization
                optimiser.step()

                # logging...
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            total_loss.append(train_loss)
            # Print Training Result
            print('%02d epoch finished >> Training Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch, train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        return total_loss

    def train_vae(self):
        # Set vae to train, resnet to eval
        self.resnet.eval()
        self.vae.train()
        optimiser = self.vae_optim
        total_loss = []
        for i_epoch in range(-1, self.vae_epoch):

            # epoch initialization
            train_BCE_loss = 0
            train_KLD_loss = 0
            train_class_loss = 0
            train_loss = 0

            # batch iterations... (for unlabelled data)
            for batch_idx, sample in enumerate(self.trainloader):
                # set zero
                optimiser.zero_grad()

                # forward (origin network)
                inputs, targets, weights = sample['img'].to(self.device), sample['target'].to(self.device), sample['weight'].to(self.device)
                (feats, outputs) = self.resnet.module.extract_feature(inputs)
                _, predicted = outputs.max(1)

                # forward (vae module)
                x, z, recon_x, mu, logvar = self.vae(feats)

                # loss estimation
                (BCE_loss, KLD_loss, class_loss) = self.vae.module.loss_fnc(recon_x, z, x, mu, logvar, predicted)
                loss = BCE_loss + KLD_loss + class_loss
                #loss = BCE_loss + class_loss

                loss.backward()
                optimiser.step()

                # logging...
                train_BCE_loss += BCE_loss.item()
                train_KLD_loss += KLD_loss.item()
                train_class_loss += class_loss.item()
                train_loss += loss.item()
            total_loss.append(train_loss)
            print('%02d epoch vae training finished >> VAE Training Loss: %.3f (BCE %.3f / KLD %.3f / CLASS %.3f)' % (i_epoch, train_loss/(batch_idx+1),  train_BCE_loss/(batch_idx+1),  train_KLD_loss/(batch_idx+1),  train_class_loss/(batch_idx+1)))
        return total_loss

    def val(self):

        # network initialization...
        self.resnet.eval()
        self.vae.eval()

        # logging initialization...
        val_loss = 0
        correct = 0
        total = 0

        num_classes = len(self.classes)
        prior_prob = torch.zeros(num_classes, dtype=torch.float64).to(self.device)
        labeling_error_prob_nom = torch.zeros(num_classes, dtype=torch.float64).to(self.device)
        labeling_error_prob_den = torch.zeros(num_classes, dtype=torch.float64).to(self.device)
        likelihood_prob = []

        with torch.no_grad():
            # Likelihood estimation (On unlabelled pool)
            for batch_idx, sample in enumerate(self.trainloader):
                # forward
                inputs, targets, weights = sample['img'].to(self.device), sample['target'].to(self.device), sample['weight'].to(self.device)
                (feats, _) = self.resnet.module.extract_feature(inputs)

                likelihood_stack = []
                for sampling_idx in range(self.sampling_num):

                    # forward (vae module)
                    (likelihood_temp, predicted_labels) = self.vae.module.estimate_likelihood(feats)

                    if len(likelihood_stack) < 1:
                        likelihood_stack = likelihood_temp.clone()
                    else:
                        likelihood_stack += likelihood_temp

                    prior_prob += torch.histc(predicted_labels, bins=num_classes, min=0, max=num_classes)

                likelihood_stack /= self.sampling_num
                likelihood_prob += likelihood_stack.tolist()

            prior_prob /= self.sampling_num
            prior_prob /= self.trainloader.__len__()
            prior_prob = prior_prob.cpu().numpy()



            # label noise estimation (On labelled pool)
            for batch_idx, sample in enumerate(self.trainloader):
                # forward
                inputs, targets = sample['img'].to(self.device), sample['target'].to(self.device)
                (feats, _) = self.resnet.module.extract_feature(inputs)

                for sampling_idx in range(self.sampling_num):

                    # forward (vae module)
                    predicted_labels = self.vae.module.predict(feats)

                    labeling_error_prob_den += torch.histc(predicted_labels, bins=num_classes, min=0, \
                                                           max=num_classes).type(torch.cuda.FloatTensor) / 100.0
                    labeling_error_prob_nom += torch.histc(torch.where(predicted_labels.eq(targets), \
                                                                       predicted_labels, \
                                                                       torch.LongTensor([-1]).to(self.device)), \
                                                           bins=num_classes, min=0, \
                                                           max=num_classes).type(torch.cuda.FloatTensor) / 100.0


            labeling_error_prob = labeling_error_prob_nom / labeling_error_prob_den
            labeling_error_prob = labeling_error_prob.cpu().numpy()

        scores = 1 - np.sum(likelihood_prob * np .reshape(prior_prob, [1,-1]) * np.reshape(labeling_error_prob, [1,-1]) \
                            / np.sum(likelihood_prob * np.reshape(prior_prob, [1,-1]), axis=1, keepdims=True), \
                            axis=1)

        return scores

    def test(self):

        # network initialization...
        self.resnet.eval()

        # logging initialization...
        test_loss = 0
        val_correct = 0
        val_total = 0
        class_correct = {0:0, 1:0, 2:0}
        class_total = {0:0, 1:0, 2:0}
        class_predict = {0:0, 1:0, 2:0}
        class_keys = {0:'COVID-19', 1:'normal', 2:'pneumonia'}
        with torch.no_grad():
            # batch iterations...
            for batch_idx, sample in enumerate(self.testloader):
                # forward
                inputs, targets = sample[0].to(self.device), sample[1].to(self.device)
                outputs = self.resnet(inputs)

                # loss estimation (for logging)
                loss = self.criterion(outputs, targets)

                # logging...
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                for each in predicted:
                    class_predict[each.item()] += 1
                for key in class_correct.keys():
                    for k,lab in enumerate(targets,0):
                        if lab == key:
                            class_total[key] += 1
                            if predicted[k] == lab:
                                class_correct[key]+=1
                val_correct += predicted.eq(targets).sum().item()

            for key in class_correct.keys():
                name = class_keys[key]
                correct = class_correct[key]
                total = class_total[key]
                pred = class_predict[key]
                ppv = correct/pred if pred != 0 else 0
                accuracy = (correct/total) * 100
                print("\n",name," Guessed: ",pred,"\naccuracy: ",accuracy,"\n",correct, " out of ",total)
                print("\nPPV: %.4f Sensitivity %.4f" % (ppv, accuracy))
        # Print Test Result
        print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*val_correct/val_total, val_correct, val_total))
        covid_ppv = 100* class_correct[0]/class_predict[0] if class_predict[0] != 0 else 0
        covid_sens = 100* class_correct[0]/class_total[0] if class_total[0] != 0 else 0
        acc = 100*val_correct/val_total
        return (covid_ppv, covid_sens, acc)