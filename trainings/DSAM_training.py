import copy
import time

import torch
from torch.backends import cudnn

import numpy as np

from trainings.abstract_training import AbstractTraining


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def get_training(model, logger_path='./logs', use_gpu=False):
    return DSAMTraining(model, logger_path, use_gpu)    


class DSAMTraining(AbstractTraining):
    def __init__(self, model, logger_path='./logs', use_gpu=False):
        super(DSAMTraining, self).__init__(model, logger_path, use_gpu)


    def train_model(self, dataloaders, criterion, optimizer, scheduler, num_epochs, batch_size=32, im_size=224):

        since = time.time()

        best_model = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        dataset_sizes = {x: [len(dataloaders[x][i].dataset)
                             for i in range(len(dataloaders[x]))]
                         for x in ['train', 'val', 'test']}
        print(dataset_sizes)
        self.current_step = 0
        cudnn.benchmark = True

        sizes = [s for sublist in dataset_sizes.values() for s in sublist]
        steps_per_epoch = int(max(sizes) / batch_size)
        log_frequency = steps_per_epoch // 5
        print('Working with %d steps per epoch' % steps_per_epoch)
        training_epoch_size = batch_size * len(dataloaders['train']) * steps_per_epoch
        
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            ########################
            # START TRAINING PHASE #
            ########################

            phase = 'train'

            scheduler.step()

            self.logger.scalar_summary("learning_rate", scheduler.get_lr()[-1], self.current_step)
            print("Lr: " + str(scheduler.get_lr()))
            self.model.train(True)

            running_loss = 0.0
            running_corrects = 0

            dataiters = [iter(cycle(dataloader)) for dataloader in dataloaders['train']]

            for _ in range(steps_per_epoch):

                inputs = torch.ones(batch_size*len(dataloaders[phase]), 3, im_size, im_size)
                labels = -torch.ones(batch_size*len(dataloaders[phase])).long()
                
                for i in range(len(dataloaders[phase])):
                    
                    domain_inputs, domain_labels = next(dataiters[i])

                    s_idx = batch_size*i
                    e_idx = batch_size*(i+1)

                    inputs[s_idx:e_idx,:,:,:] = domain_inputs
                    labels[s_idx:e_idx] = domain_labels

                if self.use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()

                outputs = self.model(inputs)

                outputs = torch.cat(outputs, 0)

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                accuracy = (labels.data == preds.squeeze()).float().mean()

                loss.backward()
                optimizer.step()
                self.current_step += 1
                if (self.current_step % log_frequency) == 0:
                    self.log_iteration(accuracy, loss)

                running_loss += loss.data.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).data.item()

            epoch_loss = running_loss / float(training_epoch_size)
            epoch_acc = running_corrects / float(training_epoch_size)
            
            train_loss = epoch_loss

            print('{} Loss: {:.4f} DG Loss: {} Acc: {:.4f}'.format(phase, epoch_loss, 'none', epoch_acc))

            ##########################
            # START VALIDATION PHASE #
            ##########################

            phase = 'val'

            val_dataset_size = np.asarray(dataset_sizes['val']).sum()

            predictions_list = []
            labels_data_list = []
            self.model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for i in range(len(dataloaders[phase])):
                for data in dataloaders[phase][i]:
                    inputs, labels = data
                    
                    if self.use_gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                        
                    optimizer.zero_grad()

                    indexes = list(filter(lambda x: x != i, range(len(dataloaders[phase]))))
                    multiple_outputs = [self.model((inputs, idx)) for idx in indexes]
                    outputs = multiple_outputs[0]
                    for o in multiple_outputs[1:]:
                        outputs.add_(o)
                    outputs.mul_(1./len(multiple_outputs))

                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    
                    accuracy = (labels.data == preds.squeeze()).float().mean()
                    running_loss += loss.data.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data).data.item()

                    predictions_list.append(preds)
                    labels_data_list.append(labels.data)

            epoch_loss = running_loss / float(val_dataset_size)
            epoch_acc = running_corrects / float(val_dataset_size)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            self.log_val_stats(epoch_acc, epoch_loss)
            if epoch_acc > best_acc:
                print('Saving best val acc model')
                best_acc = epoch_acc
                best_model = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model)


    def test_model(self, dataloaders):        

        print('Running best model on test set')

        phase = 'test'
        
        dataset_sizes= {'test': [len(dataloaders['test'][i].dataset) for i in range(len(dataloaders['test']))]}

        test_dataset_size = np.asarray(dataset_sizes['test']).sum()

        print('testing on %d images' % test_dataset_size)
        
        predictions_list = []
        labels_data_list = []
        self.model.train(False)

        running_corrects = 0

        assert len(dataloaders[phase]) == 1

        for i in range(len(dataloaders[phase])):

            for data in dataloaders[phase][i]:
                inputs, labels = data
                if self.use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                multiple_outputs = [self.model((inputs, idx)) for idx in range(len(dataloaders['train']))]
                outputs = multiple_outputs[0]
                for o in multiple_outputs[1:]:
                    outputs.add_(o)
                outputs.mul_(1./len(multiple_outputs))

                _, preds = torch.max(outputs.data, 1)
                #print(preds)
                #print(labels.data)
                accuracy = (labels.data == preds.squeeze()).float().mean()
                #print(accuracy)
                running_corrects += torch.sum(preds == labels.data).data.item()
                predictions_list.append(preds)
                labels_data_list.append(labels.data)
            
        epoch_acc = running_corrects / float(test_dataset_size)

        print('{} Test Acc: {:.4f}'.format('test', epoch_acc))
