import time
import torch

from trainings.logger import Logger

from abc import ABC, abstractmethod

def to_np(x):
    return x.data.cpu().numpy()

class AbstractTraining(ABC):

    def __init__(self, model, logger_path='./logs', use_gpu=False):
        self.logger = Logger(logger_path)
        self.model = model
        self.use_gpu = use_gpu
        self.current_step = 0
        self.total_steps = -1
        self.num_log_updates = 5


    @abstractmethod
    def train_model(self, dataloaders, criterion, optimizer, scheduler, num_epochs):
        pass


    @abstractmethod
    def test_model(self, dataloader):
        pass


    def save_model(self, path):
        torch.save(self.model.state_dict(), '%s_state_dict.pth' % path)
        torch.save(self.model, '%s.pth' % path)


    def log_iteration(self, accuracy, loss):
        print('Step [%d/%d], Loss: %.4f, Acc: %.2f'
              % (self.current_step, self.total_steps, loss.data.item(), accuracy))
        # ============ TensorBoard logging ============#
        # (1) Log the scalar values
        info = {
            'loss': loss.data.item(),
            'accuracy': accuracy
        }
        for tag, value in info.items():
            self.logger.scalar_summary("train/" + tag, value, self.current_step + 1)


    def log_network_params(self):
        # (2) Log values and gradients of the parameters (histogram)
        for tag, value in self.model.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary(tag, to_np(value), self.current_step + 1)


    def log_val_stats(self, accuracy, loss):
        self.logger.scalar_summary('val/accuracy', accuracy, self.current_step)
        self.logger.scalar_summary('val/loss', loss, self.current_step)
