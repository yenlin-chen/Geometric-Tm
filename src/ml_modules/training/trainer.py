import torch, sys
import numpy as np

# import torch_geometric as pyg

df_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        scheduler=None,
        # CAUTION: RISK OF DATA LEAKAGE
        # TODO: update default values
        min_max=(25, 100),
        mean_std=(50, 15),
        num_workers=4,
        device=df_device,
        reset_weights=True,
    ):

        '''
        Parameters
        ----------
        model : torch.nn.Module
            Model to be trained.
        loss_fn : torch.nn.Module
            Loss function to be used.
        optimizer : torch.optim.Optimizer
            Optimizer to be used.
        min_max : tuple
            Minimum and maximum values of the dataset labels.
        mean_std : tuple
            Mean and standard deviation of the dataset labels.
        num_workers : int
            Number of workers for the DataLoader.
        device : torch.device
            Device on which to train the model.
        '''

        # components of one ML experiment
        if device == torch.device('cuda'):
            self.model = torch.nn.DataParallel(model).to(device)
        else:
            self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        # values for standardizing dataset labels
        self.val_min, self.val_max = min_max[0], min_max[1]
        self.val_range = self.val_max - self.val_min
        self.val_mean, self.val_std = mean_std[0], mean_std[1]

        # machine-specific parameters
        self.num_workers = num_workers
        self.device = device

        # initialize model weights
        if reset_weights:
            self.reset_model_weights()

    def reset_model_weights(self):

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.reset_parameters()
        else:
            self.model.reset_parameters()

    def load_model_state_dict(self, state_dict):

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

    # @pyg.profile.profileit()
    def train_one_epoch(self, train_loader):
        '''Trains the model for one epoch, iterating over `train_set`.

        In this implementation, model parameters are updated on evey
        batch. Loss should go down as training progresses. The average
        loss is return at the end of the epoch, along with the predicted
        values and the labels of the entire dataset.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader object for the training set.

        Returns
        -------
        avg_loss : float
            Average loss over the entire training set.
        outputs : torch.Tensor
            Predicted values for the entire training set.
        labels : torch.Tensor
            Labels for the entire training set.
        '''

        # information on train_loader
        train_size = len(train_loader.dataset)
        batch_size = train_loader.batch_size
        if isinstance(train_loader.dataset, torch.utils.data.dataset.Subset):
            Tm_dict = train_loader.dataset.dataset.Tm_dict
        else:
            Tm_dict = train_loader.dataset.Tm_dict

        # set model to training mode
        self.model.train()

        ### iterate over train set, one batch at a time
        total_loss = 0.0
        outputs = torch.zeros(
            (train_size,), dtype=torch.float32, device=self.device
        )
        labels = torch.zeros(
            (train_size,), dtype=torch.float32, device=self.device
        )
        batch_Tm = torch.zeros(
            (batch_size,), dtype=torch.float32, device=self.device
        )
        accessions = np.empty((train_size,), dtype='U16')
        for i, data_batch in enumerate(train_loader):

            ### zero the gradients
            # mandatory for every batch using PyTorch
            self.optimizer.zero_grad()

            data_batch = data_batch.to(self.device)

            ### make predictions for data batch
            pred = self.model(data_batch).squeeze()

            ### compute the loss and its gradients
            # standardize true labels to mean 0 and std 1
            for j, a in enumerate(data_batch.accession):
                batch_Tm[j] = Tm_dict[a]
            # batch_Tm = data_batch.Tm
            true = (batch_Tm[:pred.numel()] - self.val_mean) / self.val_std
            # true = (batch_Tm - self.val_mean) / self.val_std
            loss = self.loss_fn(pred, true)
            # backpropagate
            loss.backward()

            ### adjust learning weights
            self.optimizer.step()

            ### compute total loss
            total_loss += loss.item()

            ### gather model outputs and labels
            # convert predicted values back to Celsius
            outputs[i * batch_size : (i + 1) * batch_size] = (
                pred * self.val_std + self.val_mean
            )
            labels[i * batch_size : (i + 1) * batch_size] = batch_Tm[
                : pred.numel()
            ]
            accessions[i * batch_size : (i + 1) * batch_size] = (
                data_batch.accession
            )

        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / train_size

        return avg_loss, outputs, labels, accessions

    @torch.no_grad()
    def evaluate(self, dataloader):
        '''Evaluates the model on `dataloader`.

        In this implementation, model parameters are not updated. The
        average loss is return at the end of the epoch, along with the
        predicted values and the labels of the entire dataset.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            DataLoader object for the dataset.
        Returns
        -------
        avg_loss : float
            Average loss over the entire dataset.
        outputs : torch.Tensor
            Predicted values for the entire dataset.
        labels : torch.Tensor
            Labels for the entire dataset.
        '''

        # information on dataset
        dataset_size = len(dataloader.dataset)
        batch_size = dataloader.batch_size
        if isinstance(dataloader.dataset, torch.utils.data.dataset.Subset):
            Tm_dict = dataloader.dataset.dataset.Tm_dict
        else:
            Tm_dict = dataloader.dataset.Tm_dict

        # set model to evaluation mode
        self.model.eval()

        ### iterate over dataset, one batch at a time
        total_loss = 0.0
        outputs = torch.zeros(
            (dataset_size,), dtype=torch.float32, device=self.device
        )
        labels = torch.zeros(
            (dataset_size,), dtype=torch.float32, device=self.device
        )
        batch_Tm = torch.zeros(
            (batch_size,), dtype=torch.float32, device=self.device
        )
        accessions = np.empty((dataset_size,), dtype='U16')
        for i, data_batch in enumerate(dataloader):

            data_batch = data_batch.to(self.device)

            ### make predictions
            pred = self.model(data_batch).squeeze()

            ### compute loss
            # standardize true labels to mean 0 and std 1
            for j, a in enumerate(data_batch.accession):
                batch_Tm[j] = Tm_dict[a]
            # batch_Tm = data_batch.Tm
            true = (batch_Tm[:pred.numel()] - self.val_mean) / self.val_std
            # true = (batch_Tm - self.val_mean) / self.val_std
            total_loss += self.loss_fn(pred, true).item()

            ### gather model outputs and labels
            # convert predicted values back to Celsius
            outputs[i * batch_size : (i + 1) * batch_size] = (
                pred * self.val_std + self.val_mean
            )
            labels[i * batch_size : (i + 1) * batch_size] = batch_Tm[
                : pred.numel()
            ]
            accessions[i * batch_size : (i + 1) * batch_size] = (
                data_batch.accession
            )

        avg_loss = total_loss / dataset_size

        return avg_loss, outputs, labels, accessions
