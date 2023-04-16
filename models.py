"""
For documentation of different layers, please refer to torch.nn
https://pytorch.org/docs/stable/nn.html
"""

import torch
from torch import nn


class OneLayerNN(nn.Module):

    def __init__(self, input_features=11):
        """
        Initializes a liner layer.
        :param input_features: The number of features of each sample.
        """
        super().__init__()
        self.layer = torch.nn.Linear(input_features, 1)

        # TODO: Initialize a linear layer. HINT: torch.nn.Linear

    def forward(self, X):
        """
        Applies the linear layer defined in __init__() to input features X.
        :param X: 2D torch tensor of shape [n, 11], where n is batch size.
            Represents features of a batch of data.
        :return: 2D torch tensor of shape [n, 1], where n is batch size.
            Represents prediction of wine quality.
        """

        # TODO: Apply the linear layer defined in __init__() to input features X
        return self.layer(X)


class TwoLayerNN(nn.Module):

    def __init__(self, input_features=11):
        """
        Initializes model layers.
        :param input_features: The number of features of each sample.
        """
        super().__init__()

        # TODO: Tune the hidden size hyper-parameter
        self.hidden_size = 32

        # TODO: Initialize a linear layer. HINT: torch.nn.Linear
        self.layer1 = torch.nn.Linear(input_features, self.hidden_size)

        # TODO: Initialize a sigmoid activation layer. HINT: torch.nn.Sigmoid
        self.activation = torch.nn.Sigmoid()

        # TODO: Initialize another linear layer
        self.layer2 = torch.nn.Linear(self.hidden_size, 1)

        self.model = torch.nn.Sequential(self.layer1, self.activation, self.layer2)

    def forward(self, X):
        """
        Applies the layers defined in __init__() to input features X.
        :param X: 2D torch tensor of shape [n, 11], where n is batch size.
            Represents features of a batch of data.
        :return: 2D torch tensor of shape [n, 1], where n is batch size.
            Represents prediction of wine quality.
        """

        # TODO: Apply the layers defined in __init__() to input features X
        return self.model(X)


class CNN(nn.Module):

    def __init__(self, input_channels=1, class_num=10):
        """
        Initializes model layers.
        :param input_channels: The number of features of each sample.
        :param class_num: The number of categories.
        """
        super().__init__()

        # TODO: Initialize convolution layers, activation layers, a flatten layer,
        #  and linear layers. In between convolution and linear layers, you need to
        #  apply the flatten layer to transform a 4D tensor to a 2D tensor.

        # Be careful about the calculation of in_features for the linear layer,
        # If the shape of the output of the last convolution layer is [N, C, H, W],
        # then in_features of the first linear layer would be C * W * H.

        # You are free to create your model. If things don't work,
        # please refer to the suggest architecture in the handout.

        # HINT: torch.nn.Conv2d, torch.nn.Linear, torch.nn.Flatten,
        #       torch.nn.Sigmoid, torch.nn.ReLU,
        self.model = nn.Sequential(
            torch.nn.Conv2d(input_channels, 24, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(24, 32, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 32),
            torch.nn.Sigmoid(),
            torch.nn.Linear(32, class_num)
        )

    def forward(self, X):
        """
        Applies the layers defined in __init__() to input features X.
        :param X: 4D torch tensor of shape [n, 1, 8, 8], where n is batch size.
            Represents a batch of 8 * 8 gray scale images.
        :return: 2D torch tensor of shape [n, 10], where n is batch size.
            Represents logits of different categories.
        """

        # TODO: Apply the layers defined in __init__() to input features X
        return self.model(X)


def train(model, dataloader, loss_func, optimizer, num_epoch, correct_num_func=None, print_info=True):
    """
    Trains the model for `num_epoch` epochs.
    :param model: A deep model.
    :param dataloader: Dataloader of the training set. Contains the training data equivalent to ((Xi, Yi)),
        where (Xi, Yi) is a batch of data.
        X: 2D torch tensor for UCI wine and 4D torch tensor for MNIST.
        X: 2D torch tensor for UCI wine and 1D torch tensor for MNIST, containing the corresponding labels
            for each example.
        Refer to the Data Format section in the handout for more information.
    :param loss_func: An MSE loss function for UCI wine and a cross entropy loss for MNIST.
    :param optimizer: An optimizer instance from torch.optim.
    :param num_epoch: The number of epochs we train our network.
    :param correct_num_func: A function to calculate how many samples are correctly classified.
        You need to implement correct_predict_num() below.
        To train the CNN model, we also want to calculate the classification accuracy in addition to loss.
    :param print_info: If True, print the average loss (and accuracy, if applicable) after each epoch.
    :return:
        epoch_average_losses: A list of average loss after each epoch.
            Note: different from HW10, we will return average losses instead of total losses.
        epoch_accuracies: A list of accuracy values after each epoch. This is applicable when training on MNIST.
    """

    # TODO: Initializing variables
    # Initialize an empty list to save average losses in all epochs.
    avg_loss = []

    # Initialize an empty list to save accuracy values in all epochs.
    acc_val = []

    # TODO: Tell the model we are in the training phase. HINT: model.train()
    # This is useful if you use batch normalization or dropout
    # because the behavior of these layers in the training phase is different from testing phase.
    model.train(True)

    # train network for num_epochs
    for epoch in range(num_epoch):

        # Initializing variables

        # Sum of losses in an epoch. Will be used to calculate average loss.
        # The reason we are using (epoch_loss_sum / #samples in each batch) to calculate the
        # average loss is that the number of samples in the last batch may be fewer than your batch_size.
        epoch_loss_sum = 0

        # Sum of the number of correct predictions. Will be used to calculate average accuracy for CNN.
        epoch_correct_num = 0

        # TODO: Iterate through batches. HINT: for X, Y in dataloader:
        for X, Y in dataloader:

            # TODO: Run a forward pass and get model output
            output = model.forward(X)

            # TODO: Set all gradients to zero by calling optimizer.zero_grad()
            optimizer.zero_grad()

            # TODO: Calculate loss of this batch
            loss = loss_func(output, Y)

            # TODO: Run a backward pass by calling loss.backward(),
            #  where loss is the output of the loss function.
            loss.backward()

            # TODO: Update parameters by calling optimizer.step()
            optimizer.step()

            # TODO: Increase epoch_loss_sum by (loss * #samples in the current batch)
            #  Use loss.item() to get the python scalar of loss value because the output of
            #   loss function also contains gradient information, which takes a lot of memory.
            #  Use X.shape[0] to get the number of samples in the current batch.
            epoch_loss_sum += loss.item() * X.shape[0]

            
            # TODO: Calculate the number of correct predictions for CNN on MNIST
            num_correct_pred = correct_predict_num(output, Y)

            # TODO: When correct_num_func is not None,
            #  increase epoch_correct_num by #correct predictions in the current batch
            if correct_num_func != None:
                epoch_correct_num += num_correct_pred

        # TODO: Append the average loss of the current epoch to your list.
        #  You can get the number of training samples by len(dataloader.dataset)
        avg_loss.append(epoch_loss_sum / len(dataloader.dataset))

        # TODO: When correct_num_func is not None,
        #  calculate average accuracy for CNN on MNIST if correct_num_func:
        #  Append the average accuracy of the current epoch to your list.
        #  You can get the number of training samples by len(dataloader.dataset)
        if correct_num_func != None:
            acc_val.append(epoch_correct_num / len(dataloader.dataset))
        
        # Print the loss after every epoch. Print accuracies if specified
        if print_info:
            print('Epoch: {} | Loss: {:.4f} '.format(epoch, epoch_loss_sum / len(dataloader.dataset)), end="")
            if correct_num_func:
                print('Accuracy: {:.4f}%'.format(epoch_correct_num / len(dataloader.dataset) * 100), end="")
            print()

    # TODO: When correct_num_func is None, only return a list of average losses.
    #  When correct_num_func is not None, return a list of average losses and a list of accuracies.
    if correct_num_func == None:
        return avg_loss
    else:
        return avg_loss, acc_val


def test(model, dataloader, loss_func, correct_num_func=None):
    """
    Tests the model.
    :param model: A deep model.
    :param dataloader: Dataloader of the testing set. Contains the testing data equivalent to ((Xi, Yi)),
        where (Xi, Yi) is a batch of data.
        X: 2D torch tensor for UCI wine and 4D torch tensor for MNIST.
        X: 2D torch tensor for UCI wine and 1D torch tensor for MNIST, containing the corresponding labels
            for each example.
        Refer to the Data Format section in the handout for more information.
    :param loss_func: An MSE loss function for UCI wine and a cross entropy loss for MNIST.
    :param correct_num_func: A function to calculate how many samples are correctly classified.
        You need to implement correct_predict_num() below.
        To test the CNN model, we also want to calculate the classification accuracy in addition to loss.
    :return:
        Average loss.
        Average accuracy. This is applicable when testing on MNIST.
    """
    """
    :param dataloader: Contains the training data equivalent to ((X, Y))
        :param X: 2D Numpy array where each row contains an example
        :param Y: 1D Numpy array containing the corresponding values for each example
    :param loss_func: An MSE loss function from the Pytorch Library
    :return: epoch loss and accuracies to be graphed
    """

    # TODO: Initalizing variables
        # Initialize sum of losses in an epoch. Will be used to calculate average loss.
        # Initialize sum of the number of correct predictions. Will be used to calculate average accuracy for CNN.
    sum_loss = 0
    sum_corr = 0

    # TODO: Tell the model we are in the testing phase. HINT: model.eval()
    model.eval(True)

    # TODO: During testing, we don't need to calculate gradients. HINT: use 'with torch.no_grad():'
    with torch.no_grad():

        # TODO: Iterate through batches.
        for X, Y in dataloader:

            # TODO: Run a forward pass and get model output
            output = model.forward(X)

            # TODO: Calculate loss of this batch
            loss = loss_func(output, Y)

            # TODO: Increase loss sum by (loss * #samples in the current batch)
            #  Use loss.item() to get the python scalar of loss value.
            #  Use X.shape[0] to get the number of samples in the current batch.
            sum_loss += loss.item() * X.shape[0]

            # TODO: When correct_num_func is not None, calculate the number of correct predictions for CNN on MNIST
            #  Increase the total number of correct predictions by #correct predictions in the current batch
            num_correct_pred = correct_predict_num(output, Y)
            if correct_num_func != None:
                sum_corr += num_correct_pred
                
    # TODO: When correct_num_func is None, return average loss.
    #  When correct_num_func is not None, return average loss and accuracy.
    if correct_num_func == None:
        return sum_loss / len(dataloader.dataset)
    else:
        return sum_loss / len(dataloader.dataset), sum_corr / len(dataloader.dataset)


def correct_predict_num(logit, target):
    """
    Returns the number of correct predictions.
    :param logit: 2D torch tensor of shape [n, class_num], where
        n is the number of samples, and class_num is the number of classes (10 for MNIST).
        Represents the output of CNN model.
    :param target: 1D torch tensor of shape [n],  where n is the number of samples.
        Represents the ground truth categories of images.
    :return: A python scalar. The number of correct predictions.
    """
    # TODO: Calculate the number of correct predictions.
    # HINT: torch.sum, torch.argmax
    # You may need .long() to convert a torch tensor to LongTensor.
    # Use .item() to convert a torch tensor of size 1 to python scalar.
    return torch.sum(torch.argmax(logit, dim=1) == target).item()
