# Defining a class for the feedforward network.

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        '''
        super().__init__()
        # Add the first layer, input to a hidden layer.
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers.
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output log softmax.
            Arguments
            ---------
            self: all layers
            x: tensor vector
        '''
        
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout.        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)

        x = self.output(x)
        
        return F.log_softmax(x, dim=1)




#TensorBoard
!pip install tensorboardcolab
from tensorboardcolab import TensorBoardColab





# Defining a function for the validation pass.

def validation(model, validloader, criterion, device):
    ''' Builds a feedforward network with arbitrary hidden layers, 
        returns the validation loss and  validation accuracy.
        
        Arguments
        ---------
        model: the pre-trained model.
        validloader: generator, the validation dataset.
        criterion: loss function.
        device: the used device for the training [GPU or CPU].
    '''
    # Initiate the validation accuracy & validation loss with 0.
    valid_accuracy = 0
    valid_loss = 0
    # Move model to the device
    model.to(device)
    
    # Looping through the data batches.
    for inputs, labels in validloader:
        # Move input and label tensors to the device.
        inputs, labels = inputs.to(device), labels.to(device)
        
        #inputs.resize_(inputs.shape[0], 48*48)

        # Forward pass through the network.
        output = model.forward(inputs)
        # Increase the validation loss by the loss of the predicted output with the labels.
        valid_loss += criterion(output, labels).item()

        ## Calculating the accuracy 
        # Model's output is log-softmax, so take exponential to get the probabilities.
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label.
        equality = (labels.data == ps.max(dim=1)[1])
        # Accuracy is number of correct predictions divided by all predictions, so we just take the mean.
        valid_accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, valid_accuracy





# Defining a function for the training process

def training(model, criterion, optimizer, device, trainloader, validloader, epochs=5, print_every=40*6):
    ''' Builds a feedforward network with arbitrary hidden layers.
        
        Arguments
        ---------
        model: the pre-trained model.
        optimizer: which we will take a step with it to update the weights.
        criterion: loss function.
        device: the used device for the training [GPU or CPU].
        trainloader: generator, the training dataset.
        validloader: generator, the validation dataset.
        epochs: integer, number of trainings.
        print_every: integer, printing the updates on loss & accuracy every print_every value.
    '''
    
    steps = 0
    running_loss = 0
  
    train_l=[]
    valid_l=[]
    tb = TensorBoardColab()
  
    # Move model to the device
    model.to(device)
    
    for e in range(epochs):
        # Model in training mode, dropout is on.
        model.train()
        for inputs, labels in trainloader:
            # Move input and label tensors to the device.
            inputs, labels = inputs.to(device), labels.to(device)
            
            steps += 1
            
            # zero-ing the accumalated gradients.
            optimizer.zero_grad()

            # Forward pass through the network
            output = model.forward(inputs)
            # Calculate the loss
            loss = criterion(output, labels)
            # Backward pass through the network 
            loss.backward()
            # Take a step with the optimizer to update the weights
            optimizer.step()
            
            running_loss += loss.item()
            train_l.append(running_loss)

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()
                
                # Turn off gradients for validation saves memory and computations, so will speed up inference
                with torch.no_grad():
                    valid_loss, valid_accuracy = validation(model, validloader, criterion, device)
                valid_l.append(valid_loss)
                
                
                # Displaying the validation loss and accuracy during the training. 
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(valid_accuracy/len(validloader)))
            
                running_loss = 0
                
                # Make sure dropout and grads are back on for training
                model.train()
            
   
                # Recording to Tensorboard -------------------------------------------------
 
                tb.save_value('Valid_Loss', 'valid_loss', steps, loss.item())
                tb.save_value('Training_Loss', 'running_loss', steps, loss.item())
                tb.save_value('Valid_Accuracy', 'valid_accuracy', steps, loss.item())
    
                
                
    return valid_l,train_l







# Using a Pretrained Network
model = models.vgg19(pretrained=True)
model



print (model.classifier)

# Setting all hyper parameters in a dictionary to ease the dealing.
hyper_parameters = {'input_size': 25088,
                    'output_size': 7,
                    'hidden_layers': [1024],
                    'drop_p': 0.2,
                    'learn_rate': 0.0001,
                    'epochs': 25,
                    'model': 'vgg19'
                   }



# Freezing the parameters so we don't backprop through them, 
# we will backprop through the classifier parameters only later
for param in model.parameters():
    param.requires_grad = False

# Creating Feedforward Classifier
classifier = Network(input_size = hyper_parameters['input_size'], 
                     output_size = hyper_parameters['output_size'], 
                     hidden_layers = hyper_parameters['hidden_layers'], 
                     drop_p = hyper_parameters['drop_p'])

model.classifier = classifier



# Define the criterion (Loss function). 
criterion = nn.NLLLoss()
# Define the optimizer. Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=hyper_parameters['learn_rate'])


#Training the model
valid, train = training(model, criterion, optimizer, device,
                        trainloader = dataloaders['trainloader'],
                        validloader = dataloaders['validloader'],
                        epochs = hyper_parameters['epochs'])