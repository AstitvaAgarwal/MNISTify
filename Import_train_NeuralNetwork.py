import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets 
from torchvision.transforms import ToTensor

BATCH_SIZE = 128    
EPOCHS =10
LEARNING_RATE = .001


class FeedForwardNet(nn.Module):       #constructor
    def __init__(self):
        super().__init__()              #invoking constructor 
        self.flatten = nn.Flatten()                 #this layer will flatten the images --> reshapes into 1D tensors
        self.dense_layers = nn.Sequential( 
            nn.Linear(28*28,256),        #equivalent to dense layer
            nn.ReLU(),                   #activation function 
            nn.Linear(256,10)           #another linear layer
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_data):   #specifies the data flow  || how to manupilate the data
        flattened_data = self.flatten(input_data)     #passing the input data to flatten layers to get the data flattened 
        logits = self.dense_layers(flattened_data)    #passing flattened data to dense layer and getting output
        predictions = self.softmax(logits)            #passing the output to softmax layer to get the predictions
        return predictions 
        
def download_mnist_datasets():
    train_data = datasets.MNIST(
        root = "data",               # where to store the data
        download = True,             #if not downloaded then download it
        train = True,                #Train set part of this dataset
        transform = ToTensor()     #transfrorming it to tensors
        )
    
    validation_data = datasets.MNIST(     #same for validation set
        root = "data",
        download = True,
        train = False,                  #False bcoz its not for training set
        transform = ToTensor()
        )
    return train_data, validation_data 


#training the model :--
def train_one_epoch(model , data_loader , loss_fn, optimiser, device):          #training one epoch of your model
#loop through all the samples in the dataset and in each iterations will get a new batch of samples
    for inputs , targets in data_loader:                                         

        inputs, targets = inputs.to(device) , targets.to(device)  #assigning
        
        predictions = model(inputs)
        loss = loss_fn(predictions , targets)    #calculating loss  --> will compare these two and come up with the loss
        
        
        #at every iteration the optimiser is gonna claculate a gradience that will need to decide the updation of weights , so the gradience at each iteration gets saved
        optimiser.zero_grad()             #at each training iteration resets the gradience to zero so to start from scratch
        
        loss.backward()                   #backpropogating loss
        optimiser.step()                  #update weights
        
    print(f"loss:{loss.item()}")


def train(model , data_loader , loss_fn, optimiser, device ,epochs):       #this fnc will go through all the epochs that we want to train the model for 
    for i in range(epochs):
        print(f"Epochs {i+1}")
        train_one_epoch(model,data_loader,loss_fn ,optimiser , device)
        print("-------------------------------")
    print("training is done")

#End of training




if __name__ == "__main__":
    train_data,_ = download_mnist_datasets()     #downloading MNIST dataset
    print("MNIST dataset downloaded")

    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE )
    
    #------------cheking the device -------------------
    if torch.cuda.is_available():
        device = "cuda"
    else :
        device = "cpu"
    print(f"Using {device} device")
    #-------------------------------------------------
    
    feed_forward_net = FeedForwardNet().to(device)     #assigning the FeedForwardNet to a device || Device --> CPU or GPU 
    
    
    loss_fn =nn.CrossEntropyLoss()                      
    optimiser = torch.optim.Adam(feed_forward_net.parameters() , lr = LEARNING_RATE)              
    
    train(feed_forward_net, train_data_loader, loss_fn, optimiser , device , EPOCHS)         #train model
    
    
    #storing after training  ||   state_dict() --> python dictionary that has all imp info about layers and parameters that have been trained 
    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")                          
    print("Model trained and stored at feedforwardnet.pth") 
    
    
    