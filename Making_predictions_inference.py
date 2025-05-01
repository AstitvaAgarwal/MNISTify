import torch
from Import_train_NeuralNetwork import FeedForwardNet, download_mnist_datasets #importing facilities built before 

class_mapping = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
    ]

def predict (model, input , target , class_mapping):
    model.eval()   #this method changes how the pytorch model behave --> if turned on certain layers like dropout, normalization ,etc gets off bcoz not needed in evaluation
    with torch.no_grad():           #context manager --> the model doesnt calc any gradience 
        predictions = model(input)            #2D Tensor (1,10) --> [ [0.1, 0.01, ....,0.6] -->sum == 1(bcoz of softmax) ||||| (1)-->no. of samples passed , (10) --> no. of classes that the model tries to predict 
        predicted_index = predictions[0].argmax(0)       
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected
        
    

if __name__ == "__main__":
    
    
    #loading back the model
    feed_forward_net = FeedForwardNet()
    state_dict = torch.load("feedforwardnet.pth")
    feed_forward_net.load_state_dict(state_dict)
    
    
    #loading MNIST validation dataset
    _, validation_data = download_mnist_datasets()
    
    
    #get a sample from the validation dataset for inference 
    input , target = validation_data[0][0] , validation_data[0][1]
    
    
    #make an inference 
    predicted, expected = predict(feed_forward_net, input , target, class_mapping)
    
    
    print(f"Predicted: '{predicted}', expected: '{expected}'")  