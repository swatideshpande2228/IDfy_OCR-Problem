import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(filepath_or_buffer = "D:\Interview\IDfy\dataset.csv", sep = ',', header = None)

#seperating target & feature variables 
Images = df.iloc[:,:-1].values 
Labels = df.iloc[:,-1].values

print("Total Samples :", len(Images))

X_train, X_test, y_train, y_test = train_test_split(Images, Labels, train_size = 0.8, test_size = 0.2, shuffle=False)

print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

imageH = 32
imageW = 128

transform_train = transforms.Compose([transforms.Resize([imageH, imageW]),   #creates 96x96 image
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])
                                    ])


transform_test = transforms.Compose([transforms.Resize([imageH, imageW]),   #creates 96x96 image
                                    transforms.ToTensor(),   #converts the image to a Tensor
                                    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

Train_dir="D:/Interview/IDfy/train"
Test_dir="D:/Interview/IDfy/test"

Train_data = datasets.ImageFolder(Train_dir,       
                    transform=transform_train)

Test_data = datasets.ImageFolder(Test_dir,
                   transform=transform_test)

batch_size = 1
train_load = torch.utils.data.DataLoader(dataset = Train_data, 
                                         batch_size = batch_size,
                                         shuffle = False)

test_load = torch.utils.data.DataLoader(dataset = Test_data,
                                        batch_size = batch_size,
                                       shuffle = False)   

#Show a batch of images
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.detach().numpy()
    plt.figure(figsize=(20,20))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

epoch = 10
learning_rate = 0.1

#SOURCE: https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv5 = nn.Conv2d(32, 1, kernel_size=3)
        self.relu = nn.ReLU()
        self.MaxPool = nn.MaxPool2d(kernel_size=3, stride = 2)
        self.lstm1 = nn.LSTM(56, 32, 2, bidirectional=True)
        
    def forward(self, x):
        out_A = self.conv1(x)
        out_A = self.conv2(out_A)
        out_A = self.conv3(out_A)
        out_A = self.conv4(out_A)
        out_A = self.conv5(out_A)
        out_A = self.relu(out_A)
        out_A = self.MaxPool(out_A)
        out_A = out_A[:,0,:,:]
        out_A = self.lstm1(out_A)
        return out_A  
    
def Convert(string): 
    list1=[] 
    list1[:0]=string
    return list1 

def extractDigits(lst): 
    return [[el] for el in lst]

def Target_preprocess(target):
    max_l = 0
    ts_list = []
    for w in target:
        ts_list.append(torch.ByteTensor(list(bytes(w, 'utf8'))))
        max_l = max(ts_list[-1].size()[0], max_l)
        
    w_t = torch.zeros((len(ts_list), max_l), dtype=torch.uint8)
    for i, ts in enumerate(ts_list):
        w_t[i, 0:ts.size()[0]] = ts
    return w_t

print('Building the model...')
    
Net = NeuralNetwork()
print(Net)

train_loss = []
Test_loss = []

#Loss function Net
criterion_netA = nn.CTCLoss(zero_infinity=False, reduction='none')
optimizer_netA = optim.SGD(Net.parameters(), lr=learning_rate)

print('\nTraining...')
Y0_train = iter(y_train)
Y0_test = iter(y_test)

for ep in range(epoch):   
    iterations = 0
    total_loss1 = 0.0   
    test_loss = 0.0
    val_iterations = 0
    
    target = next(Y0_train)
    targettest = next(Y0_test)
    for labels, images in enumerate(train_load):
        
        Net.train()
        inp = images[0]
        
        optimizer_netA.zero_grad()             #Clears old gradients from last step
        
        output, (hidden, cell) = Net(inp)
        output = output.log_softmax(1)
        output = output.unsqueeze(-2)
        output = output[0,:,:,:]
        
        target = Convert(target)
        targetTensors = Target_preprocess(target)
        targetTensors = targetTensors.unsqueeze(-3)
        targetTensors = targetTensors[:,:,0]
        
        input_lengths = torch.full(size=(1,), fill_value=8, dtype=torch.long)
        Target_lengths = torch.randint(low=3, high=7, size=(1,), dtype=torch.long)
        
        loss = criterion_netA(output, targetTensors, input_lengths, Target_lengths)
        total_loss1 += loss.item()
        loss.backward()
        optimizer_netA.step()                #Updates the weights
        iterations += 1
    
    train_loss.append(total_loss1/iterations)
    
    Net.eval()
    for labels, test_images in enumerate(test_load):
        inptest = test_images[0]
        
        outputs, (hidden, cell) = Net(inptest)
        outputs = outputs.log_softmax(1)
        outputs = outputs.unsqueeze(-2)
        outputs = outputs[0,:,:,:]
        
        target_test = Convert(targettest)
        targetTensorstest = Target_preprocess(target_test)
        targetTensorstest = targetTensorstest.unsqueeze(-3)
        targetTensorstest = targetTensorstest[:,:,0]
        
        testloss = criterion_netA(outputs, targetTensorstest, input_lengths, Target_lengths)
        test_loss += testloss.item()
        val_iterations += 1
        
    Test_loss.append(test_loss/val_iterations)
        
    print('\nEpoch :',ep+1, 'Train_Loss :', train_loss[ep], 'Test_Loss :', Test_loss[ep])
print('Fineshed Training')

plt.figure(figsize=(5,5))
plt.plot(range(epoch), train_loss, label = 'Train Loss')
plt.plot(range(epoch), Test_loss, label = 'Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show