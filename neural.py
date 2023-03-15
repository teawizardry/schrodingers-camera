import torch.nn as nn
import copy
import torch
from torch.utils.data import Dataset

class PrivateModel(nn.Module):

    def __init__(self):
        super(PrivateModel, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=20,            
                kernel_size=3,              
                stride=1,                   
                padding=2,                  
            ),                            
            nn.ReLU(),                          
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(20, 20, 3, 1, 0),     
            nn.ReLU(),                                      
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(20, 22, 3, 1, 6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(22, 10, 3, 2, 5),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(10, 10, 3, 1, 4),
            nn.ReLU(),
        )
        # fully connected layer, output 36 classes
        self.out = nn.Linear(10 * 17 * 17, 36)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)     
        output = self.out(x)
        return output, x    # return x for visualization

# class PublicModel(nn.Module):

#     def __init__(self):
#         super(PublicModel, self).__init__()
#         self.conv1 = nn.Sequential(         
#             nn.Conv2d(
#                 in_channels=1,              
#                 out_channels=20,            
#                 kernel_size=3,              
#                 stride=1,                   
#                 padding=2,                  
#             ),                            
#             nn.ReLU(),                          
#         )
#         self.conv2 = nn.Sequential(         
#             nn.Conv2d(20, 20, 3, 1, 0),     
#             nn.ReLU(),                                      
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(20, 22, 3, 1, 6),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(22, 10, 3, 2, 5),
#             nn.ReLU(),
#         )
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(10, 10, 3, 1, 4),
#             nn.ReLU(),
#         )
#         # fully connected layer, output 36 classes
#         self.out = nn.Linear(10 * 17 * 17, 2)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         x = x.view(x.size(0), -1)     
#         output = self.out(x)
#         return output, x    # return x for visualization

class PublicModel(nn.Module):

    def __init__(self):
        super(PublicModel, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=20,            
                kernel_size=3,              
                stride=1,                   
                padding=2,                  
            ),                            
            nn.ReLU(),                         
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(20, 20, 3, 1, 0),     
            nn.ReLU(),                                      
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(20, 22, 3, 1, 6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(22, 10, 3, 2, 5),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(10, 10, 3, 1, 4),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(10, 5, 3, 2, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(5, 5, 2, 1, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # fully connected layer, output 2 classes
        self.out = nn.Linear(5 * 7 * 7, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = x.view(x.size(0), -1)     
        output = self.out(x)
        return output, x    # return x for visualization


class PrintShape(nn.Module):
    def __init__(self):
        super(PrintShape, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x


class AgentNet(nn.Module):
    
    '''mini cnn structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    '''
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h = input_dim

        self.online = nn.Sequential(
            nn.Linear(h, output_dim)
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == 'online':
            return self.online(input)
        elif model == 'target':
            return self.target(input)

# class AgentNet_2(nn.Module):
    
#     '''mini cnn structure
#     input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
#     '''
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         c, h = input_dim

#         self.online = nn.Sequential(
#             nn.Conv1d(c, c, 3, 1, 0),
#             nn.ReLU(),
#             nn.Linear(3133, output_dim)
#         )

#         self.target = copy.deepcopy(self.online)

#         # Q_target parameters are frozen.
#         for p in self.target.parameters():
#             p.requires_grad = False

#     def forward(self, input, model):
#         if model == 'online':
#             out = self.online(input)
#             # print("out size: ", out.size())
#             return out
#         elif model == 'target':
#             out = self.target(input)
#             # print("out size: ", out.size())
#             return out

class EmnistCustomDataset(Dataset):
    """EMNIST numbers and uppercase letters dataset."""

    def __init__(self, labels, second_labels, images, transform=None):
        """
        Args:
        labels (array): array of the labels
        images (array): image array
        transform (callable, optional): optional transform to be applied on a sample.
        """
        self.labels = labels
        self.second_labels = second_labels
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx,:,:]
        label = self.labels[idx]
        second_label = self.second_labels[idx]
        sample = {'image': image, 'label': label, 'second_label': second_label}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample