import os
import glob


import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.pooling import MaxPool2d
from torch.utils.data import DataLoader, Dataset
import torch.utils
import torchvision
import torchvision.transforms as transforms
from IPython.display import Image
from PIL import Image
from torchvision.models import alexnet, googlenet
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix


def get_mean_and_std(dir_name):

    files = glob.glob(os.path.join(dir_name, '*', '*', '*.jpeg'))
    scaler = StandardScaler(with_mean=True, with_std=True)
    for file in files:
        with open(file, 'rb') as file:
            image = np.asarray(Image.open(file).convert('L'), dtype='float32')/255.0
            scaler.partial_fit(image.reshape(-1, 1))
    
    return scaler.mean_, scaler.scale_

class imageloader(Dataset):
    def __init__(self,
                root_dir: str,
                transform: torchvision.transforms.Compose = None):
        self.root = os.path.expanduser(root_dir)
        self.transform = transform

        classes_list = [directory.name for directory in os.scandir(root_dir) if directory.is_dir()]
        self.class_dict = {classes_list[i]: i for i in range(len(classes_list))}
        

        img_paths = []
        for class_name, class_idx in self.class_dict.items():
            img_dir = os.path.join(root_dir, class_name, '*.jpeg')
            files = glob.glob(img_dir)
            img_paths += [(f, class_idx) for f in files]
        self.dataset = img_paths


    def __getitem__(self, index):
        img = None
        class_idx = None
        filename, class_idx = self.dataset[index]
        with open(filename, 'rb') as f:
            img = Image.open(f)
            img = img.convert('L')
        if self.transform is not None:
            img = self.transform(img)
        
        return img, class_idx

    def __len__(self):
        return len(self.dataset)


def get_basic_transforms(input_size, pixel_mean, pixel_std):
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(pixel_mean, pixel_std)
    ])

def get_augmented_transforms(input_size, pixel_mean, pixel_std):
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08,hue=0.08),
        transforms.RandomRotation(5),
        transforms.RandomResizedCrop(size=input_size, scale = (0.80, 1.05), ratio = (0.99, 1.01)),
        transforms.ToTensor(),
        transforms.Normalize(pixel_mean,pixel_std)
    ])

def predict_label(model_output):
    return torch.argmax(model_output, dim = 1)

def compute_loss(model, model_output, target_labels, noramlize=True):
    loss_criterion = model.loss_criterion
    loss = loss_criterion(model_output, target_labels)
    if noramlize:
        loss /= len(target_labels)
    return loss

def get_optimizer(model, configuration):
    optimizer_type = configuration["optimizer_type"]
    learning_rate = configuration["lr"]
    weight_decay = configuration["weight_decay"]
    momentum  = configuration["momentum"] 

    if optimizer_type =='SGD':
      optimizer = f"torch.optim.{optimizer_type}(model.parameters() , lr={learning_rate},weight_decay={weight_decay}, momentum={momentum})"
  
    else:
      optimizer = f"torch.optim.{optimizer_type}(model.parameters() , lr={learning_rate},weight_decay={weight_decay})"
    return eval(optimizer)
    
class ModifiedAlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')

        pretrained_alexnet = alexnet(pretrained=True)
        for idx in [0, 3, 6, 8, 10]:
            pretrained_alexnet.features[idx].weight.requires_grad = False
        for idx in [1, 4, 6]:
            pretrained_alexnet.classifier[idx].weight.requires_grad = False
            pretrained_alexnet.classifier[idx].weight.requires_grad = False
        self.cnn_layers = pretrained_alexnet.features
        self.cnn_layers.add_module('13',pretrained_alexnet.avgpool)
        self.cnn_layers.add_module('14' , nn.Flatten())
        self.fc_layers = pretrained_alexnet.classifier
        self.fc_layers.add_module('7', nn.Linear(1000,2))
    def forward(self, x_input):
        x_input = x_input.repeat(1, 3, 1, 1)
        cnn_output = self.cnn_layers(x_input)
        fc_output = self.fc_layers(cnn_output)
        return fc_output

class ModifiedGoogleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')
    
        # self.model = googlenet(pretrained=True)
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # self.model.fc = nn.Linear(1024,2)

        pretrained_googlenet = googlenet(pretrained=True)
        
        # freeze_module = [pretrained_googlenet.conv1, pretrained_googlenet.conv2, pretrained_googlenet.conv3,\
        #                     pretrained_googlenet.inception3a, pretrained_googlenet.inception3b, \
        #                     pretrained_googlenet.inception4a, pretrained_googlenet.inception4b, \
        #                     pretrained_googlenet.inception4c, pretrained_googlenet.inception4d, \
        #                     pretrained_googlenet.inception4e]

        freeze_module = [pretrained_googlenet.conv1, pretrained_googlenet.conv2, pretrained_googlenet.conv3,\
                            pretrained_googlenet.inception3a, pretrained_googlenet.inception3b, \
                            pretrained_googlenet.inception4a, pretrained_googlenet.inception4b, \
                            pretrained_googlenet.inception4c, pretrained_googlenet.inception4d]
        for module in freeze_module:
            for params in module.parameters():
                params.requires_grad = False
        self.model = pretrained_googlenet
        self.model.fc = nn.Linear(1024,2)

    def forward(self, x_input):
        x_input = x_input.repeat(1, 3, 1, 1)
        fc_output = self.model(x_input)
        #fc_output = self.fc_layers(cnn_output)
        return fc_output


class Trainer():
    def __init__(self, 
                train_data_dir,
                test_data_dir,
                model, 
                optimizer, 
                model_dir, 
                train_data_transforms, 
                test_data_transforms,
                batch_size=200,
                load_from_disk=True,
                cuda=False):
        self.model_dir = model_dir
        self.model = model
        self.cuda = cuda
        if cuda:
            self.model.cuda()
        dataloader_args = {'num_workers': 1, 'pin_memory': True} if cuda else {}

        self.train_dataset = imageloader(train_data_dir, transform=train_data_transforms)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, 
                                                        batch_size=batch_size, 
                                                        shuffle=True,
                                                        **dataloader_args)

        self.test_dataset = imageloader(test_data_dir, transform=test_data_transforms)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=True,
                                                    **dataloader_args)
        self.optimizer = optimizer
        self.train_loss_list = []
        self.validation_loss_list = []
        self.train_accuracy_list = []
        self.validation_accuracy_list = []

        # load the model from the disk if it exists
        if os.path.exists(model_dir) and load_from_disk:
            checkpoint = torch.load(os.path.join(self.model_dir, 'checkpoint.pt'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.model.train()

    def save_model(self):
        torch.save({'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict()
        }, os.path.join(self.model_dir, 'checkpoint.pt'))

    def train(self, num_epochs):
        self.model.train()
        train_loss, train_acc = self.evaluate(split='train')
        val_loss, val_acc = self.evaluate(split='test')

        self.train_loss_list.append(train_loss)
        self.train_accuracy_list.append(train_acc)
        self.validation_loss_list.append(val_loss)
        self.validation_accuracy_list.append(val_acc)


        print('Epoch:{}, Training Loss:{:.4f}, Validation Loss:{:.4f}'.format(
        0, self.train_loss_list[-1], self.validation_loss_list[-1])
        )

        for epoch_idx in range(num_epochs):
            self.model.train()
            for _, batch in enumerate(self.train_loader):
                if self.cuda:
                    input_data, target_data = Variable(batch[0]).cuda(), Variable(batch[1]).cuda()
                else:
                    input_data, target_data = Variable(batch[0]), Variable(batch[1])

                output_data = self.model(input_data)
                loss = compute_loss(self.model, output_data, target_data)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_loss, train_acc = self.evaluate(split='train')
            val_loss, val_acc = self.evaluate(split='test')

            self.train_loss_list.append(train_loss)
            self.train_accuracy_list.append(train_acc)
            self.validation_loss_list.append(val_loss)
            self.validation_accuracy_list.append(val_acc)

            print('Epoch:{}, Training Loss:{:.4f}, Validation Loss:{:.4f}'.format(
                epoch_idx+1, self.train_loss_list[-1], self.validation_loss_list[-1])
            )

        self.save_model()

    def evaluate(self, split = 'test'):
        self.model.eval()
        count_total = 0 
        count_correct = 0
        loss = 0
        for _, batch in enumerate(self.test_loader if split == 'test' else self.train_loader):
            if self.cuda:
                input_data, target_data = Variable(batch[0]).cuda(), Variable(batch[1]).cuda()
            else:
                input_data, target_data = Variable(batch[0]), Variable(batch[1])
            output_data = self.model(input_data)

            count_total += input_data.shape[0]
            loss += float(compute_loss(self.model,
                                        output_data, target_data, noramlize=False))
            predicted_labels = predict_label(output_data)
            count_correct += torch.sum(predicted_labels == target_data).cpu().item()
        
        self.model.train()
        loss = loss/float(count_total)
        accuracy = float(count_correct)/float(count_total)
        return loss, accuracy

    def plot_loss(self):
        plt.figure()
        count_epoch = range(len(self.train_loss_list))
        
        plt.plot(count_epoch, self.train_loss_list, '-r', label = 'training')
        plt.plot(count_epoch, self.validation_loss_list, '-g', label = 'validation')
        plt.title("Loss Plot against Epochs")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.show()
    
    def plot_accuracy(self):
        plt.figure()
        count_epoch = range(len(self.train_accuracy_list))
        
        plt.plot(count_epoch, self.train_accuracy_list, '-r', label = 'training')
        plt.plot(count_epoch, self.validation_accuracy_list, '-g', label = 'validation')
        plt.title("Accuracy Plot against Epochs")
        plt.legend()
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.show()
    
    def final_evaluate(self):
        self.model.eval()
        
        y_true = []
        y_predicted = [] 
        
        for _, batch in enumerate(self.test_loader):
            if self.cuda:
                input_data, target_data = Variable(batch[0]).cuda(), Variable(batch[1]).cuda()
            else:
                input_data, target_data = Variable(batch[0]), Variable(batch[1])
                
            output_data = self.model(input_data)
            predicted_labels = predict_label(output_data)

            if self.cuda:
                y_true += list(target_data.cpu())
                y_predicted += list(predicted_labels.cpu())
            else:
                y_true += list(target_data)
                y_predicted += list(predicted_labels)
        
        self.model.train()
       
        precision = precision_score(y_true, y_predicted)
        recall = recall_score(y_true, y_predicted)
        f1 = f1_score(y_true, y_predicted)
        print("Confusion Matrix for the Trained Model")
        print(confusion_matrix(y_true, y_predicted),'\n')
        
        return precision, recall, f1
        
        

        



