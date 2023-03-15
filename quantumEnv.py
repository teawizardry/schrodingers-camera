import gym
from gym import spaces
import numpy as np
from typing import Optional
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer
from qiskit import transpile
from qiskit.tools.jupyter import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import wandb
from neural import PrivateModel, PublicModel, EmnistCustomDataset
from QiskitHelper import FRQIHelper
import time
from math import comb
import pickle

class QuantumEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=False, test=False):
        self.action_space = spaces.Discrete(comb(28,1)+comb(28,2)+comb(28,3)+comb(28,4)) # 24157
        self.observation_space = spaces.Tuple(
            (spaces.Box(low=0, high=255, shape=(16,16)), spaces.Discrete(36), spaces.Discrete(2)) # image, private, public
        )
        self.action_list = range(28)
        self.render_mode = render_mode

        # load in images and labels
        if test == True:
            labels_train = np.load('./resources/emnist-test-labels.npy').astype(np.uint8)
            second_labels = np.load('./resources/emnist-test-second-labels.npy').astype(np.uint8)
            images_train = np.load('./resources/emnist-test-images.npy')
        else:
            labels_train = np.load('./resources/emnist-train-labels.npy').astype(np.uint8)
            second_labels = np.load('./resources/emnist-train-second-labels.npy').astype(np.uint8)
            images_train = np.load('./resources/emnist-train-images.npy')

        self.emnist_custom_dataset = EmnistCustomDataset(labels=labels_train, second_labels=second_labels, images=images_train, transform=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()]))
        self.trainloader = iter(DataLoader(self.emnist_custom_dataset, batch_size=1, shuffle=True))

        # select GPU / CPU
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device_cnn = 'cuda:1' if torch.cuda.is_available() else 'cuda:0'

        # load model and the weights
        self.private_model = PrivateModel()
        self.public_model = PublicModel()

        self.private_model.load_state_dict(torch.load('./resources/private_model.pt'))
        self.private_model.to(self.device_cnn);
        self.public_model.load_state_dict(torch.load('./resources/public_model.pt'))
        self.public_model.to(self.device_cnn);

        self.private_correct = 0
        self.public_correct = 0
        self.total = 0
        self.total_ep = 0

        # loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # set up vars
        self.meas_img = np.zeros((16,16))
        self.private_prediction = 0
        self.public_prediction = 0
        self.meas_img_np = np.zeros((1,1,16,16), dtype=np.float32)
        self.meas_img_t = torch.from_numpy(self.meas_img_np).to(self.device_cnn)
        self.pub_x = torch.zeros([1, 245]).to(self.device_cnn)
        self.priv_x = torch.zeros([1, 2890]).to(self.device_cnn)
        self.flag = False
        self.step_number = 0

        self.actions_per_ep = 0
        self.mean_reward = 0

        # set up quantum
        self.frqi = FRQIHelper()
        self.c = QuantumRegister(1, 'c')
        self.p = QuantumRegister(8, 'p')
        self.cl = ClassicalRegister(9)
        self.circ = QuantumCircuit(self.c, self.p, self.cl)

    def step(self, action):
        assert self.action_space.contains(action), 'action should be within action space'
        done = False
        reward = 0
        # action = 8 # no redaction #27 # max #

        # start timer
        start = time.time()

        # get the image
        try:
            data = next(self.trainloader)
            images = data['image']
            self.labels = data['label']
            second_labels = data['second_label']
            done = False
        except StopIteration:
            self.flag = True
            done = True
        
        if done is False:
            for img_num in range(images.size(dim=0)):    
                # resize
                image = images[img_num, 0, :].numpy()
                
                # set up quantum
                self.c = QuantumRegister(1, 'c')
                self.p = QuantumRegister(8, 'p')
                self.cl = ClassicalRegister(9)
                self.circ = QuantumCircuit(self.c, self.p, self.cl)

                # frqi
                self.frqi.frqi_encoder(self.circ, self.p, self.c, image.flatten(), action)
                self.circ.barrier()

                # measure the image
                self.circ.measure(self.c, self.cl[0])
                self.circ.measure(self.p, self.cl[1:9])

                # simulation
                aer_sim = Aer.get_backend('aer_simulator')
                aer_sim.set_options(device='GPU')
                t_qc = transpile(self.circ, aer_sim)
                shots = 4096
                result = aer_sim.run(t_qc, shots=4096).result()
                counts = result.get_counts(self.circ)

                # generated image
                self.meas_img = self.frqi.frqi_decode(shots, counts, len(image)**2, inverse_norm=True)
                self.meas_img_np[img_num,0,:,:] = self.meas_img
                self.meas_img_t = torch.from_numpy(self.meas_img_np).to(self.device_cnn)

            # cnns
            with torch.no_grad():
                outputs, self.pub_x = self.public_model(self.meas_img_t)
                # public_loss = self.loss_fn(outputs[0], second_labels.to(self.device_cnn))
                # print(outputs[0])
                # print(public_loss)
                # print(self.pub_x.shape)
                _, self.public_prediction = torch.max(outputs, 1)

                if(self.public_prediction.cpu()==second_labels):
                    self.public_correct += 1
                    # reward = 100
                    reward = 1000
                    wandb.log({"Public Accuracy step": 1})
                else:
                    wandb.log({"Public Accuracy step": 0})
            
                outputs, self.priv_x = self.private_model(self.meas_img_t)
                # private_loss = self.loss_fn(outputs[0], labels.to(self.device_cnn))
                _, self.private_prediction = torch.max(outputs, 1)  

                if(self.private_prediction.cpu()==self.labels):
                    self.private_correct += 1
                    # reward -= 100
                    wandb.log({"Private Accuracy step": 1})
                else: 
                    wandb.log({"Private Accuracy step": 0})

                self.total += 1
                self.total_ep += 1
                self.mean_reward += reward
                wandb.log({"reward": reward})
                wandb.log({"action": action})
                
                # log
                wandb.log({"actions per step": self.frqi.num_of_actions})
                self.actions_per_ep += self.frqi.num_of_actions

        # end timer
        end = time.time()
        wandb.log({"Private Accuracy running average": self.private_correct/self.total, "Public Accuracy running average": self.public_correct/self.total})

        # done condition
        # if (self.private_correct/self.total) > 0.03 and self.total_ep > 3:
        #     done = True
        #     print("Private Prediction too good!")
        if (self.public_correct/self.total) < 0.8 and self.total_ep > 3:
            done = True
            print("Public prediction not good!")
        else:
            pass
            # reward += self.total_ep * 1000 
            # reward += (self.public_correct/self.total) * 1000

        if done is True:
            info = [(self.private_correct/self.total), (self.public_correct/self.total)]
            print(f"Done! \nPrivate Accuracy: {info[0]}, Public Accuracy: {info[1]}")
            wandb.log({"Private Accuracy": info[0], "Public Accuracy": info[1]})
            wandb.log({"actions per episode": self.actions_per_ep})
            wandb.log({"mean reward": self.mean_reward})
        else:
            info = [self.public_prediction, self.private_prediction, self.labels]
            print(f"Step {self.step_number} Length: {(end-start)/60} mins")
            print(info)
            self.step_number += 1

        return self._get_obs(), reward, done, info, #.to(self.device)

    def _get_obs(self):
        # return self.meas_img_t.to(self.device)
        return torch.cat((self.pub_x, self.priv_x), 1).to(self.device).squeeze()

    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)

        # reset dataloader
        self.trainloader = iter(DataLoader(self.emnist_custom_dataset, batch_size=1, shuffle=True))

        # set up vars
        self.meas_img = np.zeros((16,16))
        self.private_prediction = 0
        self.public_prediction = 0
        self.meas_img_np = np.zeros((1,1,16,16), dtype=np.float32)
        self.meas_img_t = torch.from_numpy(self.meas_img_np).to(self.device_cnn)

        self.step_number = 0
        self.actions_per_ep = 0
        self.mean_reward = 0

        self.private_correct = 0
        self.public_correct = 0
        self.total = 0
        self.total_ep = 0

        return self._get_obs()

    def render(self, path):
        pickle.dump((self.meas_img, self.labels), open(path, 'ab'))
        print('image saved!')
