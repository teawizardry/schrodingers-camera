from typing import Deque
import torch
import random, numpy as np
from neural import AgentNet
from collections import deque
import pickle


class Agent:
    def __init__(self, state_dim, action_dim, save_dir, config, checkpoint=None, memory_checkpoint=None, adam=False, noise_machine=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=100000)
        self.batch_size = config.batch_size

        self.exploration_rate = config.exploration_rate
        self.exploration_rate_decay = config.exploration_rate_decay
        self.exploration_rate_min = config.exploration_rate_min
        self.gamma = config.gamma

        self.curr_step = 0
        self.burnin = config.burnin  # min. experiences before training
        self.learn_every = config.learn_every  # no. of experiences between updates to Q_online 2 5
        self.sync_every = config.sync_every   # no. of experiences between Q_target & Q_online sync

        self.save_every = config.save_every   # no. of experiences between saving Mario Net
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        self.noise_machine = noise_machine

        # DNN to predict the most optimal action - we implement this in the Learn section
        if config.model_selection is False:
            self.net = AgentNet(self.state_dim, self.action_dim).float()
        else: 
            pass
            # self.net = AgentNet_2(self.state_dim, self.action_dim).float()
        # if self.use_cuda:
        self.net = self.net.to(device='cuda:0')
        if checkpoint:
            self.load(checkpoint)
        
        if memory_checkpoint:
            self.load_memory(memory_checkpoint)

        if adam == True:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        else:
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=config.learning_rate)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        self.loss_fn = torch.nn.SmoothL1Loss()


    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.
        Inputs:
        state(image): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """
        # INITIALIZE 
        if self.curr_step == 0 or self.curr_step == self.burnin:
            action_idx = 27 # [27, -1, -1, -1] # initialize completely redacted
            self.curr_step += 1
            return action_idx

        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            # action_idx = np.ones((4,), dtype=int)*(-1)
            # for sz in range(np.random.randint(4)+1):
            #     action_idx[sz] = np.random.randint(self.action_dim)
            # action_idx = np.random.randint(self.action_dim, size=(np.random.randint(4)+1))
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            # state = torch.cuda.FloatTensor(state) # if self.use_cuda else torch.FloatTensor(np.asarray(state))
            # state = state.unsqueeze(0)
            print("state size: ", state.size())
            action_values = self.net(state, model='online')
            # action_values = model_out.detach().cpu().numpy().squeeze()
            # print("action values size: ", action_values.size())
            # try:
            print("action values: ", action_values.size())
            action_idx = torch.argmax(action_values, axis=0).item()
            
        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)
        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        
        reward = torch.LongTensor([reward]).to(device='cuda:0')
        action = torch.LongTensor([action]).to(device='cuda:0') # if self.use_cuda else torch.LongTensor([action])
        done = torch.BoolTensor([done]).to(device='cuda:0') # if self.use_cuda else torch.BoolTensor([done])

        self.memory.append( (state, next_state, action, reward, done,) )


    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch)) # zip(*batch) 
        # print(state.dtype, state.shape)
        # print(next_state.dtype, next_state.shape)
        # print(action.dtype, action.shape)
        # print(reward.dtype, reward.shape)
        # print(done.dtype, done.shape)
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


    def td_estimate(self, state, action):
        # print("state size: ", state.size())
        # state = state.squeeze()
        current_Q =  torch.squeeze(self.net(state, model='online'), axis=1)
        print("current q: ", current_Q.size())
        current_Q = current_Q[np.arange(0, self.batch_size), action] # Q_online(s,a)
        # print("td_estimate: ", current_Q)
        return current_Q


    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        # next_state = next_state.squeeze()
        # print("next state size: ", next_state.size())
        next_state_Q = self.net(next_state, model='online')
        print("next state q size: ", next_state_Q.size())
        # try:
        best_action = torch.argmax(next_state_Q, axis=1)
        print("best action size: ", best_action.size())
        next_Q = torch.squeeze(self.net(next_state, model='target'), axis=1)
        print("next q size: ", next_Q.size())
        next_Q = next_Q[np.arange(0, self.batch_size), best_action]
        # except:
            # next_state_Q = torch.unsqueeze(next_state_Q, 0)
            # best_action = torch.argmax(next_state_Q, axis=1)
            # print("best action size: ", best_action.size())
            # next_Q = next_Q[0, best_action]
        output = (reward + (1 - done.float().to(device='cuda:0')) * self.gamma * next_Q).float()
        # print("td_target: ", output)
        return output


    def update_Q_online(self, td_estimate, td_target) :
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())


    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None, None

        if self.curr_step % self.learn_every != 0:
            return None, None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), td_tgt.mean().item(), loss)


    def save(self):
        save_path = self.save_dir / f"agent_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        save_path = self.save_dir / f"agent_cache_{int(self.curr_step // self.save_every)}.pkl"
        pickle.dump(self.memory, open(f"{save_path}", "wb"))
        print(f"AgentNet saved to {save_path} at step {self.curr_step}")


    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate


    def load_memory(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        self.memory = deque(pickle.load(open(load_path, "rb")), maxlen=100000)
        print(f"Memory loaded from {load_path}")