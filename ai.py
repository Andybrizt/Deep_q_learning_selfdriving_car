import numpy as np
import random
import os
import torch
import torch.nn as nn #神經網路
import torch.nn.functional as F #huber函數
import torch.optim as optim #梯度下降
import torch.autograd as autograd
from torch.autograd import Variable


class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size #輸入(5 input defined below)
        self.nb_action = nb_action #輸出 (3 output defined below)
        
        self.hidden_1_size = 8
        self.hidden_2_size = 10 # second layer added
        self.hidden_3_size = 6
        self.fc1 = nn.Linear(input_size, self.hidden_1_size)
        self.fc2 = nn.Linear(self.hidden_1_size, self.hidden_2_size)
        self.fc3 = nn.Linear(self.hidden_2_size, self.hidden_3_size)
        self.fc4 = nn.Linear(self.hidden_3_size, nb_action)
    
        #self.fc1 = nn.Linear(input_size, 30)#輸入層和隱藏層之間的full-connected全連接神經元。30為可自定義神經元個數。經實驗，自定義30個神經元結果不錯
        #self.fc2 = nn.Linear(30, nb_action)#隱藏層與輸出層的全聯接神經，所以第一個變數是30，第二個是輸出層名稱
    
    def forward(self, state):#激活函數，用於正向傳播state是神經網路的輸入。註：這裡不需要定義輸出3q值（前、左、右），此函數會直接返回q值
        x = F.relu(self.fc1(state))#x代表隱藏神經元。使用relu這個激活函數來激活第一隱藏層神經元
        y = F.relu(self.fc2(x))
        z = F.relu(self.fc3(y))
        q_values = self.fc4(z)
        #q_values = self.fc2(x)
        return q_values


class ReplayMemory(object):
    
    def __init__(self, capacity): #capacity是記憶的容量數目，將指定為100000（記憶至前100000步）
        self.capacity = capacity
        self.memory = []#儲存記憶的list
    
    def push(self, event):#增加新的狀態（event）到memory
        self.memory.append(event)#增加到memory list
        if len(self.memory) > self.capacity:#若增加記憶點後已經超過memory容量(也就是capacity)，將刪除第一個記憶點（最舊的記憶點）
            del self.memory[0]
    
    def sample(self, batch_size):#batch_size:抽取隨機樣本的數量
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):#input_size, nb_action即上面class Network的參數；gamma是q-learning公式中的延遲係數(r) 
        self.gamma = gamma
        self.reward_window = []#近100次的獎勵平均變化的滑動窗
        self.model = Network(input_size, nb_action)#使用上方定義的class Ｎetwork()，創造一個神經網路模型
        
        self.memory = ReplayMemory(100000)#使用上方定義的class ReplayMemory()，創造一個記憶回放的儲存體
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.003)
        #從pytorch.optim選擇Adam創造一個優化器。
        #self.model.parameters()可獲取model的參數，為了把優化器optimizer和model接起來
        #lr是learning rate(學習速度)，數值太大的話無法好好學習，必須給ＡＩ充裕時間來學習，例如掉進沙坑時
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*50) # T=100
        action = probs.multinomial(1)
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph = True)
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")