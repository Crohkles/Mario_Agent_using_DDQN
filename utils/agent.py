import numpy as np
import torch 
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from datetime import datetime

from utils.net import MarioNet
class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        #输入状态的维度 (即observation的shape)
        self.state_dim = state_dim
        #动作空间维度
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Mario's DNN to predict the most optimal action
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        # 以exploration_rate的概率进行explore（探索随机动作）
        # 以1-exploration_rate的概率进行exploit(利用)
        self.exploration_rate = 1
        # 探索率衰减因子
        self.exploration_rate_decay = 0.99999975
        # 探索率下限
        self.exploration_rate_min = 0.1
        # 于环境交互的步数计数器
        self.curr_step = 0
        # 每次保存模型的步数间隔
        self.save_every = 5e3  

        self.memory = TensorDictReplayBuffer(storage = LazyMemmapStorage(100000,
                                                                         device = torch.device("cpu")))
        self.batch_size = 32    

        self.gamma=0.9
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        # 预热期，在replay buffer中积累足够经验前，先采用epsilon-greedy策略填充经验池
        # 之后再正式开始学习
        self.burnin = 1e4
        # Q_online学习的经验间隔
        self.learn_every = 3
        # Q_target与Q_online同步的经验间隔
        self.sync_every = 1e4
    
    # 给定状态，将通过explore或exploit返回对应action的index
    def act(self, state):
        # EXPLORE
        # 随机探索
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        # 利用最优action
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # 探索率衰减
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # 步数记录
        self.curr_step += 1
        return action_idx
    
    # 每次执行动作后将经验存储到memory中
    def cache(self, state, next_state, action, reward, done):
        def first_if_tuple(x):
            return x[0] if isinstance(x,tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])
        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))

    # 从记忆中随机抽取一批经验
    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
        
    def td_estimate(self, state, action):
        current_Q = self.net(state,model="online")[
            np.arange(0,self.batch_size),action
        ]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0,self.batch_size),best_action
        ]
        return (reward + (1 - done.float()) * self.gamma *next_Q).float()
    
    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = (
        self.save_dir / f"mario_net_{timestamp}.chkpt")
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def load(self, load_path):
        """加载已保存的模型"""
        if not load_path.exists():
            raise FileNotFoundError(f"No file at {load_path}")
    
        ckp = torch.load(load_path, map_location=self.device)
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate

    def learn(self):
        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()
        
        # 从经验中随机采样
        state, next_state, action, reward, done = self.recall()

        td_est = self.td_estimate(state, action)

        td_tgt = self.td_target(reward, next_state, done)
        
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)