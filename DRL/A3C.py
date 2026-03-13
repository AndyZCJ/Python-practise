import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

LR_ACTOR = 0.0001     # 策略网络的学习率
LR_CRITIC = 0.001   # 价值网络的学习率
GAMMA = 0.99         # 奖励的折扣因子
EPSILON = 0.9       # ϵ-greedy 策略的概率
TARGET_REPLACE_ITER = 100                 # 目标网络更新的频率
env = gym.make('CartPole-v0')             # 加载游戏环境
N_ACTIONS = env.action_space.n            # 动作数
N_SPACES = env.observation_space.shape[0] # 状态数量
env = env.unwrapped
env.seed(0)
torch.manual_seed(0)
UPDATE_GLOBAL_ITER = 5
def init_weights(m) :
    if isinstance(m, nn.Linear) :
        nn.init.normal_(m.weight, mean = 0, std = 0.1)


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )

class AC_Network(nn.Module):#AC网络
    def __init__(self):
        super(AC_Network, self).__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(N_SPACES, 128),
            nn.ReLU(),
            nn.Linear(128, N_ACTIONS)
        )

        self.critic_net = nn.Sequential(
            nn.Linear(N_SPACES, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, s):
        pi = self.actor_net(s)
        pi = F.softmax(pi, dim=-1)
        v = self.critic_net(s)
        return pi, v

    def loss_func(self, s, a, r, s_):#损失函数
        s = torch.FloatTensor(s)
        s_ = torch.FloatTensor(s_)
        r = torch.FloatTensor(r)
        pi, v = self.forward(s)
        pi_, v_ = self.forward(s_)
        v_.detach()
        target = r + GAMMA * v_
        advantage = (v - target).detach()
        log_q_actor = torch.log(pi)
        actor_loss = []
        for i in range(len(log_q_actor)):
            log_q_actor[i]= log_q_actor[i][a[i]]*advantage[i]
        criterion = nn.MSELoss()
        loss_critic = criterion(v, target)
        total_loss = (loss_critic+log_q_actor).mean()
        return total_loss

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), dim=0)
        if np.random.uniform()<EPSILON:
            action_value, _ = self.forward(s)
            action = torch.max(action_value, dim=1)[1].item()
        else:
            action = np.random.randint(0,N_ACTIONS)

        return action



class Worker(mp.Process):
    def __init__(self, opt,gnet, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.global_ep, self.global_ep_r, self.res_queue= global_ep, global_ep_r, res_queue
        self.gnet, self.opt= gnet, opt
        self.lnet = AC_Network().apply(init_weights)
        self.env = gym.make('CartPole-v0').unwrapped



    def run(self):
        total_step = 1
        while self.global_ep.value<10000:
            s = self.env.reset()
            ep_r = 0
            while True:
                #if self.name == 'w00':
                    #self.env.render()
                buffer_s, buffer_a, buffer_r, buffer_s_ = [], [], [],[]
                a = self.lnet.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                if done:
                    r  = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)
                buffer_s_.append(s_)
                if total_step%UPDATE_GLOBAL_ITER == 0 or done:#更新全局网络
                    loss = self.lnet.loss_func(buffer_s, buffer_a,buffer_r, buffer_s_)
                    self.opt.zero_grad()
                    loss.backward()
                    for lp, gp in zip(self.lnet.parameters(), self.gnet.parameters()):
                        gp._grad = lp.grad
                    self.opt.step()
                    self.lnet.load_state_dict(self.gnet.state_dict())
                    buffer_s, buffer_a, buffer_r, buffer_s_ = [], [], [], []
                    if done:
                        record(self.global_ep, self.global_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
            self.res_queue.put(None)


class SharedAdam(torch.optim.Adam):#优化器
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

if __name__ == "__main__":
    gnet = AC_Network().apply(init_weights)
    gnet.share_memory()
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # 并行训练
    workers = [Worker(opt,gnet,  global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()


