import pickle
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nes_py.wrappers import JoypadSpace
from pathlib import Path
import sys
sys.path.append(str(Path("../gym-super-mario-bros").resolve()))
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

from wrappers import *
import MODEL

q_name = "mario_q.pd"
q_target_name = "mario_q_target.pd"


def arrange(s: LazyFrames):
    if not type(s) == "numpy.ndarray":
        s = np.array(s)
    assert len(s.shape) == 3
    ret = np.transpose(s, (2, 0, 1))
    return np.expand_dims(ret, 0)


class replay_memory(object):
    def __init__(self, N):
        self.memory = deque(maxlen=N)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, n):
        return random.sample(self.memory, n)

    def __len__(self):
        return len(self.memory)


def train(q, q_target, memory, batch_size, gamma, optimizer, device):
    s, r, a, s_prime, done = list(map(list, zip(*memory.sample(batch_size))))
    s = np.array(s).squeeze()
    s_prime = np.array(s_prime).squeeze()
    a_max = q(s_prime).max(1)[1].unsqueeze(-1)
    r = torch.FloatTensor(r).unsqueeze(-1).to(device)
    done = torch.FloatTensor(done).unsqueeze(-1).to(device)
    with torch.no_grad():
        y = r + gamma * q_target(s_prime).gather(1, a_max) * done
    a = torch.tensor(a).unsqueeze(-1).to(device)
    q_value = torch.gather(q(s), dim=1, index=a.view(-1, 1).long())

    loss = F.smooth_l1_loss(q_value, y).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def copy_weights(q, q_target):
    q_dict = q.state_dict()
    q_target.load_state_dict(q_dict)


def main(env: FrameStack, q, q_target, optimizer, device):
    t = 0
    gamma = 0.99
    batch_size = 256

    N = 50000
    memory = replay_memory(N)
    update_interval = 50
    print_interval = 1
    max_noop = 10

    score_lst = []
    total_score = 0.0
    loss = 0.0

    for k in range(1000000):
        (s, info) = env.reset()
        s = arrange(s)
        done = False

        while not done:
            if device == "cpu":
                a = q(s).detach()
            else:
                a = q(s).cpu().detach()
            a = F.softmax(a, -1).numpy()[0]
            a = np.random.choice(list(i for i in range(len(a))), p=a)
            s_prime, r, done, _, info = env.step(a)
            s_prime = arrange(s_prime)
            total_score += r
            r = np.sign(r) * (np.sqrt(abs(r) + 1) - 1) + 0.001 * r
            memory.push((s, float(r), int(a), s_prime, int(1 - done)))
            s = s_prime
            stage = env.unwrapped._stage

            if (len(memory) > 2000):
                loss += train(q, q_target, memory, batch_size, gamma, optimizer, device)
                t += 1
                
            if (t % update_interval == 0 and t != 0):
                copy_weights(q, q_target)
                torch.save(q.state_dict(), q_name)
                torch.save(q_target.state_dict(), q_target_name)

        if k % print_interval == 0:
            print(
                "Epoch : %d | score : %f | loss : %.2f | stage : %d | train_count : %d | mem : %d"
                % (
                    k,
                    total_score / print_interval,
                    loss / print_interval,
                    stage,
                    t,
                    len(memory),
                )
            )
            score_lst.append(total_score / print_interval)
            total_score = 0
            loss = 0.0
            # pickle.dump(str(score_lst), open("score.p", "wb"))


if __name__ == "__main__":
    n_frame = 4
    fps = 60
    env = gym_super_mario_bros.make("SuperMarioBros-v0", render_mode="human")
    env.metadata["render_fps"] = fps
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = wrap_mario(env)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = MODEL.duel_dqn(n_frame, env.action_space.n, device).to(device)
    q_target = MODEL.duel_dqn(n_frame, env.action_space.n, device).to(device)

    continue_train = input("continue train")
    if (len(continue_train) > 0 and continue_train.lower()[0] == "y"):
        q.load_state_dict(torch.load(q_name))
        q.load_state_dict(torch.load(q_target_name))
        
    optimizer = optim.AdamW(q.parameters(), lr=0.0001)
    print(device)
    main(env, q, q_target, optimizer, device)