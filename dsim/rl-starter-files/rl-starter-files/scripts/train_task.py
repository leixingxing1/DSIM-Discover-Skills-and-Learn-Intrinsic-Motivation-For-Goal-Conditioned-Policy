import argparse
import time
import torch
import os
import json
import shutil
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from utils.replay_memory import ReplayMemory
import torch.distributed as dist
from random import choice

import utils
import torch.nn as nn
from utils.policy import StochasticPolicy
from utils.value_function import Value
from model import VQVAEDiscriminator
from base.modules.generic import CategoricalWithoutReplacement
from gcsl.algo import networks
from gym.spaces import Box, Discrete

class DistanceStochasticAgent(nn.Module):
    def __init__(self, env, skill_n ,curr_skill,noise=None, epsilon=None, **module_kwargs):
        super().__init__(**kwargs)
        self.batch_keys += ['goal']  # 'goal' is only used for visualization purposes
        self.skill_n = int(skill_n)

        self.skill_dist = CategoricalWithoutReplacement(self.skill_n)

        self.curr_skill = curr_skill

        self.batch_keys = [
            'state', 'next_state', 'skill',
            'action', 'n_ent', 'log_prob', 'action_logit',
            'reward', 'terminal', 'complete',
        ]
        self.no_squeeze_list = []

        self.noise = max(0.0, float(noise)) if noise is not None else noise
        self.epsilon = max(0.0, min(1.0, float(epsilon))) if epsilon is not None else epsilon

        self.env = env
        self._make_modules(**module_kwargs)

        self.episode = []

    def _make_modules(self, policy, skill_embedding, vae):
        self.policy = policy
        self.skill_embedding = skill_embedding
        self.vae = vae

    def reset(self, skill=None, *args, **kwargs):
        self.reset_skill(skill)
        kwargs['goal'] = self.vae.get_centroids(dict(skill=self.curr_skill.view([]))).detach().numpy()
        self.env.reset(*args, **kwargs)
        self.episode = []

    def preprocess_skill(self, curr_skill):
        assert curr_skill is not None
        return self.skill_embedding(curr_skill).detach()

    def sample_skill(self):
        return self.skill_dist.sample(sample_shape=(1,)).view([])

    def reset_skill(self, skill=None):
        self.curr_skill = self.sample_skill()
        if skill is not None:
            self.curr_skill = self.curr_skill * 0 + skill

    def reset(self, skill=None, *args, **kwargs):
        self.env.reset(*args, **kwargs)
        self.episode = []
        self.reset_skill(skill)

    def collect_transitions(self, num_transitions, reset_dict={}, do_eval=False):
        self.episode = []
        for _ in range(num_transitions):
            if self.env.is_done:
                self.env.reset(**reset_dict)
                self.reset_skill()
            self.step(do_eval)

    def step(self, do_eval=False):
        s = self.env.state
        z = self.preprocess_skill(self.curr_skill)
        a, logit, log_prob, n_ent = self.policy(s.view(1, -1), z.view(1, -1), greedy=do_eval)
        a = a.view(-1)
        logit = logit.view(-1)
        log_prob = log_prob.sum()

        self.env.step(a)
        complete = self.env.is_complete if hasattr(self.env, 'is_complete') else self.env.is_success
        complete = float(complete) * torch.ones(1)
        terminal = float(self.env.is_done) * torch.ones(1)
        s_next = self.env.state
        r = torch.zeros(1)
        env_rew = self.env.reward * torch.ones(1)
        discriminator_rew = torch.zeros(1)

        self.episode.append({
            'state': s,
            'skill': self.curr_skill.detach(),
            'action': a,
            'action_logit': logit,
            'log_prob': log_prob.view([]),
            'n_ent': n_ent.view([]),
            'next_state': s_next,
            'terminal': terminal.view([]),
            'complete': complete.view([]),
            'env_reward': env_rew.view([]),
            'im_reward': discriminator_rew.view([]),  # to be filled during relabeling
            'reward': r.view([]),  # to be filled during relabeling
        })

    def play_episode(self, reset_dict={}, do_eval=False):
        self.reset(**reset_dict)
        while not self.env.is_done:
            self.step(do_eval)

    def collect_transitions(self, num_transitions, reset_dict={}, do_eval=False):
        self.episode = []
        for _ in range(num_transitions):
            if self.env.is_done:
                self.env.reset(**reset_dict)
            self.step(do_eval)

    @property
    def rollout(self):
        states = torch.stack([e['state'] for e in self.episode] + [self.episode[-1]['next_state']]).data.numpy()
        actions = torch.stack([e['action'] for e in self.episode]).data.numpy()
        return states,actions

@staticmethod
def condense_loss(loss_):
    if isinstance(loss_, torch.Tensor):
        return loss_.mean()
    if isinstance(loss_, (list, tuple)):
        net_loss = 0.
        for sub_loss in loss_:
            net_loss += sub_loss.mean()
        return net_loss
    else:
        raise TypeError

# Set up saving
def checkpoint(agent,curr_epoch,dur):
    agent.save_checkpoint(model_path)
    torch.save(optim, optim_path)
    checkpoint_path = os.path.join(exp_dir, '{:04d}_model.pth.tar'.format(curr_epoch))
    agent.save_checkpoint(checkpoint_path)

    n_episodes_played = int(agent.train_steps.data.item())

    optimization_steps = 0
    for pg in optim.param_groups:
        for p in pg['params']:
            optimization_steps = max(optimization_steps, int(optim.state[p]['step'][0]))

    print(
        '\nCHECKPOINT REACHED  --  Epochs = {}/{}  --  N Episodes = {}  --  N Optimizations = {}\n'.format(
            curr_epoch, int(dur), n_episodes_played, optimization_steps
        ),
        flush=True
    )

def rollout_wrapper(agent):
    st = time.time()
    agent.eval()
    agent.play_episode()

    agent.train()
    loss = condense_loss(agent())
    dur = time.time() - st

    return loss,dur

def eval_wrapper(agent,config):
    stats = []
    episodes = {}
    for evi in range(config.get('eval_iters', 10)):
        agent.play_episode(do_eval=bool(config.get('greedy_eval', True)))

        ep_stats = [float(x) for x in agent.episode_summary()]
        stats.append(ep_stats)

        dump_ep = []
        for t in agent.curr_ep:
            dump_t = {k: np.array(v.detach()).tolist() for k, v in t.items()}
            dump_ep.append(dump_t)
        episodes[evi] = dump_ep

    return stats, episodes

def log_eval_results(stats, episodes,curr_epoch,exp_dir):
        # Save this crop of episodes
        dstr = 'eval_{:04d}'.format(curr_epoch)
        # c_path = os.path.join(BASE_DIR, self.tag, dstr + '.json')
        c_path = os.path.join(exp_dir, dstr + '.json')
        with open(c_path, 'wt') as f:
            json.dump(episodes, f)

        # Save the stats dictionary
        eval_stats[str(curr_epoch)] = np.array(stats)
        np.savez(os.path.join(exp_dir, 'stats_{:02d}.npz'.format(0)), **eval_stats)

        # Print out some summary stats
        mean_stats = np.array(stats).mean(axis=0)
        n = len(mean_stats)
        f_str = 'E{:04d} Eval Stats, rank {:02d}:  ' + (', '.join(['{:+07.3f}'] * n))
        print(f_str.format(curr_epoch, 0, *mean_stats), flush=True)

        # Produce a summary
        total_stats = torch.from_numpy(np.array(stats))
        dist.all_reduce(total_stats)

        
        net_mean = total_stats / dist.get_world_size()
        net_mean = net_mean.mean(dim=0).data.numpy()

        f_str = '\nE{:04d} Eval Stats, AVERAGE:  ' + (', '.join(['{:+07.3f}'] * n))
        print(f_str.format(curr_epoch, *net_mean), flush=True)
    
def open_experiment(dataset):
    parser = argparse.ArgumentParser("restore VQ-VAE.")
    parser.add_argument('--config-path', type=str, default='config/vqvae_representation.json',help='Path to experiment config file (expecting a json)')
    parser.add_argument('--log-dir', type=str, default='logs',help='Parent directory that holds experiment log directories')
    args = parser.parse_args()

    vqvae_config_path = args.config_path
    assert os.path.isfile(vqvae_config_path)
    vqvae_config = json.load(open(vqvae_config_path))

    exp_name = vqvae_config_path.split('/')[-1][:-5]
    exp_dir = os.path.join(args.log_dir, exp_name)

    print('Experiment directory is: {}'.format(exp_dir), flush=True)

    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
        shutil.copyfile(config_path, os.path.join(exp_dir, 'config.json'))

    dataset = torch.from_numpy(dataset).float()
    state_size = dataset.ndim

    return state_size,vqvae_config, args,exp_dir

def sample_skill(skill_n):
        skill_dist = CategoricalWithoutReplacement(skill_n)
        return skill_dist.sample(sample_shape=(1,)).view([])

def play_episode(agent, skill, do_eval, reset_dict={}):
    agent.reset(**reset_dict)
    agent.curr_skill = agent.curr_skill * 0 + skill
    while not agent.env.is_done:
        agent.step(do_eval)

def KLdivergence(p,q):
    # print(p)
    # print(q)
    print([b * np.log(a/b) for a,b in zip(p,q)])   
    KL=(np.sum([b * np.log(a/b) for a,b in zip(p,q)]))*(-1)
    return KL

def default_markov_policy(env, env_params):
    assert isinstance(env.action_space, Discrete)
    if env.action_space.n > 100: # Too large to maintain single action for each
        policy_class = networks.IndependentDiscretizedStochasticGoalPolicy
    else:
        policy_class = networks.DiscreteStochasticGoalPolicy
    return policy_class(
                env,
                state_embedding=None,
                goal_embedding=None,
                layers=[400, 300], #[400, 300], # TD3-size
                max_horizon=None, # Do not pass in horizon.
                # max_horizon=get_horizon(env_params), # Use this line if you want to include horizon into the policy
                freeze_embeddings=True,
                add_extra_conditioning=False,
            )


# Parse arguments
#Skill learning and agent rollout args
parser = argparse.ArgumentParser(description='PyTorch  Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--num_layers', type=int, default=4, metavar='N',
                    help='MLP number layers (default: 4)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=500001, metavar='N',
                    help='maximum number of steps (default: 500000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='hidden size (default: 128)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--normalize_inputs', type=bool, default=False,
                    help='normalize and input operator')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: True)')
parser.add_argument('--config-path', type=str,
                    help='Path to experiment config file (expecting a json)')
parser.add_argument('--log-dir', type=str,
                    help='Parent directory that holds experiment log directories')
#Goal-conditioned policy optimazition args
parser.add_argument("--distance_threshold", default=0.5, type=float)
parser.add_argument("--start_timesteps",    default=1e4, type=int) 
parser.add_argument("--eval_freq",          default=1e3, type=int)
parser.add_argument("--max_timesteps",      default=10, type=int)
parser.add_argument("--max_episode_length", default=100, type=int)
parser.add_argument("--batch_size",         default=128, type=int)
parser.add_argument("--replay_buffer_size", default=1e5, type=int)
parser.add_argument("--n_eval",             default=5, type=int)
parser.add_argument("--device",             default="cuda")
parser.add_argument("--seed",               default=42, type=int)
parser.add_argument("--exp_name",           default="GCRL_ant")
parser.add_argument("--alpha",              default=0.1, type=float)
parser.add_argument("--Lambda",             default=0.1, type=float)
parser.add_argument("--h_lr",               default=1e-4, type=float)
parser.add_argument("--q_lr",               default=1e-3, type=float)
parser.add_argument("--pi_lr",              default=1e-3, type=float)
parser.add_argument('--log_loss', dest='log_loss', action='store_true')
parser.add_argument('--no-log_loss', dest='log_loss', action='store_false')
parser.set_defaults(log_loss=True)
args = parser.parse_args()
print(args)

config_path = args.config_path
assert os.path.isfile(config_path)
config = json.load(open(config_path))

exp_name = config_path.split('/')[-1][:-5]
exp_dir = os.path.join(args.log_dir, exp_name)

print('Experiment directory is: {}'.format(exp_dir), flush=True)
model_path = os.path.join(exp_dir, 'model.pth.tar')
optim_path = os.path.join(exp_dir, 'optim.pth.tar')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load environments
env=utils.make_env(args.env)

# Memory
memory = ReplayMemory(args.replay_size)

# Load observations
obs_space= env.observation_space

#Interpret the arguments. Create the model, optimizer and dataset. Fetch the config file.
state_size,vqvae_config, args, save_dir = open_experiment(obs_space)
vae_model_path = os.path.join(save_dir, "model.pth.tar")
vae = VQVAEDiscriminator(state_size=state_size, **config['vae_args'])
vae.load_checkpoint(vae_model_path)

skill_embed=vae.vq
skill_embedding = vae.vq.embedding
vq_vae=vae 

curr_skill = sample_skill(config['skill_n'])

# preprocess_skill(self, curr_skill):
assert curr_skill is not None
sample_skillembedding=skill_embedding(curr_skill).detach()

kwargs = dict(env=env, hidden_size=args.hidden_size, num_layers=args.num_layers,
                      goal_size=vae.code_size, normalize_inputs=args.normalize_inputs)
eval_stats = {}

policy = StochasticPolicy(**kwargs)
v_module = Value(use_antigoal=False, **kwargs)

agent = DistanceStochasticAgent(env=env, policy=policy, skill_n=vae.codebook_size,curr_skill=curr_skill,
                                       skill_embedding=vae.vq.embedding, vae=vae)

optim = optim.Optimizer(agent.parameters(), lr=config['learning_rate'])

#Skill learning and agent rollout
for _ in range(config['cycles_per_epoch'] * config["updates_per_cycle"]):
        optim.zero_grad()

        loss = 0
        cycle_ep_counter = torch.zeros(1)
        for _ in range(config["rollouts_per_update"]):
            this_loss,dur = rollout_wrapper(agent)
            loss += this_loss / config["rollouts_per_update"]

        loss.backward()
        for p in agent.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad.data)
                p.grad.data /= dist.get_world_size()
        optim.step()

        for v in agent.state_dict().values():
            dist.broadcast(v.data, src=0)

        dist.all_reduce(cycle_ep_counter)
        agent.train_steps += cycle_ep_counter.item()

stats, episodes = eval_wrapper(agent,config)
log_eval_results(stats, episodes,_,exp_dir)

checkpoint(agent,_,dur)

dist.barrier()

if agent.skill_n <= 10:
    cmap = plt.get_cmap('tab10')
elif 10 < agent.skill_n <= 20:
    cmap = plt.get_cmap('tab20')
else:
    cmap = plt.get_cmap('viridis', agent.skill_n)

reset_dict = agent.env.sibling_reset  # fix s_0 across trajectories and skills
trajectory_list ={}
action_list = {}

for skill_idx in range(agent.skill_n):
    # Collect rollout
    play_episode(agent, skill_idx, do_eval=False, reset_dict=reset_dict)

    #*agent.rollout is the numpy array of agent rollout
    #Goal-conditioned policy optimization:actor-critic goal-conditioned policy and minimize D_kl in q′φ(gτ )||qφ(τ )) * w(gτ , τ ) where w(gτ , τ ) horizon of gτ in τ

    #Sample and relabel goal from agent rollout trajectory
    trajectory,actions = agent.rollout
    trajectory_list.append(trajectory)
    action_list.append(actions)

    #Wight is represented by horizon
    start_state_wight = np.random.choice(len(trajectory))
    start_state = trajectory[start_state_wight]
    goal_t_wight = np.random.choice(start_state_wight,len(trajectory))

    goal_state = trajectory[goal_t_wight]
    goal_state_wight = goal_t_wight - start_state_wight
    choose_wight = goal_state_wight
    choose_skill_index = skill_index
    #Go through the list and find the skill with the lowest weight
    for skill_index in range(len(trajectory_list)):
        for t in range(len(trajectory_list[skill_idx])):
            if start_state == trajectory_list[skill_idx][t]:
               start_t = t
            if goal_state == trajectory_list[skill_idx][t]:
               goal_t = t
        current_wight = goal_t - start_t
        if current_wight < choose_wight:
           choose_wight = current_wight
           choose_skill_index = skill_idx
 
    #minimize KL between q_phi_goal_state and q_phi_tau
    model = VQVAEDiscriminator(state_size=goal_state.ndim, **config['vae_args'])
    goal_encoder = model.encoder
    q_phi_goal_optimizer = torch.optim.Adam(goal_encoder.parameters(), lr=1e-3)
    
    q_phi_goal_loss= KLdivergence(goal_encoder.parameters(),vq_vae.encoder.parameters()).mean() 
    
    q_phi_goal_optimizer.zero_grad()
    q_phi_goal_loss.backward()
    q_phi_goal_optimizer.step()

    for param,target_param in zip(goal_encoder.parameters(),vq_vae.encoder.parameters()): 
        param.data.copy_(target_param.data)
    
    q_phi_goal = model.encoder(torch.tensor(goal_state).float())
    #Push skills to replay buffer and choose low wight skill to optimize gcrl policy
    
    # Initialize environment
    optimization_trajectory = trajectory_list(choose_skill_index)
    optimization_trajectory_actions = action_list(choose_skill_index)
    state = start_state
    goal = q_phi_goal

    #Imitate optimal trajectory with goal-conditioned behavioral cloning
    policy = default_markov_policy(env)
    algo = gcsl.GCSL(
        env,
        policy,
        start_state,
        goal,
        optimization_trajectory,
        optimization_trajectory_actions
    )

    exp_prefix = 'example/%s/gcsl/' % (args.env,)

    with log_utils.setup_logger(exp_prefix=exp_prefix, log_base_dir=output_dir):
        algo.train()
    

