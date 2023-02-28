import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

""" Actor """

class GaussianPolicy(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
		super(GaussianPolicy, self).__init__()
		fc = [nn.Linear(2*state_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim_in, hidden_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc += [nn.Linear(hidden_dim_in, hidden_dim_out), nn.ReLU()]
		self.fc = nn.Sequential(*fc)

		self.mean_linear = nn.Linear(hidden_dims[-1], action_dim)
		self.logstd_linear = nn.Linear(hidden_dims[-1], action_dim)

		self.LOG_SIG_MIN, self.LOG_SIG_MAX = -20, 2

	def forward(self, state, goal):
		x = self.fc(torch.cat([state, goal], -1))
		mean = self.mean_linear(x)
		log_std = self.logstd_linear(x)
		std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX).exp()
		normal = torch.distributions.Normal(mean, std)
		return normal

	def sample(self, state, goal):
		normal = self.forward(state, goal)
		x_t = normal.rsample()
		action = torch.tanh(x_t)
		log_prob = normal.log_prob(x_t)
		log_prob -= torch.log((1 - action.pow(2)) + 1e-6)
		log_prob = log_prob.sum(-1, keepdim=True)
		mean = torch.tanh(normal.mean)
		return action, log_prob, mean

""" Critic """
class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
		super(Critic, self).__init__()
		fc = [nn.Linear(2*state_dim + action_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim1, hidden_dim2 in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc += [nn.Linear(hidden_dim1, hidden_dim2), nn.ReLU()]
		fc += [nn.Linear(hidden_dims[-1], 1)]
		self.fc = nn.Sequential(*fc)

	def forward(self, state, action, goal):
		x = torch.cat([state, action, goal], -1)
		return self.fc(x)

class EnsembleCritic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], n_Q=2):
		super(EnsembleCritic, self).__init__()
		ensemble_Q = [Critic(state_dim=state_dim, action_dim=action_dim, hidden_dims=hidden_dims) for _ in range(n_Q)]			
		self.ensemble_Q = nn.ModuleList(ensemble_Q)
		self.n_Q = n_Q

	def forward(self, state, action, goal):
		Q = [self.ensemble_Q[i](state, action, goal) for i in range(self.n_Q)]
		Q = torch.cat(Q, dim=-1)
		return Q


class GCRL(object):
	def __init__(self, state_dim, action_dim, alpha=0.1, Lambda=0.1, image_env=False, n_ensemble=10, gamma=0.99, tau=0.005, target_update_interval=1, h_lr=1e-4, q_lr=1e-3, pi_lr=1e-4, enc_lr=1e-4, epsilon=1e-16, logger=None, device=torch.device("cuda")):		
		# Actor
		self.actor = GaussianPolicy(state_dim, action_dim).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=pi_lr)
		self.actor_target = GaussianPolicy(state_dim, action_dim).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())

		# Critic
		self.critic 		= EnsembleCritic(state_dim, action_dim).to(device)
		self.critic_target 	= EnsembleCritic(state_dim, action_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=q_lr)

		# Actor-Critic Hyperparameters
		self.tau = tau
		self.target_update_interval = target_update_interval
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon

		# High-level policy hyperparameters
		self.Lambda = Lambda
		self.n_ensemble = n_ensemble

		# Utils
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.device = device
		self.logger = logger
		self.total_it = 0

	def save(self, folder, save_optims=False):
		torch.save(self.actor.state_dict(),		 folder + "actor.pth")
		torch.save(self.critic.state_dict(),		folder + "critic.pth")
		if save_optims:
			torch.save(self.actor_optimizer.state_dict(), 	folder + "actor_opti.pth")
			torch.save(self.critic_optimizer.state_dict(), 	folder + "critic_opti.pth")

	def load(self, folder):
		self.actor.load_state_dict(torch.load(folder+"actor.pth", map_location=self.device))
		self.critic.load_state_dict(torch.load(folder+"critic.pth", map_location=self.device))

	def select_action(self, state, goal):
		with torch.no_grad():
			state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
			goal = torch.FloatTensor(goal).to(self.device).unsqueeze(0)
			action, _, _ = self.actor.sample(state, goal)
		return action.cpu().data.numpy().flatten()
		
	def value(self, state, goal):
		_, _, action = self.actor.sample(state, goal)
		V = self.critic(state, action, goal).min(-1, keepdim=True)[0]
		return V

	def sample_action(self, state, goal):
		# Sample action, subgoals and KL-divergence
		action_dist = self.actor(state, goal)
		action = action_dist.rsample()

		action = torch.tanh(action)
		return action

	def train(self, state, action, reward, next_state, done, goal):
		""" Critic """
		# Compute target Q
		with torch.no_grad():
			next_action, _, _ = self.actor.sample(next_state, goal)
			target_Q = self.critic_target(next_state, next_action, goal)
			target_Q = torch.min(target_Q, -1, keepdim=True)[0]
			target_Q = reward + (1.0-done) * self.gamma*target_Q

		# Compute critic loss
		Q = self.critic(state, action, goal)
		critic_loss = 0.5 * (Q - target_Q).pow(2).sum(-1).mean()

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		""" Actor """
		# Sample action
		action= self.sample_action(state, goal)

		# Compute actor loss
		Q = self.critic(state, action, goal)
		Q = torch.min(Q, -1, keepdim=True)[0]
		actor_loss =  - Q.mean()

		# Optimize the actor 
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Update target networks
		self.total_it += 1
		if self.total_it % self.target_update_interval == 0:
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


		# Log variables
		if self.logger is not None:
			self.logger.store(
				actor_loss   = actor_loss.item(),
				critic_loss  = critic_loss.item(),				
			)
