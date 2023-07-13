import copy
import cardio_rl
import numpy as np
import torch as th
import torch.nn.functional as F

class DQN():
	def __init__(
			self,
			cfg,
			env,
			network,
			batch_size=64,
			train_freq=4,
			warmup_len=100,
			gamma=0.99,
			lr=2.3e-3,
			target_update=10,
			logger_kwargs=None):
		 
		self.cfg = cfg
		self.env = env
		self.runner = cardio_rl.Runner(
			env=env,
			policy='argmax',
			sampler=True,
			capacity=100_000,
			batch_size=batch_size,
			collector=cardio_rl.Collector(
				env=env,
				rollout_len=train_freq,
				warmup_len=warmup_len,
				logger_kwargs=logger_kwargs
			),
			backend='pytorch'
		)

		self.lr = lr
		self.network = network
		self.targ_net = copy.deepcopy(network)
		self.optimizer = th.optim.Adam(network.parameters(), lr=self.lr)

		self.gamma = gamma
		self.target_update = target_update

	def fit(self, grad_steps):
		for t in range(grad_steps):
			batch = self.runner.get_batch(self.network)
			s, a, r, s_p, d, i = batch()
			self._update(t, s, a, r, s_p, d, i)

		# add a function to runner that grabs logger stats
		return np.mean(self.runner.collector.logger.episodic_rewards)

	def _update(self, t, s, a, r, s_p, d, i):
		with th.no_grad():
			q_p = th.max(self.targ_net(s_p), keepdim=True, dim=-1).values
			y = r + self.gamma * q_p * (1 - d)

		q = self.network(s).gather(1, a.long())

		loss = F.mse_loss(q, y)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		if t % self.target_update == 0:        
			self.targ_net = copy.deepcopy(self.network)
