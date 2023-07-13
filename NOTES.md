
* currently only trying on cartpole but initial results are not as bad as it could have been, its significantly less stable and suffers from exploding Q-values a lot of the time but apart from that performance is very similar to vanilla DQN at early stages.

* using sampling for estimating Y seems to perform better

* could maybe work on something with improving the policy, such as picking actions based on standard deviation?

* could also try a biased normal distribution when estimating Y?

* finally can try with DDQN to curb exploding Q-values