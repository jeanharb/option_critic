From the paper:

Pierre-Luc Bacon, Jean Harb, Doina Precup. "The Option-Critic Architecture". Thirthy-first AAAI Conference On Artificial Intelligence (AAAI), 2017.


> We first consider a navigation task in the four-rooms domain
(Sutton, Precup, and Singh 1999). Our goal is to evaluate
the ability of a set of options learned fully autonomously
to recover from a sudden change in the environment. (Sutton,
Precup, and Singh 1999) presented a similar experiment
for a set of pre-specified options; the options in our results
have not been specified a priori.
Initially the goal is located in the east doorway and the
initial state is drawn uniformly from all the other cells. After
1000 episodes, the goal moves to a random location in the
lower right room. Primitive movements can fail with probability
1/3, in which case the agent transitions randomly to
one of the empty adjacent cells. The discount factor was
0.99, and the reward was +1 at the goal and 0 otherwise.
We chose to parametrize the intra-option policies with Boltzmann
distributions and the terminations with sigmoid functions.
The policy over options was learned using intra-option
Q-learning. We also implemented primitive actor-critic (denoted
AC-PG) using a Boltzmann policy. We also compared
option-critic to a primitive SARSA agent using Boltzmann
exploration and no eligibility traces. For all Boltzmann policies,
we set the temperature parameter to 0.001. All the
weights were initialized to zero.

Corresponding command for 4 options, with results averaged over 100 runs:

 ``python transfer.py --baseline --discount=0.99 --epsilon=0.01 --noptions=4 --lr_critic=0.5 --lr_intra=0.25 --lr_term=0.25 --nruns=100 --nsteps=1000``
