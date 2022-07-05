# Task 4: Reinforcement Learning

Official text, credits to : [Authors](https://las.inf.ethz.ch/teaching/pai-f21).


**Task description**

Your task is to control a lunar lander in a smooth descent to the ground. It must land in-between two flags, quickly, using minimal fuel, and without damaging the lander. Implement a reinforcement learning algorithm that, by practicing on a simulator, learns a control policy for the lander.


**Environment and scoring details**

At each discrete time step, the controller can maneuver the lander by taking one of four actions: doing nothing, firing the left engine, firing the right engine, or firing the main engine. The goal is to accumulate as much reward as possible in 300 timesteps. In evaluation, after 300 timesteps the episode ends. Positive reward is obtained for landing between the flags and landing on the lander’s legs. Negative reward is obtained for firing the main engine or side engines (more negative for the main engine) or for crashing the lander. Note that the lander only obtains positive reward for land- ing on its legs: if you land so fast that the legs hit the ground followed by the main lander body, it is counted as only crashing. Since the focus of this task is the implementation of reinforcement learning algorithms, it is not necessary to have a detailed understanding of the mechanics of the lunar lander environment beyond the observation and action space sizes (given in the train function of ‘solution.py’).


When you run ‘solution.py’, your algorithm will have access to the standard lunar lander environment that is commonly used in evaluating RL algorithms. To run ‘solution.py‘ you will need to install the packages listed in ‘requirements.txt’. You are encouraged to run ‘solution.py’ for testing.

In a single run of ‘runner.sh’, you will be able to query up to 150000 transitions (single timepoints) in order to learn a policy for the modified lunar lander environment. Each individual episode can contain up to 300 transitions. The score is then based upon the average performance of the learned policy over 100 episodes after the training episodes.In other words, the final score is an estimate of the expected cumulative reward of your final policy over an episode. Your goal is to maximize this final score.
