import gym 
from gym import wrappers
import numpy as np 

def run_episode(env, policy, gamma=1.0, render=False):
	obs = env.reset()
	total_reward = 0
	step_idx = 0
	while True:
		if render:
			env.render()
		obs,reward,done,_ = env.step(int(policy[obs]))
		total_reward+=(gamma ** step_idx * reward)
		step_idx+=1
		if done:
			break
	return total_reward

def evaluate_policy(env, policy, gamma=1.0, n=100):
	scores = [
			run_episode(env,policy,gamma=gamma,render=False)
			for _ in range(n)]
	return np.mean(scores)

def extract_policy(v,gamma=1.0):
	policy = np.zeros(env.nS)
	for s in range(env.nS):
		q_sa = np.zeros(env.action_space.n)
		for a in range(env.action_space.n):
			for next_sr in env.P[s][a]:
				prob, state_, reward, _ = next_sr
				q_sa[a] += (prob*(reward+gamma*v[state_]))
		policy[s] = np.argmax(q_sa)
	return policy

def value_iteration(env, gamma=1.0):
	v = np.zeros(env.nS)
	max_iterations = 100000
	eps = 1e-20
	for i in range(max_iterations):
		prev_v = np.copy(v)
		for s in range(env.nS):
			q_sa = [sum([p*(r+prev_v[state_]) for p,state_,r,_  in env.P[s][a]]) for a in range(env.nA)]
			v[s] = max(q_sa)
		if (np.sum(np.fabs(prev_v - v)) <= eps):
			print('Value iteration converged at',i+1)
			break
	return v

env_name = 'FrozenLake8x8-v0'
gamma = 0.5
env = gym.make(env_name)
env.monitor.start('./frozenlake-experiment-1', force=True)
optimal_v = value_iteration(env,gamma)
policy = extract_policy(optimal_v,gamma)
policy_score = evaluate_policy(env,policy,gamma,n=1000)
print('Policy average score=',policy_score)
env.monitor.close()
