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
		obs, reward, done, _ = env.step(int(policy[obs]))
		total_reward+=reward
		step_idx+=1
		if done:
			break
	return total_reward

def evaluate_policy(env, policy, gamma=1.0, n=100):
	scores = [run_episode(env,policy) for _ in range(n)]
	return np.mean(scores) 

def extract_policy(v, gamma=1.0):
	policy = np.zeros(env.nS)
	for s in range(env.nS):
		q_sa = np.zeros(env.nA)
		for a in range(env.nA):
			for next_arr in env.P[s][a]:
				p,s_,r,_ = next_arr
				q_sa[a] += (p*(r+(gamma*v[s_])))
		policy[s] = np.argmax(q_sa)
	return policy

def compute_policy_v(env,policy,gamma=1.0):
	v = np.zeros(env.nS)
	eps = 1e-10
	while True:
		prev_v = np.copy(v)
		for s in range(env.nS):
			policy_a = policy[s]
			v[s] = sum([p*(r+(gamma*prev_v[s_])) for p,s_,r,_ in env.P[s][policy_a]])
		if(np.sum((np.fabs(prev_v-v)))<=eps):
			break
	return v

def policy_iteration(env, gamma=1.0):
	policy = np.random.choice(env.nA,size=(env.nS))
	max_iterations = 200000
	gamma = 1.0
	for i in range(max_iterations):
		print("At ",i)
		old_policy_v = compute_policy_v(env,policy,gamma)
		new_policy = extract_policy(old_policy_v,gamma)
		if(np.all(policy==new_policy)):
			print('Converged at',i)
			break
	return policy


env_name = 'FrozenLake8x8-v0'
env = gym.make(env_name)
optimal_policy = policy_iteration(env,gamma=1.0)
scores = evaluate_policy(env,optimal_policy)
print('Average scores =',np.mean(scores))	