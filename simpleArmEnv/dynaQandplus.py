import time
import sys
import numpy as np
sys.path.append("../")
from envs.mdp_gridworld import MDPGridworldEnv

env = MDPGridworldEnv()

def dyna_q(episodes, planning_steps, alpha, epsilon, gamma):
    aselection = EGreedySelection(epsilon)

    q = np.zeros((env.observation_space.n, env.action_space.n))
    q.fill(float('-inf'))

    model_nextR = np.zeros((env.observation_space.n, env.action_space.n))
    model_nextS = np.zeros((env.observation_space.n, env.action_space.n))

    visited = {}

    for s in range(env.observation_space.n):
        actions = range(0, env.action_space.n)
        for a in actions:
            q[s, a] = 0

    for _ in range(episodes):
        s = env.reset()
	terminal = False;

        while not terminal:
	    env.render()
            a = aselection(q[s], actions)

            if s not in visited:
                visited[s] = set()
            visited[s].add(a)

	    nextS, r, terminal, _ = env.step(a)

            model_nextS[s, a] = nextS
            model_nextR[s, a] = r

            a_star_next = argmax_random(actions)
            q[s, a] = q[s, a] + alpha * (r + gamma * q[nextS, a_star_next] - q[s, a])

            real_nextS = nextS

            for _ in range(planning_steps):
                s = np.random.choice(list(visited.keys()))
                a = np.random.choice(tuple(visited[s]))

                nextS, r = model_nextS[s, a], model_nextR[s, a]

                a_star_next = argmax_random(actions)
                q[s, a] = q[s, a] + alpha * (r + gamma * (q[int(nextS), a_star_next] - q[s, a]))

            s = real_nextS

# Kappa is exploration bonus scalar
def dyna_q_plus(episodes, planning_steps, alpha, epsilon, gamma, kappa):
    aselection = EGreedySelection(epsilon)

    q = np.zeros((env.observation_space.n, env.action_space.n))
    q.fill(float('-inf'))

    model_nextR = np.zeros((env.observation_space.n, env.action_space.n))
    model_nextS = np.zeros((env.observation_space.n, env.action_space.n))
    last_visit =  np.zeros((env.observation_space.n, env.action_space.n))

    visited = {}

    for s in range(env.observation_space.n):
        actions = range(0, env.action_space.n)
        for a in actions:
            q[s, a] = 0

    for e in range(episodes):
        s = env.reset()
	terminal = False;

        while not terminal:
	    env.render()
            a = aselection(q[s], actions)

            if s not in visited:
                visited[s] = set()
            visited[s].add(a)
	    last_visit[s, a] = e

	    nextS, r, terminal, _ = env.step(a)

            model_nextS[s, a] = nextS
            model_nextR[s, a] = r

            a_star_next = argmax_random(actions)
            q[s, a] = q[s, a] + alpha * (r + gamma * q[nextS, a_star_next] - q[s, a])

            real_nextS = nextS

            for _ in range(planning_steps):
                s = np.random.choice(list(visited.keys()))
                a = np.random.choice(tuple(visited[s]))

                nextS, r = model_nextS[s, a], model_nextR[s, a]
		time_since_visit = e - last_visit[s, a]
		r += kappa * np.sqrt(time_since_visit)

                a_star_next = argmax_random(actions)
                q[s, a] = q[s, a] + alpha * (r + gamma * (q[int(nextS), a_star_next] - q[s, a]))

            s = real_nextS

def argmax_random(A):
    arg = np.argsort(A)[::-1]
    n_tied = sum(np.isclose(A, A[arg[0]]))
    return np.random.choice(arg[0:n_tied])


class EGreedySelection:

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, q, actions):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(actions)
        else:
            return actions[argmax_random(q[actions])]


'''
def main():
	dyna_q_plus(3, 5, .7, .05, .7, .01)
	dyna_q(5, 5, .7, .05, .7)

main()
'''
