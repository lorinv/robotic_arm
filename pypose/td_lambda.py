

from __future__ import print_function
import gym
import pandas as pd
import numpy as np
import random
import sys

#For Robotic Arm
from rl_env import *

try:
    xrange
except NameError:
    xrange = range	#python3 has no xrange function; python3 range is python2's xrange

class TDLambdaLearner(object):
	def __init__(self,
			num_states=100,
			num_actions=4,
			alpha=0.2,
			gamma=0.9,
			randomActionRate=0.5,
			randomActionDecayRate=0.9,
			tdLambda=0.5,  #the lambda parameter
			nTraces=200,  #the number of eligibility traces to store; cart-pole episodes are bounded by 200 steps
			traceUpdate="standard", # standard or 'replacing'
			tdAlgorithm="watkins"): # "watkins" or "sarsa"
		self.num_states = num_states
		self.num_actions = num_actions
		self.alpha = alpha
		self.gamma = gamma
		self.randomActionRate = randomActionRate
		self.randomActionDecayRate = randomActionDecayRate
		self.state = 0
		self.action = 0
		self.qtable = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))
		self._lambda = tdLambda
		if tdAlgorithm.lower() == "sarsa":
			self.isWatkins = False
		else:
			self.isWatkins = True
		#the traces
		self.eligibilityRing = np.zeros((1,nTraces),dtype=np.uint32)
		self.eligibilityTable = np.zeros((num_states, num_actions),dtype=np.uint32)
		self.ResetEligibilities()
		self.nTraces = nTraces
		
		if traceUpdate == "replacing":
			self._traceUpdate = self._replacingTraceUpdate
		else:
			self._traceUpdate = self._incrementingTraceUpdate
			
		#cache a prime number of random numbers to avoid rand() calls
		self.randNums = [np.random.uniform(0, 1) for i in xrange(4091)]
		self.randNumIndex = 0

	def InitState(self, state):
		self.state = state
		self.action = self.qtable[state].argmax()
		return self.action

	def _updateEligibilityRing(self,state,action):
		#add current state-action pair to ring-index; encoded as a uint: (state << 8 | action)
		self.eligibilityRing[0,self.ringIndex] = (state << 8) | action
		self.ringIndex += 1
		if self.ringIndex >= self.eligibilityRing.shape[1]:
			self.ringIndex = 0

	def ResetEligibilities(self):
		self.eligibilityRing[0,:] = 0xFFFFFFFF
		self.eligibilityTable[:,:] = 0.0
		self.ringIndex = 0

	"""
	Standard eligibility update: increment most-recently visited state by one.
	"""
	def _incrementingTraceUpdate(self,state,action):
		self._updateEligibilityRing(state,action)
		#update this state-action pair's eligibility
		self.eligibilityTable[state,action] += 1

	"""
	Replacing traces update: pin most recently visited state to one.
	"""
	def _replacingTraceUpdate(self,state,action):
		self._updateEligibilityRing(state,action)
		#update this state-action pair's eligibility
		self.eligibilityTable[state,action] = 1

	"""
	Get a random number from cached random numbers
	"""
	def _getRand(self):
		self.randNumIndex += 1
		if self.randNumIndex >= 4091:
			self.randNumIndex = 0
		return self.randNums[self.randNumIndex]

	"""
	@statePrime: the new state
	@reward: the reward just acquired
	"""
	def move(self, statePrime, reward):
		
		actRandomly = (1 - self.randomActionRate) <= self._getRand()

		if actRandomly:
			actionPrime = random.randint(0, self.num_actions - 1)
		else:
			actionPrime = self.qtable[statePrime].argmax()

		#accounting required for watkins q(lambda) algorithm            
		isMaxAction = self.qtable[statePrime].argmax() == actionPrime

		self.randomActionRate *= self.randomActionDecayRate
	
		if self.isWatkins:
			#watkins Q(lambda) straight from barto: Figure 7.14. update made wrt to max-q_prime
			delta = reward + self.gamma * self.qtable[statePrime].max() - self.qtable[self.state, self.action]
		else:
			#sarsa: update made wrt action taken
			delta = reward + self.gamma * self.qtable[statePrime, actionPrime] - self.qtable[self.state, self.action]
			
		#update the traces (incrementing or replacing trace update)
		self._traceUpdate(self.state, self.action)
		#for all (state,action) pairs with non-zero eligibility, update em; this is done fastest by marching backward from current ringIndex
		i = self.ringIndex - 1
		while i >= 0 and self.eligibilityRing[0,i] != 0xFFFFFFFF: #TODO: this loop construction assumes eligibility ring index never wraps!
			encoded = self.eligibilityRing[0,i]
			action = encoded & 0xFF
			state = encoded >> 8
			#update q value
			#self.qtable[state,action] = (1.0 - self.alpha) * self.qtable[state,action] + self.alpha * delta * self.eligibilityTable[state,action]
			self.qtable[state,action] += self.alpha * delta * self.eligibilityTable[state,action]
			#decay eligibilities
			self.eligibilityTable[state,action] = self._lambda * self.gamma * self.eligibilityTable[state,action]               
			i -= 1

		#for watkins q(lambda) eligibilities are complete reset instead of decayed
		if self.isWatkins and not isMaxAction:
			self.ResetEligibilities()

		self.state = statePrime
		self.action = actionPrime

		return self.action

def buildState(features):
	return int("".join(map(lambda feature: str(int(feature)), features)))

def getBin(value, bins):
	return np.digitize(x=[value], bins=bins)[0]

def main():
	#env = gym.make('CartPole-v0')
	#experimentFilename = './cartpoleExperiment'
	env = Arm_Env(grid_dim=(8,8), num_motors=3)
	experimentFilename = './roboticArmExperiment'
	
	#env.monitor.start(experimentFilename, force=True)

	perfLog = open("performance.txt","a+")
	perfLog.write("\n") #start convergence values on a fresh line
	
	reward = 0.0
	goalMeanSteps = 195
	maxSteps = 200
	meanIterations = 100
	maxEpisodes = 50000
	
	#numFeatures = env.observation_space.shape[0]
	numFeatures = env.num_states
	episodeSteps = np.ndarray(0)

	cartPositionBins = pd.cut([-2.4, 2.4], bins=10, retbins=True)[1][1:-1]
	poleAngleBins = pd.cut([-2, 2], bins=10, retbins=True)[1][1:-1]
	cartVelocityBins = pd.cut([-1, 1], bins=10, retbins=True)[1][1:-1]
	angleRateBins = pd.cut([-3.5, 3.5], bins=10, retbins=True)[1][1:-1]
	
	#decent default values, just based on observation
	numTraces = 200
	maxEpisodes = 50000 # the standard max, per the original cart-pole spec
	alpha =0.2
	gamma = 0.99
	randomActionRate = 0.5
	randomActionDecayRate = 0.9999
	tdLambda = 0.5
	algorithm = "sarsa" # "sarsa" or "watkins" for q-learning
	traceMethod = "normal" # "normal" for normal (per Barto), or "replacing" for replacing traces (see Barto)
	
	#get any cmd line params
	for arg in sys.argv:
		if "alpha=" in arg:
			alpha = float(arg.split("=")[1])
		elif "gamma=" in arg:
			gamma = float(arg.split("=")[1])
		elif "randomRate=" in arg:
			randomActionRate = float(arg.split("=")[1])
		elif "randomDecay=" in arg:
			randomActionDecayRate = float(arg.split("=")[1])
		elif "lambda=" in arg:
			tdLambda = float(arg.split("=")[1])
		elif "algorithm=" in arg:
			algorithm = arg.split("=")[1]
		elif "traceMethod=" in arg:
			traceMethod = arg.split("=")[1]
		elif "maxEpisodes=" in arg:
			maxEpisodes = int(arg.split("=")[1])

	"""
	Some good params, so far: alpha 0.2, gamma 0.9, randomActionDecayRate 0.5, randomActionDecayRate 0.9999, tdLambda 0.6/0.5
	"""
	learner = TDLambdaLearner(10 ** numFeatures, env.num_actions, alpha, gamma,
												randomActionRate, randomActionDecayRate, tdLambda,
												numTraces, traceMethod, algorithm)

	for episode in xrange(maxEpisodes):
		observation = env.reset()
		cartPosition, poleAngle, cartVelocity, angle_ROC = observation
		state = buildState([getBin(cartPosition, cartPositionBins),
							 getBin(poleAngle, poleAngleBins),
							 getBin(cartVelocity, cartVelocityBins),
							 getBin(angle_ROC, angleRateBins)])
		#reset the learner's eligibility traces for this episode
		learner.ResetEligibilities()
		action = learner.InitState(state)

		if episode % 100 == 99:
			print("Episode: "+str(episode)+"  "+str(episodeSteps.mean())+"  last reward: "+str(reward)+"  rand_rt: "+str(learner.randomActionRate))
			print(str(episodeSteps))
			perfLog.write(str(episodeSteps.mean())+",")

		for step in xrange(maxSteps - 1):
			observation, reward, done, info = env.step(action)
			cartPosition, poleAngle, cartVelocity, angle_ROC = observation

			statePrime = buildState([getBin(cartPosition, cartPositionBins),
									   getBin(poleAngle, poleAngleBins),
									   getBin(cartVelocity, cartVelocityBins),
									   getBin(angle_ROC, angleRateBins)])

			if done:
				#print("R: "+str(reward))
				reward = -200

			action = learner.move(statePrime, reward)

			if done:
				episodeSteps = np.append(episodeSteps, [int(step + 1)])
				if len(episodeSteps) > meanIterations:
					episodeSteps = np.delete(episodeSteps, 0)
				break

		if episodeSteps.mean() > goalMeanSteps:
			print("MEAN: "+str(episodeSteps.mean()))
			print("GOAL REACHED")
			print("Max episodes: "+str(episode + 1))
			break

	perfLog.close()
	#env.monitor.close()

if __name__ == "__main__":
	random.seed(0)
	main()