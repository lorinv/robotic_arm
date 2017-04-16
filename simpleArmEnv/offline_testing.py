from TDLambdaAgent import *
import matplotlib.pyplot as plt
import time

"""
This isn't a formal script or set of tests yet. For now just playing around 
training a td_lambda agent offline, using previous data in the experienceCache.txt file,
which is a bunch of tuples representing (state, action, reward, statePrime).


(state, statePrime, action, reward)
"""

#reads all experiences back in as 
experiences = [eval(line) for line in open("./data/experienceCache.txt","r").readlines() if len(line.strip()) > 0 and "-----" not in line]

num_states = 10
num_actions = 5
alpha = 0.1
gamma = 0.9
randomActionDecayRate = 0.9
randomActionRate = 0.2
tdLambda = 0.99
traceMethod = "normal"
algorithm = "sarsa"

learner = TDLambdaAgent(num_states, num_actions, alpha, gamma,
                                        randomActionRate, randomActionDecayRate, tdLambda,
                                        200, traceMethod, algorithm, True, "qValues.txt", False, False)

for i in range(50):
    for experience in experiences: #the tuples are stored as (state, action, reward, statePrime)
        state = experience[0]
        action = experience[1]
        reward = experience[2]
        statePrime = experience[3]

        #re-learn previous data
        learner.state = state
        learner.action = action
        learner.move(statePrime, reward)
        learner.PrintQValues()
    
