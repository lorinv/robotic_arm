from TDLambdaAgent import *
import matplotlib.pyplot as plt
import time
import random

"""
This isn't a formal script or set of tests yet. For now just playing around 
training a td_lambda agent offline, using previous data in the experienceCache.txt file,
which is a bunch of tuples representing (state, action, reward, statePrime).


(state, statePrime, action, reward)
"""

#reads all experiences back in as 
experiences = [eval(line) for line in open("./data/archived_FiveByFive_Experiences.txt","r").readlines() if len(line.strip()) > 0 and "---" not in line]

num_states = 26
num_actions = 4
alpha = 0.1
gamma = 0.9
randomActionDecayRate = 0.99
randomActionRate = 0.2
tdLambda = 0.9
traceMethod = "normal"
algorithm = "sarsa"

"""
learner = TDLambdaAgent(num_states, num_actions, alpha, gamma,
                                        randomActionRate, randomActionDecayRate, tdLambda,
                                        200, traceMethod, algorithm, False, "qValues.txt", False, False)
"""

iterations = 40
performanceIterations = 10
numExperiences = len(experiences)
#rewardLists = [[] for i in range(iterations*numExperiences)]
deltas = []

for perfIt in range(performanceIterations):
    learner = TDLambdaAgent(num_states, num_actions, alpha, gamma,
                                            randomActionRate, randomActionDecayRate, tdLambda,
                                            200, traceMethod, algorithm, False, "qValues.txt", True, False)

    for i in range(iterations):
        for j in range(numExperiences): #the tuples are stored as (state, action, reward, statePrime)
            experience = experiences[random.randint(0,len(experiences)-1)]
            #experience = experiences[j]
            state = experience[0]
            action = experience[1]
            reward = experience[2]
            statePrime = experience[3]

            #re-learn previous data
            learner.state = state
            learner.action = action
            learner.move(statePrime, reward)
            #print("Iteration: "+str(i))
            #learner.PrintQValues()
            #rewardLists[i*numExperiences + j].append(reward)

    deltas.append(learner.deltas)

print(str(deltas))
print([str(len(deltas[i])) for i in range(len(deltas))])

avgDeltas = []
for i in range(len(deltas[0])):
    sumDelta = 0.0
    for j in range(len(deltas)):
        sumDelta += deltas[j][i]
    avgDeltas.append(sumDelta / float(len(deltas)))

k = 500
avgDeltas = [sum(avgDeltas[i:i+k])/float(k) for i in range(len(avgDeltas)-k)]

print(str(avgDeltas))

xs = [i for i in range(len(avgDeltas))]

open("avgDeltas.txt","a+").write(str(avgDeltas)+"\n")

plt.suptitle("Q-Value Convergence Via Average Delta Per Episode",fontweight="bold")
plt.ylabel("Delta")
plt.xlabel("Episode (offline)")
plt.plot(xs, avgDeltas)
plt.savefig("perf.png")
plt.show()





