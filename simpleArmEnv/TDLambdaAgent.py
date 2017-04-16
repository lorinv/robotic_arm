import numpy as np
import random
import sys
import os

try:
    xrange
except NameError:
    xrange = range    #python3 has no xrange function; python3 range is python2's xrange

    
class TDLambdaAgent(object):
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
            tdAlgorithm="watkins",   # "watkins" or "sarsa"
            useExperienceCache=True,
            qFilePath=None, #Path to a q-file, containing saved q-values from previous training. If None, agent is initialized with random q-values.
            resetQVals=False,
            resetCache=False): #if True, blow away the old experienceCache.txt file; DO THIS WHENEVER STATE-ACTION SPACE IS CHANGED
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.randomActionRate = randomActionRate
        self.randomActionDecayRate = randomActionDecayRate
        self.state = 0
        self.action = 0
        self.stepCtr = 0 #not an algorithmic parameter, just a way of tracking/signalling good times to write files and other expensive things, for instances
        self.dtype = "float32"
        
        #init the q values
        self._initQValues(qFilePath, resetQVals, num_states, num_actions)
        
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

        #is useExperienceCache, then configure the agent to cache k last experiences, such as for Dyna-Q paradigms
        self.InitCache(useExperienceCache, resetCache)

    def _initQValues(self, qFilePath, resetQVals, numStates, numActions):
        """
        If qFilePath is not None, q-values are read from this path. If reading fails, or if @qFilePath==None, then
        random q values are initialized.
        
        @resetQVals: if true, q values will be initialized to random vals
        """
        self.qFilePath = qFilePath
        if qFilePath and not resetQVals: #qFilePath not None, so attempt to read previous q values from file
            #check the file exists and is not empty
            if not os.path.exists(qFilePath) or os.path.getsize(qFilePath) <= 0:
                print "WARNING q-file "+qFilePath+" empty or doesn't exist. Initializing random q-values."
                self.qtable = self._randQArray(numStates, numActions)
            else: #file exists, so attempt to read previous values
                if not self._readQFile(qFilePath):
                    print "ERROR there was a problem reading qvalues from "+qFilePath+". Using random default init instead..."
                    self.qtable = self._randQArray(numStates, numActions)
                #read succeeded, but verify qtable size is consistent over disk read/writes; otherwise we could screw up every time we alter our state/action size
                elif self.qtable.shape[0] != numStates or self.qtable.shape[1] != numActions:
                        print "ERROR read qtable from "+self.qFilePath+" is size "+str(self.qtable)+" but target is size "+str((numStates,numActions))
                        print "Q-Table will be reinitialized with random values!"
                        self.qtable = self._randQArray(numStates,numActions)
        else: #qFilePath is None, so init random, small q values
            print "Initializing agent with random q values..."
            self.qtable = self._randQArray(numStates, numActions)

    def _randQArray(self, numStates, numActions):
        return np.random.uniform(low=-1, high=1, size=(numStates, numActions)).astype(self.dtype)
            
    def _readQFile(self, qFilePath):
        """
        Simple function for desrializing a q-table from @qFilePath, which is required to contain
        Returns true if values read successfully, False otherwise.
        """
        success = False

        try:
            self.qtable = np.loadtxt(self.qFilePath, dtype=self.dtype)
            success = True
        except:
            print "ERROR could not read q values from "+qFilePath

        return success
        
    def _writeQFile(self, qFilePath):
        """
        Writes current q-values to @qFilePath, overwriting any previous stored values.
        
        NOTE this is directly paired with _readQFile.
        """
        success = False
        try:
            np.savetxt(self.qFilePath, self.qtable, fmt='%f')
            success = True
        except:
            print "ERROR problem encountered writing q values to file."
        
        return success
                
    def PrintQValues(self):
        for row in range(self.qtable.shape[0]):
            rowStr = ""
            for col in range(self.qtable.shape[1]):
                rowStr += str(self.qtable[row,col])[0:4] + " "
            #rowStr += "\n"
            print(rowStr)
	print("\n")
        
    """
    A pass-through null-function so the code is agnostic to the cache.
    """
    def _nullCache(self, state, action, reward, statePrime):
        pass

    def _appendToCache(self, state, action, reward, statePrime):
        experience = (state, action, reward, statePrime)
        self._experienceCache.append(experience)
        self._cacheFile.write(str(experience)+"\n")

    """
    A full reset of the agent to its base state, wiping all q-values.
    """
    def HardReset(self):
        self.qtable = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))
        self.ResetEligibilities()
        self.InitCache()
        
    """
    @useExperienceCache: If true, we will store all experiences in an experience cache, and periodically write them to experienceCache.txt
    @resetCache: If true, the experienceCache.txt file will be opened for write, blowing away previous file content and restarting the cache. This
                         param only applies if useExperienceCache==True.
    """
    def InitCache(self, useExperienceCache, resetCache):
        if useExperienceCache:
            #point the _appendCache() call at the desired function
            self._cacheExperience = self._appendToCache
            #select whether or not to blow away old experienceCache.txt file
            if resetCache:
                fileMode = "w+"
            else:
                fileMode = "a+"
            #cache file just stores the cached experiences as tuples; writing them as str(tuple) means they can be read back in just reading each line and calling eval(line)
            self._cacheFile = open("data/experienceCache.txt",fileMode)
            #output a line of dashes signifying a new set of experiences
            self._cacheFile.write("----------------------New Run----------------------\n")
            #cache is a list of previous transitions as tuples: (state, statePrime, action, reward)
            self._experienceCache = []
        else:
            self._cacheExperience = self._nullCache
            
    def InitState(self, state):
        self.state = state
        #raw_input(state)
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
        
        #this param is not an algorithmic parameter, just a way of tracking and controlling good times to write files, other periodic bookkeeping tasks, etc
        self.stepCtr += 1
        if self.stepCtr % 10 == 9: #write the qvalues to file every 200 steps, for America. Writing files is expensive, so only do so periodically.
            self._writeQFile(self.qFilePath)
        
        actRandomly = (1 - self.randomActionRate) <= self._getRand()
        if actRandomly:
            actionPrime = random.randint(0, self.num_actions - 1)
        else:
            actionPrime = self.qtable[statePrime].argmax()

        #accounting required for watkins q(lambda) algorithm            
        isMaxAction = self.qtable[statePrime].argmax() == actionPrime

        self.randomActionRate *= self.randomActionDecayRate
    
        #cache the experience, for possible offline learning methods like dyna-q. If no cache, this function is a pass-through.
        self._cacheExperience(self.state, self.action, reward, statePrime)
    
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

        #for watkins q(lambda) eligibilities are completely reset when non-max action is taken
        if self.isWatkins and not isMaxAction:
            self.ResetEligibilities()

        self.state = statePrime
        self.action = actionPrime

        return self.action

    #These two are obsolete, but are useful for more complex state spaces, such as tilings of contniuous spaces
    def buildState(self, features):
        return int("".join(map(lambda feature: str(int(feature)), features)))
    def getBin(self, value, bins):
        return np.digitize(x=[value], bins=bins)[0]
