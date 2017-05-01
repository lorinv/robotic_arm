import matplotlib.pyplot as plt

lines = open("avgDeltas.txt","r").readlines()

#q deltas
q_ds = eval(lines[0])
#sarsa deltas
s_ds = eval(lines[1])

xs = [i for i in range(len(q_ds))]
plt.plot(xs, q_ds, color="green")
plt.plot(xs, s_ds, color="blue")
plt.suptitle("Q-Value Convergence Via Average Delta Per Episode",fontweight="bold")
plt.legend(["Watkins Q-Lambda","SARSA-Lambda"])
plt.ylabel("Delta")
plt.xlabel("Episode (offline)")
plt.savefig("perfMultiple.png")
plt.show()




