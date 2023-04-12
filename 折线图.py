import matplotlib.pyplot as plt

acc = [77.76,76.51,78.03,77.71,78.08]

y_tick=[0,20,40,60,80,100]#自己改
x=[0,1,2,3,4]
plt.plot(x,acc)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("Accuracy Plot")
plt.xticks(x)
plt.yticks(y_tick)
plt.show()