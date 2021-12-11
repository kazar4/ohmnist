
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import seaborn_image as isns
import tensorflow as tf

# Program to create plots

# losses for each model on a graph
# use something like this to get the actual data
m1 = tf.data.experimental.load("./loss_densemodel.db")
m2 = tf.data.experimental.load("./loss_onebandcnn.db")
m3 = tf.data.experimental.load("./loss_fullresistor.db")
lossesM1 = []
for i in m1:
    lossesM1.append(float(i))
lossesM2 = []
for i in m2:
    lossesM2.append(float(i))
lossesM3 = []
for i in m3:
    lossesM3.append(float(i))

batchNums = [i for i in range(len(lossesM1))] + [i for i in range(len(lossesM2))] + [i for i in range(len(lossesM3))]
losses = lossesM1 + lossesM2 + lossesM3
models = ['feed forward model']*len(lossesM1) + ['cnn by bands']*len(lossesM2) + ['cnn by resistor']*len(lossesM3)

d = {'Batch': batchNums, 'Loss': losses, "Model": models}
df = pd.DataFrame(data=d)
#print(df)

sns.set_style("darkgrid", {'axes.grid' : False})

# Plot the responses for different events and regions
g = sns.lineplot(x="Batch", y="Loss",
             hue="Model",
             data=df).set(title='Model Losses Per Batch')

#g.despine(left=True)
#g.legend.set_title("Model Losses Per Batch")

plt.show()


"""
# color vs accuracy graph

numToColor = {
    0:"Black",
    1:"Brown",
    2:"Red",
    3:"Orange",
    4:"Yellow",
    5:"Green",
    6:"Blue",
    7:"Purple",
    8:"Gray",
    9:"White",
    10:"Gold",
    11:"Silver"
}

#penguins = sns.load_dataset("penguins")
#print(penguins)

colors = [c for i, c in numToColor.items()]
#*3
colorAcc = [293, 378, 95, 57, 11, 53, 42, 33, 36, 23, 59, 3]
#+ [1,1,2,3,4,5,6,7,8,9,10,11] + [1,1,2,3,4,5,6,7,8,9,10,11]
#models = ['feed forward model']*12 
#+ ['cnn by bands']*12 + ['cnn by resistor']*12

d = {'Colors': colors, 'Resistor Band Count': colorAcc}
#"Models": models}
df = pd.DataFrame(data=d)
print(df)

sns.set_style("darkgrid", {'axes.grid' : False})

g = sns.barplot(
    data=df,
    x="Colors", y="Resistor Band Count", palette = colors, alpha=.6
    #palette="dark", alpha=.6
).set(title='Number of Bands for Each Color in Dataset')

#g.fig.subplots_adjust(top=.95)
#g.despine(left=True)
#g.set_axis_labels("", "Body mass (g)")
#g.legend.set_title("Model Accuracy Per Color")

plt.show()
"""

"""
############ Accuracies for each model ############
forward_model_acc = 0.265625
cnn_model_acc = 0.291666
cnn_model_r_acc = 0.579487

names = ['Feed Forward Model', 'CNN by Bands', 'CNN by Resistor']
x = [forward_model_acc, cnn_model_acc, cnn_model_r_acc]
d = {'Models': names, 'Accuracy': x}
df = pd.DataFrame(data=d)

sns.set_style("darkgrid")

ax = sns.barplot(x = "Models", y = "Accuracy", data = df, palette=("Blues_d")).set(title='Accuracy found by Models')
sns.set(style='dark')
sns.set_context("poster")
isns.set_context(mode="poster", fontfamily="sans-serif")

#ax.legend.set_title("Accuracy Per Model")

plt.show()
"""