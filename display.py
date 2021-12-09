
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import seaborn_image as isns

# losses for each model on a graph


# color vs accuracy graph
"""
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

colors = [c for i, c in numToColor.items()]*3
colorAcc = [1,1,2,3,4,5,6,7,8,9,10,11] + [1,1,2,3,4,5,6,7,8,9,10,11] + [1,1,2,3,4,5,6,7,8,9,10,11]
models = ['feed forward model']*12 + ['cnn by bands']*12 + ['cnn by resistor']*12

d = {'Colors': colors, 'Accuracy': colorAcc, "Models": models}
df = pd.DataFrame(data=d)
print(df)

sns.set_style("darkgrid", {'axes.grid' : False})

g = sns.catplot(
    data=df, kind="bar",
    x="Models", y="Accuracy", hue="Colors", palette = colors, alpha=.6
    #palette="dark", alpha=.6
)

g.despine(left=True)
#g.set_axis_labels("", "Body mass (g)")
g.legend.set_title("")

plt.show()
"""

"""
############ Accuracies for each model ############
forward_model_acc = 0.3
cnn_model_acc = 0.1
cnn_model_r_acc = 0.5

names = ['feed forward model', 'cnn by bands', 'cnn by resistor']
x = [5, 6, 15]
d = {'Models': names, 'Accuracy': x}
df = pd.DataFrame(data=d)

sns.set_style("darkgrid")

ax = sns.barplot(x = "Models", y = "Accuracy", data = df, palette=("Blues_d")).set(title='Accuracy found by Models')
sns.set(style='dark')
sns.set_context("poster")
isns.set_context(mode="poster", fontfamily="sans-serif")

plt.show()
"""