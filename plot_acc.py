import matplotlib.pyplot as plt
import numpy as np
#plt.style.use('seaborn-deep')

params = {
            'axes.labelsize': '16',
            'xtick.labelsize': '12',
            'ytick.labelsize': '10',
            'lines.linewidth': '2',
            'legend.fontsize': '20',
            'figure.figsize': '5, 5'  # set figure size
        }
plt.rcParams.update(params)

x = [0,1,2,3]
y1=[90.8,91.8,93.6,92.2]
y2=[90.6,92.3,94.3,93.6]

x1=[0.01,0.05,0.1,0.5]
y3=[90.2,93.6,91.8,90.7]
y4=[90.3,94.3,92.6,89.3]
fig, ax = plt.subplots()

ax.set_ylabel('(%)')

labels = ['0.01','0.05','0.1','0.5']
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
#, 'ro-' 'g*:',
plt.plot(x, y3, 'ro-',label='A>W')
plt.plot(x, y4, 'g*:',label='A>D')
plt.ylim(70, 100)
plt.legend()

plt.savefig('plot_acc_lamda.png', format='png', dpi=1000)
#plt.show()
