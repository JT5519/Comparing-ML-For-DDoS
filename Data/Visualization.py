import pandas as pd
import matplotlib.pyplot as plt
icmp = pd.read_csv('ICMP.csv')
colors = {'ICMPFlood':'r','ICMPNormal':'g'}
fig,ax = plt.subplots()
L = list(range(1,1000)) + list(range(len(icmp['Delta Time'])-1000,len(icmp['Delta Time'])-1))
for i in L:
    ax.scatter(icmp['Delta Time'][i],icmp['Length'][i],color=colors[icmp['Class'][i]])
    #ax.scatter(icmp['Delta Time'][i],icmp['Cumulative Bytes'][i],color=colors[icmp['Class'][i]])
ax.set_title('ICMP Dataset')
ax.set_xlabel('Delta Time')
ax.set_ylabel('Length')
#ax.set_ylabel('Cumulative Bytes')
plt.show()    
