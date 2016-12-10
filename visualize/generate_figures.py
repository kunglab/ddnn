import os

import matplotlib
matplotlib.rcParams['font.size'] = 20.0
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def align_y_axis(ax1, ax2, minresax1, minresax2):
    """ Sets tick marks of twinx axes to line up with 7 total tick marks

    ax1 and ax2 are matplotlib axes
    Spacing between tick marks will be a factor of minresax1 and minresax2"""

    ax1ylims = ax1.get_ybound()
    ax2ylims = ax2.get_ybound()
    ax1factor = minresax1 * 6
    ax2factor = minresax2 * 6
    ax1.set_yticks(np.linspace(ax1ylims[0],
                               ax1ylims[1]+(ax1factor -
                               (ax1ylims[1]-ax1ylims[0]) % ax1factor) %
                               ax1factor,
                               7))
    ax2.set_yticks(np.linspace(ax2ylims[0],
                               ax2ylims[1]+(ax2factor -
                               (ax2ylims[1]-ax2ylims[0]) % ax2factor) %
                               ax2factor,
                               7))


data_dir = 'data/'
save_dir = 'figures/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


img_type = '.pdf'

linewidth = 4
ms = 8
colors = {'cam': '#FF944D', 'local': '#FF8F80', 'cloud': '#98D1F1'}
styles = {'cam': '-.o', 'local': '--o', 'cloud': '-o'}
legend = {'cam': 'Individual', 'local': 'Local', 'cloud': 'Cloud'}
names = ['cloud', 'local', 'cam']

acc_df = pd.read_csv(data_dir + 'add_cam.csv', delimiter=' ')
miss_df = pd.read_csv(data_dir + 'missing_cam.csv', delimiter=' ')
comm_df = pd.read_csv(data_dir + 'comm.csv', delimiter=' ')
ent_df = pd.read_csv(data_dir + 'entropy.csv', delimiter=' ')
thres_df = comm_df[comm_df['T'] == 0.8]
idxs = np.arange(1, 7)

#Accuracy figure
plt.figure(figsize=(8, 6.5))
for name in names:
    plt.plot(idxs, acc_df[name]*100., styles[name], linewidth=linewidth, ms=ms,
             color=colors[name], label=legend[name])
plt.xticks(idxs)
plt.xlim(0.5, 6.5)
plt.xlabel('Number of Device(s)')
plt.ylabel('Classification Accuracy')
plt.legend(loc=0, prop={'size': 16})
plt.tight_layout()
plt.grid()
plt.savefig(save_dir + 'increasing' + img_type)
plt.clf()


#Impact of missing camera
plt.figure(figsize=(8, 6.5))
for name in names:
    plt.plot(idxs, miss_df[name]*100., styles[name], linewidth=linewidth, ms=ms,
             color=colors[name], label=legend[name])
plt.xticks(idxs)
plt.xlim(0.5, 6.5)
plt.xlabel('Missing Device Index')
plt.ylabel('Classification Accuracy')
plt.legend(loc=0, prop={'size': 16})
plt.tight_layout()
plt.grid()
plt.savefig(save_dir + 'fault' + img_type)
plt.clf()

colors = {'overall': '#84618D', 'local': '#FF8F80', 'cloud': '#98D1F1'}
styles = {'overall': '-o', 'local': '-.o', 'cloud': '--o'}
legend = {'overall': 'Overall', 'local': 'Local', 'cloud': 'Cloud'}
names = ['overall', 'cloud', 'local']

#comm cost
fig, ax1 = plt.subplots(figsize=(8, 6.5))
for name in names:
    idxs = np.argsort(comm_df['comm'])
    ax1.plot(comm_df['comm'][idxs], comm_df[name][idxs]*100., styles[name],
             linewidth=linewidth, ms=ms, color=colors[name],
             label=legend[name] + ' Acc.')
# plt.xticks(comm_df['filters'].values)
# plt.xlim(comm_df['filters'].values[0]-0.5, comm_df['filters'].values[-1]+0.5)
ax2 = ax1.twinx()
ax2.plot(comm_df['comm'][idxs], comm_df['device_size'][idxs]/1000., ':ok', linewidth=linewidth, ms=ms, label='Device Memory')

ax1.set_xlabel('Communication (B)')
ax1.set_ylabel('Classification Accuracy')
ax1.set_ylim(80, 100)
ax2.set_ylabel('End Device Memory (KB)')
ax2.set_ylim(0.6, 4)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
leg = ax1.legend(h1+h2, l1+l2, loc='lower right', prop={'size': 16})
for l in leg.legendHandles:
    l._sizes = [6]

plt.tight_layout()
plt.savefig(save_dir + 'commvsacc' + img_type)
plt.clf()

fig, ax1 = plt.subplots(figsize=(8, 6.5))
for name in names:
    ax1.plot(thres_df['device_size']/1000., thres_df[name]*100., styles[name],
             linewidth=linewidth, ms=ms, color=colors[name],
             label=legend[name] + ' Acc.')
ax2 = ax1.twinx()
ax2.plot(thres_df['device_size']/1000., thres_df['comm'], 'o--k',
         linewidth=linewidth, ms=ms, label='Communication')

ax1.set_xlabel('End Device Memory (KB)')
ax1.set_ylabel('Classification Accuracy')
ax1.set_ylim(80, 100)
ax2.set_ylabel('Communication (B)')
ax2.set_ylim(0, 120)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
leg = ax1.legend(h1+h2, l1+l2, loc='lower right', prop={'size': 16})
for l in leg.legendHandles:
    l._sizes = [6]

plt.tight_layout()
plt.savefig(save_dir + 'sizevsacc' + img_type)
plt.clf()



#entropy
fig, ax1 = plt.subplots(figsize=(8, 6.5))
name = 'overall'
ax1.plot(ent_df['T'], ent_df[name]*100., styles[name],
             linewidth=linewidth, ms=ms, color=colors[name],
             label=legend[name] + ' Acc.')
ax2 = ax1.twinx()
ax2.plot(ent_df['T'], ent_df['exit'], 'o--k',
         linewidth=linewidth, ms=ms, label='Local Exit (%)')

ax1.set_xlabel('Threshold')
ax1.set_ylabel('Classification Accuracy')
ax1.set_xlim(0.05, 1.05)
ax1.set_ylim(80, 105)
ax2.set_ylabel('Percent Locally Exited')
ax2.set_ylim(-5, 105)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
leg = ax1.legend(h1+h2, l1+l2, loc='upper left', prop={'size': 16})
for l in leg.legendHandles:
    l._sizes = [6]

# align_y_axis(ax1, ax2, 1, 1)
# ax1.grid()
# ax2.grid(None)
plt.tight_layout()
plt.savefig(save_dir + 'thresholdvsacc' + img_type)
plt.clf()

#exit thres
plt.figure(figsize=(8, 6.5))
plt.plot(ent_df['exit'], ent_df['overall']*100., '-o', color=colors['overall'],
         linewidth=linewidth, ms=ms, label='Local Exit (%)')
plt.xlim(-2, 105)
plt.xlabel('Percentage Locally Exited')
plt.ylim(90, 100)
plt.ylabel('Classification Accuracy')
# plt.legend(loc=0)
plt.tight_layout()
plt.grid()
plt.savefig(save_dir + 'exitvsacc' + img_type)
plt.clf()
