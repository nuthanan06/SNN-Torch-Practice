import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def plot_snn_spikes (input, layer1, layer2): 

    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(2,2,1)
    splt.raster(input, ax, s=1.5, c="black")
    plt.title("Input Spikes")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")

    ax = fig.add_subplot(2, 2, 2)
    splt.raster(layer1, ax, s=1.5, c="black")
    plt.title("Hidden Layer")
    plt.xlabel("Time Steps")
    plt.ylabel("Neuron Number")

    ax = fig.add_subplot(2, 1, 2)
    splt.raster(layer2, ax, s=1.5, c="black")
    plt.title("Output Spikes")
    plt.xlabel("Time Steps")
    plt.ylabel("Neuron Number")

    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    plt.show()


num_inputs = 784
num_hidden = 1000
num_outputs = 10
num_steps= 200; 
beta = 0.99

fc1 = nn.Linear(num_inputs, num_hidden)
lif1 = snn.Leaky(beta=beta)
fc2 = nn.Linear(num_hidden, num_outputs)
lif2 = snn.Leaky(beta=beta)

mem1 = lif1.init_leaky()
mem2 = lif2.init_leaky()

mem2_rec = []
spk1_rec = []
spk2_rec = []
spk_in = spikegen.rate_conv(torch.rand((200, 784))).unsqueeze(1)

for step in range(num_steps):
    cur1 = fc1(spk_in[step]) # post-synaptic current <-- spk_in x weight
    spk1, mem1 = lif1(cur1, mem1) # mem[t+1] <--post-syn current + decayed membrane
    cur2 = fc2(spk1)
    spk2, mem2 = lif2(cur2, mem2)

    mem2_rec.append(mem2)
    spk1_rec.append(spk1)
    spk2_rec.append(spk2)

# convert lists to tensors
mem2_rec = torch.stack(mem2_rec)
spk1_rec = torch.stack(spk1_rec)
spk2_rec = torch.stack(spk2_rec)

spk_in = spk_in.reshape((num_steps, -1))
spk1_rec = spk1_rec.reshape((num_steps, -1))
spk2_rec = spk2_rec.reshape((num_steps, -1))


plot_snn_spikes(spk_in, spk1_rec, spk2_rec)
