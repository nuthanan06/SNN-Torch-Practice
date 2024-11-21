import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def plot_spk_cur_mem_spk(spike, synapticcurrent, membrane, output):
    fig, axs = plt.subplots(2, 2, figsize=(10, 5))
    splt.raster(spike, axs[0,0], s=100, c="black", marker="|")
    axs[0, 0].set_title("Input Spikes")
    axs[0, 0].set_xlabel("Time step")
    axs[0, 0].set_ylabel("Neuron Number")
    
    axs[0, 1].plot(torch.arange(len(synapticcurrent)).detach().numpy(), synapticcurrent.detach().numpy())
    axs[0, 1].set_title("Synaptic Current")
    axs[0, 1].set_xlabel("Time Step")
    axs[0, 1].set_ylabel("Synaptic Current")
    
    axs[1, 0].plot(torch.arange(len(membrane)).detach().numpy(), membrane.detach().numpy())
    axs[1, 0].set_title("Output Spikes")
    axs[1, 0].set_xlabel("Time Steps")
    axs[1, 0].set_ylabel("Neuron Number")
    
    splt.raster(output, axs[1,1], s=100, c="black", marker="|")
    axs[1, 1].set_title("Output Spikes")
    axs[1, 1].set_xlabel("Time Steps")
    axs[1, 1].set_ylabel("Neuron Number")

    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    plt.show()




# Temporal dynamics
alpha = 0.9
beta = 0.8
num_steps = 200

# Initialize 2nd-order LIF neuron
lif1 = snn.Synaptic(alpha=alpha, beta=beta)

# Periodic spiking input, spk_in = 0.2 V
w = 0.2
spk_period = torch.cat((torch.ones(1)*w, torch.zeros(9)), 0)
spk_in = spk_period.repeat(20)

# Initialize hidden states and output
syn, mem = lif1.init_synaptic()
spk_out = torch.zeros(1)
syn_rec = []
mem_rec = []
spk_rec = []

# Simulate neurons
for step in range(num_steps):
  spk_out, syn, mem = lif1(spk_in[step], syn, mem)
  spk_rec.append(spk_out)
  syn_rec.append(syn)
  mem_rec.append(mem)

spk_rec = torch.stack(spk_rec)
syn_rec = torch.stack(syn_rec)
mem_rec = torch.stack(mem_rec)

plot_spk_cur_mem_spk(spk_in, syn_rec, mem_rec, spk_rec)
