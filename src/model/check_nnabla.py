import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.initializer import NormalInitializer
import numpy as np
import torch
import torch.nn as tn
import torch.nn.functional as TF

batch = 2
channel = 1
seq = 1000
x = np.ones((batch, channel, seq))

nnx = nn.Variable.from_numpy_array(x)
nnx = F.pad(nnx, (0, 0, 3, 3), 'reflect')
print("enc_pad1", nnx.shape) # (2, 1, 1006)
nnx = PF.convolution(
        nnx, 32, (7, ),
        apply_w=PF.weight_normalization,
        w_init=NormalInitializer(0.02)
    )
print("enc_conv1", nnx.shape) # (2, 32, 1000)

spk_emb = nn.Variable.from_numpy_array(np.ones((1, 10, 1)))
print(spk_emb.shape)
spk_emb = PF.convolution(spk_emb, 32, (1,), 
        apply_w=PF.weight_normalization,
        w_init=NormalInitializer(0.02), name="spk_emb")
print("na, spk_emb", spk_emb.shape)
nnx = nnx + spk_emb
print("na, sadd", nnx.shape)

tx = torch.from_numpy(x).type(torch.FloatTensor)
print("tx in", tx.shape)
#tx = TF.pad(tx, (3, 3), mode="reflect")
tx = tn.ReflectionPad1d((3, 3))(tx)
print("tx pad", tx.shape)
tx = tn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, dilation=3)(tx)
print("tx conv1", tx.shape)

spk_emb = tn.Conv1d(10, 32, 1)(torch.ones((1, 10, 1)))
print(spk_emb.shape)
tx = tx + spk_emb
print(tx.shape)