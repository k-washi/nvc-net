from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, pad) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, bias=False
                )
        else:
            self.skip = nn.Sequential()
        
        self.block1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel, padding=pad)
        )
        
        self.block2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        )

    def forward(self, x):
        s = self.skip(x)
        s = F.avg_pool2d(s, (1, 2))
        
        o = self.block1(x)
        o = F.avg_pool2d(o, (1, 2))
        o = self.block2(o)
        
        return s + o
        

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, spk_emb_channels=None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spk_emb_channels = spk_emb_channels
        
        if in_channels != out_channels:
            raise NotImplementedError()
        else:
            self.skip = nn.Sequential()
        
        self.block1 = nn.Sequential(
            nn.ReflectionPad1d((dilation, dilation)),
            nn.Conv1d(in_channels, out_channels * 2, kernel_size=3, dilation=dilation)
        )
        
        if spk_emb_channels is not None:
            self.spk_emb_block = nn.Conv1d(
                spk_emb_channels, out_channels * 2, kernel_size=1
            )

        self._tanh = nn.Tanh()
        self._sigmoid = nn.Sigmoid()
        self.block2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=1, dilation=(dilation)
        )
    
    def forward(self, x, spk_emb=None):
        o = self.block1(x)
        if spk_emb is not None:
            o = o + self.spk_emb_block(spk_emb)
        o = self._tanh(o[:, :self.out_channels, ...]) * self._sigmoid(o[:, self.out_channels:, ...])
        o = self.block2(o)
        s = self.skip(x)
        return o + s
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, spk_emb_channels, r, num_resblock) -> None:
        super().__init__()
        # out_channels = mult * ngf
        self.gelu = nn.GELU()
        self.deconv1 = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=r * 2,
            stride=r, padding=(r // 2 + r % 2)
        )
        
        self.resblocks = []
        for i in range(num_resblock):
            self.resblocks.append(
                ResnetBlock(
                    out_channels, out_channels, 3**i, spk_emb_channels
                )
            )
        
    
    def forward(self, x, spk_emb):
        # rが偶数なので、入力も偶数が良い
        x = self.gelu(x)
        x = self.deconv1(x)
        for resblock in self.resblocks:
            x = resblock(x, spk_emb)
        return x

class DownBlock(nn.Module):
    def __init__(self,  in_channels, out_channels, r, num_resblock) -> None:
        super().__init__()
        
        self.resblocks = []
        for i in range(num_resblock):
            self.resblocks.append(
                ResnetBlock(
                    in_channels, in_channels, 3**i, None
                )
            )
        self.gelu = nn.GELU()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=r*2,
            stride=r, padding=(r//2+r%2)
        )
    
    def forward(self, x):
        # rが偶数なので、入力も偶数が良い
        for resblock in self.resblocks:
            x = resblock(x)
        x = self.gelu(x)
        return self.conv(x)
        

if __name__ == "__main__":
    import torch

    def test_resblock():
        x = torch.ones((2, 32, 1000))
        spk_emb = torch.ones((1, 128, 1))
        
        rb = ResnetBlock(32, 32, 3, 128)
        o = rb(x, spk_emb)
        print(o.shape)
        o = rb(x)
        print(o.shape)
        
    def test_residualblock():
        x = torch.ones((2, 1, 250, 250)) # mel spec
        x = nn.Conv2d(1, 32, kernel_size=(3,3), padding=(1,1))(x)
        rb = ResidualBlock(32, 128, (3,3), (1,1))
        o = rb(x)
        print(o.shape)
    
    def test_upblock():
        x = torch.ones((2, 32, 1000))
        db = DownBlock(32, 12, 2, 2)
        x = db(x)
        print(x.shape)
        spk_emb = torch.ones((1, 128, 1))
        ub = UpBlock(12, 32, 128, 2, 2)
        o = ub(x, spk_emb)
        print(o.shape)

    #test_resblock()
    #test_residualblock()
    test_upblock()
