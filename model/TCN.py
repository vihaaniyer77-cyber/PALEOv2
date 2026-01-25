

import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size]


class ResidualTCNBlock(nn.Module):
   
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
        groups_gn: int = 8,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation  

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.gn1 = nn.GroupNorm(num_groups=min(groups_gn, out_ch), num_channels=out_ch)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.gn2 = nn.GroupNorm(num_groups=min(groups_gn, out_ch), num_channels=out_ch)
        self.drop2 = nn.Dropout(dropout)

        self.downsample = None
        if in_ch != out_ch:
            self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.chomp1(y)
        y = self.gn1(y)
        y = F.gelu(y)
        y = self.drop1(y)

        y = self.conv2(y)
        y = self.chomp2(y)
        y = self.gn2(y)
        y = F.gelu(y)
        y = self.drop2(y)

        res = x if self.downsample is None else self.downsample(x)
        return y + res


class TCNEventDetector(nn.Module):
   
    def __init__(
        self,
        in_ch: int = 1,
        hidden: int = 64,
        depth: int = 6,
        kernel_size: int = 5,
        dropout: float = 0.1,
        groups_gn: int = 8,
        out_ch: int = 2,
    ):
        super().__init__()

        dilations = [2 ** i for i in range(depth)]  # 1,2,4,8,16,32 

        blocks = []
        ch_in = in_ch
        for d in dilations:
            blocks.append(
                ResidualTCNBlock(
                    in_ch=ch_in,
                    out_ch=hidden,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                    groups_gn=groups_gn,
                )
            )
            ch_in = hidden

        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Conv1d(hidden, out_ch, kernel_size=1)

        
        nn.init.zeros_(self.head.bias)
        nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")

    def forward(self, x):
        """
        x: (B, 1, T)
        returns logits: (B, 2, T)
        """
        h = self.backbone(x)
        logits = self.head(h)
        return logits


#Instantiate model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TCNEventDetector(
    in_ch=1,
    hidden=64,       
    depth=6,         
    kernel_size=5,   
    dropout=0.1,
    groups_gn=8,
    out_ch=2,        # [transit, flare]
).to(device)

#Diagnostics
with torch.no_grad():
    xb = torch.randn(4, 1, T, device=device)
    out = model(xb)
    print("Input:", tuple(xb.shape), "Output logits:", tuple(out.shape))
    print("Probabilities shape (sigmoid):", tuple(torch.sigmoid(out).shape))
