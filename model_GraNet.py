import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from torchsummary import summary

from config import cfg

"""
GraNet I - Encoder-Decoder structure Unet shape with LSTM layers for
granulation pattern classification

Autor: Saida Diaz
Institution: KIS & IAC

References: Unet model derived from https://github.com/milesial/Pytorch-UNet and 
https://github.com/amirhosseinh77/UNet-AerialSegmentation
https://github.com/EBroock/FarNet-II/blob/main/FarNet_II.py

"""

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            layers.append(nn.Dropout())

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

# ConvLSTM 
class ConvLSTM(nn.Module):

    def __init__(self, input_channel, num_filter, b_h_w, kernel_size=3, stride=1, padding=1, device = cfg.device):
        super().__init__()
        self.device = device
        self._conv = nn.Conv2d(in_channels=input_channel + num_filter,
                               out_channels=num_filter*4,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)

        self._batch_size, self._state_height, self._state_width = b_h_w
        self.Wci = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width),requires_grad=True).to(cfg.device)
        self.Wcf = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width),requires_grad=True).to(cfg.device)
        self.Wco = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width),requires_grad=True).to(cfg.device)
        self._input_channel = input_channel
        self._num_filter = num_filter

    def forward(self, inputs=None, states=None, seq_len=(2*cfg.seq_len)+1):

        if states is None:
            c = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                                  self._state_width), dtype=torch.float).to(cfg.device)
            h = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                             self._state_width), dtype=torch.float).to(cfg.device)

        else:
            h, c = states

        outputs = []

        for index in range(seq_len):

            if inputs is None:

                x = torch.zeros((h.size(0), self._input_channel, self._state_height,
                                      self._state_width), dtype=torch.float).to(cfg.device)
            
            else:
                x = inputs[index, ...]

            cat_x = torch.cat([x, h], dim=1)
            conv_x = self._conv(cat_x)
            i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)
            i = torch.sigmoid(i+self.Wci*c)
            f = torch.sigmoid(f+self.Wcf*c)
            c = f*c + i*torch.tanh(tmp_c)
            o = torch.sigmoid(o+self.Wco*c)
            h = o*torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs), (h, c)

class inconv(nn.Module):
    "Initial convolution and return to the sequencial dimension loossed during bi-directional LSTM concatenation"

    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x):
        x = self.conv(x)
        return x 

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear, dropout=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels , out_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW (channel, height, width)
        # Adjusting borders
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])


        x = torch.cat((x2, x1), dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class AttentionBlock(nn.Module):

    def __init__(self, in_channels_x, in_channels_g, int_channels):

        super(AttentionBlock, self).__init__()

        self.Wx = nn.Sequential(nn.Conv2d(in_channels_x, int_channels, kernel_size = 1),
                                nn.BatchNorm2d(int_channels))
        self.Wg = nn.Sequential(nn.Conv2d(in_channels_g, int_channels, kernel_size = 1),
                                nn.BatchNorm2d(int_channels))
        self.psi = nn.Sequential(nn.Conv2d(int_channels, 1, kernel_size = 1),
                                 nn.BatchNorm2d(1),
                                 nn.Sigmoid())
    
    def forward(self, x, g):

        x1 = self.Wx(x)
        g1 = nn.functional.interpolate(self.Wg(g), x1.shape[2:], mode = 'bilinear', align_corners = False)
        out = self.psi(nn.ReLU()(x1 + g1))
        out = nn.Sigmoid()(out)
        return out*x

class GraNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_seq=(2*cfg.seq_len)+1, n_hidden=cfg.n_hidden,
                h=cfg.h, w=cfg.w, batch=cfg.batch, bilinear=False, dropout = False):
        super(GraNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.batch = batch
        self.n_seq = n_seq
        self.dropout = dropout

        self.inc = DoubleConv(n_channels, n_hidden, dropout=self.dropout)
        self.down1 = Down(n_hidden, 2*n_hidden, dropout=self.dropout)
        self.down2 = Down(2*n_hidden, 4*n_hidden, dropout=self.dropout)
        self.down3 = Down(4*n_hidden, 8*n_hidden, dropout=self.dropout)
        
        self.LSTM1 = ConvLSTM(input_channel=8*n_hidden, num_filter=8*n_hidden, b_h_w=(self.batch, h // 8, w // 8))
        self.LSTM1_inv = ConvLSTM(input_channel=8*n_hidden, num_filter=8*n_hidden, b_h_w=(self.batch, h // 8, w // 8))
        self.conv1 = inconv(self.n_seq*2,self.n_seq, dropout=self.dropout)
        self.att1 = AttentionBlock(4*n_hidden,8*n_hidden,int(4*n_hidden))
        self.up1 = Up(8*n_hidden,4*n_hidden, bilinear=bilinear, dropout=self.dropout)
    
        self.LSTM2 = ConvLSTM(input_channel=4*n_hidden, num_filter=4*n_hidden, b_h_w=(self.batch, h // 4, w // 4))
        self.LSTM2_inv = ConvLSTM(input_channel=4*n_hidden, num_filter=4*n_hidden, b_h_w=(self.batch, h // 4, w // 4))
        self.conv2 = inconv(self.n_seq*2,self.n_seq, dropout=self.dropout)
        self.att2 = AttentionBlock(2*n_hidden,4*n_hidden,int(2*n_hidden))
        self.up2 = Up(4*n_hidden,2*n_hidden, bilinear=bilinear, dropout=self.dropout)

        self.LSTM3 = ConvLSTM(input_channel=2*n_hidden, num_filter=2*n_hidden, b_h_w=(self.batch, h // 2, w // 2))
        self.LSTM3_inv = ConvLSTM(input_channel=2*n_hidden, num_filter=2*n_hidden, b_h_w=(self.batch, h // 2, w // 2))
        self.conv3 = inconv(self.n_seq*2,self.n_seq, dropout=self.dropout)
        self.att3 = AttentionBlock(n_hidden,2*n_hidden,int(n_hidden))
        self.up3 = Up(2*n_hidden,n_hidden, bilinear=bilinear, dropout=self.dropout)

        self.outc = OutConv(n_hidden, n_classes)

    def forward(self, x):
        #x = x[:,:,np.newaxis,:,:]
        #x = rearrange(x,'B C S H W -> B S C H W')
        x = rearrange(x,'B S C H W -> (B S) C H W')
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x4 = rearrange(x4,'(B S) C H W -> B S C H W', B=self.batch)
        x4 = rearrange(x4,'B S C H W -> S B C H W')
        x4LSTM = self.LSTM1(inputs=x4, states=None)[0]
        x4_inv = torch.flip(x4,[0])
        x4_inv_LSTM = self.LSTM1_inv(inputs=x4_inv, states=None)[0]
        x4_inv_LSTM = torch.flip(x4_inv_LSTM,[0])
        x4_LSTM_glob = torch.cat((x4LSTM,x4_inv_LSTM))
        x4_LSTM_glob = rearrange(x4_LSTM_glob,'S B C H W -> B C S H W')
        x4_LSTM_glob = rearrange(x4_LSTM_glob,'B C S H W -> (B C) S H W')
        x4LSTM = self.conv1(x4_LSTM_glob)
        x4LSTM = rearrange(x4LSTM,'(B C) S H W -> B C S H W', B=self.batch)
        x4LSTM = rearrange(x4LSTM,'B C S H W -> B S C H W')
        x4LSTM = rearrange(x4LSTM,'B S C H W -> (B S) C H W')
        att1x4 = self.att1(x3,x4LSTM)
        upx4 = self.up1(x4LSTM,att1x4)

        upx4 = rearrange(upx4,'(B S) C H W -> B S C H W', B=self.batch)
        upx4 = rearrange(upx4,'B S C H W -> S B C H W')
        x3LSTM = self.LSTM2(inputs=upx4, states=None)[0]
        upx4_inv = torch.flip(upx4,[0])
        upx4_inv_LSTM = self.LSTM2_inv(inputs=upx4_inv, states=None)[0]
        upx4_inv_LSTM = torch.flip(upx4_inv_LSTM,[0])
        x3_LSTM_glob = torch.cat((x3LSTM,upx4_inv_LSTM))      
        x3_LSTM_glob = rearrange(x3_LSTM_glob,'S B C H W -> B C S H W')
        x3_LSTM_glob = rearrange(x3_LSTM_glob,'B C S H W -> (B C) S H W')
        x3LSTM = self.conv2(x3_LSTM_glob)
        x3LSTM = rearrange(x3LSTM,'(B C) S H W -> B C S H W',B=self.batch)
        x3LSTM = rearrange(x3LSTM,'B C S H W -> B S C H W',B=self.batch)
        x3LSTM = rearrange(x3LSTM,'B S C H W -> (B S) C H W')
        att2x3 = self.att2(x2,x3LSTM)
        upx3 = self.up2(x3LSTM,att2x3)
        
        upx3 = rearrange(upx3,'(B S) C H W -> B S C H W',B=self.batch)
        upx3 = rearrange(upx3,'B S C H W -> S B C H W')
        x2LSTM = self.LSTM3(inputs=upx3, states=None)[0]
        upx3_inv = torch.flip(upx3,[0])
        upx3_inv_LSTM = self.LSTM3_inv(inputs=upx3_inv, states=None)[0]
        upx3_inv_LSTM = torch.flip(upx3_inv_LSTM,[0])
        x2_LSTM_glob = torch.cat((x2LSTM,upx3_inv_LSTM))
        x2_LSTM_glob = rearrange(x2_LSTM_glob,'S B C H W -> B C S H W')
        x2_LSTM_glob = rearrange(x2_LSTM_glob,'B C S H W -> (B C) S H W')
        x2LSTM = self.conv3(x2_LSTM_glob)
        x2LSTM = rearrange(x2LSTM,'(B C) S H W -> B C S H W',B=self.batch)
        x2LSTM = rearrange(x2LSTM,'B C S H W -> B S C H W')
        x2LSTM = rearrange(x2LSTM,'B S C H W -> (B S) C H W')
        att3x2 = self.att3(x1,x2LSTM)
        upx2 = self.up3(x2LSTM,att3x2)
    
        x = self.outc(upx2)

        x = rearrange(x,'(B S) C H W -> B S C H W', B=self.batch)

        logits = x[:,self.n_seq//2,:,:,:]

        return logits
