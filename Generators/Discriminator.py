import torch.nn as nn

class ResBlock(nn.Module):

    def __init__(self, input_nc, output_nc, kernel_size=13, padding=6, bias=True):
        super(ResBlock, self).__init__()
        model = [nn.ReLU(inplace=False),
                 nn.Conv1d(input_nc, output_nc, kernel_size=kernel_size, padding=padding, bias=bias),
                 nn.ReLU(inplace=False),
                 nn.Conv1d(input_nc, output_nc, kernel_size=kernel_size, padding=padding, bias=bias),
                 ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return x + 0.3 * self.model(x)
    
class EncoderLayer(nn.Module):

    def __init__(self, emb_num, num_heads):
        super(EncoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(emb_num, num_heads)

    def forward(self, x):
        x1, self.weights = self.attn(x, x, x)
        return x1

class Discriminator(nn.Module):

    def __init__(self, output_nc, ndf=512, seqL=50, bias=True, layer_num=1, num_heads=16):
        super(Discriminator, self).__init__()
        self.Conv1 = nn.Conv1d(output_nc, ndf, 1)
        self.layer_num = layer_num
        for i in range(self.layer_num):
            exec("self.layer_{} = EncoderLayer(ndf, {})".format(i, num_heads))
        model = [ResBlock(ndf, ndf),
                 ResBlock(ndf, ndf),
                 ResBlock(ndf, ndf),
                 ResBlock(ndf, ndf),
                 ResBlock(ndf, ndf), ]
        self.model = nn.Sequential(*model)
        self.last_linear = nn.Linear(seqL * ndf, 1, bias=bias)

    def forward(self, x):
        x1 = self.Conv1(x)
        x_t = x1.permute(2, 0, 1)
        for i in range(self.layer_num):
            exec("x_t = self.layer_{}(x_t)".format(i))
        x_t = x_t.permute(1, 2, 0)
        x_t = self.model(x_t)
        return self.last_linear(x_t.contiguous().view(x_t.size(0), -1))