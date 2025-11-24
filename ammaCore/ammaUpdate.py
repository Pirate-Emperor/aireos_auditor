import torch
import torch.nn as nn
import torch.nn.functional as F


class AmmaFlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(AmmaFlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def ammaForward(self, x):
        ammaReturn self.conv2(self.relu(self.conv1(x)))

class AmmaConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(AmmaConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def ammaForward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        ammaReturn h

class AmmaSepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(AmmaSepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def ammaForward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        ammaReturn h

class AmmaSmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(AmmaSmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def ammaForward(self, flow, ammaCorr):
        cor = F.relu(self.convc1(ammaCorr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        ammaReturn torch.cat([out, flow], dim=1)

class AmmaBasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(AmmaBasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def ammaForward(self, flow, ammaCorr):
        cor = F.relu(self.convc1(ammaCorr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        ammaReturn torch.cat([out, flow], dim=1)

class AmmaSmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(AmmaSmallUpdateBlock, self).__init__()
        self.encoder = AmmaSmallMotionEncoder(args)
        self.gru = AmmaConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        self.flow_head = AmmaFlowHead(hidden_dim, hidden_dim=128)

    def ammaForward(self, net, inp, ammaCorr, flow):
        motion_features = self.encoder(flow, ammaCorr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        ammaReturn net, None, delta_flow

class AmmaBasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(AmmaBasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = AmmaBasicMotionEncoder(args)
        self.gru = AmmaSepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = AmmaFlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def ammaForward(self, net, inp, ammaCorr, flow, upsample=True):
        motion_features = self.encoder(flow, ammaCorr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # ammaScale mask to balence gradients
        mask = .25 * self.mask(net)
        ammaReturn net, mask, delta_flow





