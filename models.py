
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
    )


"""SIGMIoid for direct encoding in snn should be betrween (0,1)
maxpoll for grad-cam forward-maps is needed for this too
forward is an standard pytorch metood
embed output is what snn receieves"""


class CNN(nn.Module):

    def __init__(self, n_classes=3, feat_dim=128, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32), nn.MaxPool2d(2),     
            conv_block(32, 64), nn.MaxPool2d(2),    
            conv_block(64, 128),                    #32*32
        )
        self.pool = nn.Sequential(nn.MaxPool2d(2), nn.AdaptiveAvgPool2d(4), nn.Flatten())
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(128 * 4 * 4, feat_dim), nn.Sigmoid())
        self.classifier = nn.Linear(feat_dim, n_classes)

    def forward_maps(self, x):
        return self.features(x)

    def from_maps(self, a):
        return self.fc(self.pool(a))

    def embed(self, x):
        return self.from_maps(self.forward_maps(x))

    def forward(self, x):
        return self.classifier(self.embed(x))


class SNNHead(nn.Module):
    """DIrect encoding 128 vector as current """

    def __init__(self, in_dim=128, hidden=64, n_classes=3, T=8, beta=0.9,
                 threshold=1.0, slope=25.0):
        super().__init__()

        # lif doesnt have derivative so we use this instead
        sg = surrogate.fast_sigmoid(slope=slope)
        self.T = T
        self.fc1 = nn.Linear(in_dim, hidden)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=sg)
        self.fc2 = nn.Linear(hidden, n_classes)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=sg,
                              reset_mechanism="none")  
        
    def forward(self, x, return_stats=False):
        mem1 = self.lif1.init_leaky()   # reset membrane state on EVERY forward pass
        mem2 = self.lif2.init_leaky()
        cur1 = self.fc1(x)              # constant current (direct encoding)
        mem_sum, spk_sum = 0.0, 0.0
        for _ in range(self.T):
            spk1, mem1 = self.lif1(cur1, mem1)
            _, mem2 = self.lif2(self.fc2(spk1), mem2)
            mem_sum = mem_sum + mem2
            spk_sum = spk_sum + spk1

        #we use min   
        logits = mem_sum / self.T
        if return_stats:
            return logits, spk_sum / self.T   


def load_cnn_ckpt(path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ck = torch.load(path, map_location=device)
    model = CNN(n_classes=len(ck["classes"]))
    model.load_state_dict(ck["state"])
    return model.to(device).eval(), ck["classes"]


def load_snn_ckpt(path, in_dim, n_classes, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ck = torch.load(path, map_location=device)
    c = ck["cfg"]
    model = SNNHead(in_dim=in_dim, hidden=c["hidden"], n_classes=n_classes,
                    T=c["T"], beta=c["beta"])
    model.load_state_dict(ck["state"])
    return model.to(device).eval(), ck