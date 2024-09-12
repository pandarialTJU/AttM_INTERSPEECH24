'''
This is the ECAPA-TDNN model.
This model is modified and combined based on the following three projects:
  1. https://github.com/clovaai/voxceleb_trainer/issues/86
  2. https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py
  3. https://github.com/speechbrain/speechbrain/blob/96077e9a1afff89d3f5ff47cab4bca0202770e4f/speechbrain/lobes/models/ECAPA_TDNN.py

'''

import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
import ipdb
from torchaudio.models.wav2vec2.utils import import_fairseq_model

def str_to_bool(s):
    if s == "True":
        return True
    elif s == "False":
        return False

class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

class FbankAug(nn.Module):

    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):    
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x


class ECAPA(nn.Module):

    def __init__(self, d_args):
        super(ECAPA, self).__init__()    

        self.d_args = d_args
        self.out_dim = d_args["SSL_dim"]
        C = 512

        self.LL = nn.Linear(self.out_dim, C)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)
        self.outlayer = nn.Linear(192,2)

    def forward(self, x): # no need aug (previously in the ECAPA_TDNN code)


        x = x.squeeze()
        x = self.LL(x)
        x = x.permute(0, 2, 1)

        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        # output 192 dim when using AAM loss 
        x = self.outlayer(x)

        return x
    


class AttM(nn.Module):

    # input x is the multiple outputs from each transform layer of SSL models
    # 12 or 24 layers from WavLM, HuBert, Wav2Vec2

    def __init__(self, d_args) -> None:
        super(AttM, self).__init__()

        self.d_args = d_args
        self.n_feat = self.d_args["SSL_dim"]
        self.n_layer = self.d_args["SSL_layer_num"]

        self.W = nn.Parameter(torch.randn(self.n_feat, 1))
        self.W1 = nn.Parameter(torch.randn(self.n_layer, int(self.n_layer/2)))
        self.W2 = nn.Parameter(torch.randn(int(self.n_layer/2), self.n_layer))
        self.hidden = int(self.n_layer*self.n_feat/4)
        self.linear_proj = nn.Sequential(
            nn.Linear(self.n_layer*self.n_feat, self.hidden),
            nn.SiLU(nn.Linear(self.hidden, self.hidden)),
            nn.Linear(self.hidden, self.n_feat),
        )
        self.SWISH = nn.SiLU()

    
    def forward(self, x):

        # X1 has dimension of Bs x Layers x Time x Hidden feature

        x_input = x
   
        x = torch.mean(x, dim=2, keepdim=True) # X2 = AVG(X1) AVG across time dim

        x = self.SWISH(torch.matmul(x, self.W)) # X3

        x = self.SWISH(torch.matmul(x.view(-1, self.n_layer), self.W1))
        x = torch.sigmoid((torch.matmul(x, self.W2))) # X4
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.mul(x, x_input) # X5

        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), x.size(2), -1) # concatenate

        x = self.linear_proj(x)

        return x



class LinM(nn.Module):

    # positve weight sum to SSL model


    def __init__(self, d_args) -> None:
        super(LinM, self).__init__()

        self.d_args = d_args
        self.layer_num = d_args["SSL_layer_num"]
        self.W = nn.Parameter(torch.zeros(self.layer_num, 1))
        

    def forward(self, x):
        # x is the 24-layer maps: [bs, 24, T, H]

        x = torch.matmul(x.permute(0,2,3,1), F.softmax(self.W, dim=0))

        return x.squeeze()




class my_LSTM(nn.Module):
    def __init__(self, d_args):
        super(my_LSTM, self).__init__()

        self.d_args = d_args
        self.input_dim = d_args["SSL_dim"] # 1024 or 768
        self.hidden_dim = 192
        
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.LL = nn.Linear(self.hidden_dim, 2)

    def forward(self, x):

        self.lstm.flatten_parameters()

        x, (h_n, c_n) = self.lstm(x)
        x = torch.mean(x, dim=1)
        x = self.LL(x)
        return x


class Model(nn.Module): 

    def __init__(self, speech_model, d_args):
        super(Model, self).__init__()

        self.d_args = d_args
        Finetune_id = str_to_bool(d_args["Finetune_id"])
        self.out_dim = d_args["SSL_dim"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
   
    
        self.ssl_model = speech_model
        self.ssl_model.feature_extractor.requires_grad_(False) # 7 conv layers are frozen
        self.ssl_model.requires_grad_(Finetune_id)


        ## hidden embedding merge method: Attentive Mergeing (AttM) or Linear Mergining (LinM)
        # self.featfusion = LinM(d_args)
        self.feat_merge = AttM(d_args)

        ## classifier: single-layer LSTM or ECAPA-TDNN
        # self.decoder = ECAPA(d_args)
        self.decoder = my_LSTM(d_args)



    def forward(self, x):

        # get 24 layers feature maps
        x = x.squeeze(-1)
        _, x = self.ssl_model.extract_features(x, output_layer=self.ssl_model.cfg.encoder_layers, ret_layer_results=True)[0]
        x = torch.stack([i.transpose(0, 1) for i, _ in x])
        x = x.transpose(0,1) # Bs x layers x Time x hidden feat
        x = x[:,1:,:,:] # we don't want the CNN output


        # merge the 24 maps
        x = self.feat_merge(x)

        # decoder
        x = self.decoder(x)

        return x
    


