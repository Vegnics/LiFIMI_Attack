import torch
import torch.nn as nn
import torch.nn.functional as F 


##  Used for the Score-Matching statistic net
class ScoreLayer(nn.Module): 
    def __init__(self, dim_x, dim_y, dim_hidden, n_layer=1):
        super().__init__()        
        self.fc1 = nn.Linear(dim_x, dim_hidden)
        self.fc2 = nn.Linear(dim_y, dim_hidden)
        self.merge = nn.Linear(2*dim_hidden, dim_hidden)
        self.main = nn.Sequential(
            *(nn.Linear(dim_hidden, dim_hidden, bias=True) for i in range(n_layer)),
            #nn.Dropout(p=0.2)
        )
        self.out = nn.Linear(dim_hidden, 1)
        
    def forward(self, x, y):
        #h = self.fc1(x) + self.fc2(y)
        h = torch.cat([self.fc1(x), self.fc2(y)], dim=1)
        h = F.leaky_relu(self.merge(h), 0.2)
        for layer in self.main:
            h = F.leaky_relu(layer(h), 0.2)
        out = self.out(h)
        return out

####
class CriticLayer(nn.Module): 
    def __init__(self, architecture, dim_y, hyperparams=None):
        super().__init__()       
        dim_x, dim_y, dim_hidden = 5*architecture[-1], dim_y, 100#256#200
        # WD case; need to do spectral normalization
        self.norm_in = nn.LayerNorm(dim_x + dim_y)
        if hyperparams.estimator == 'WD':
            self.main = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(dim_x + dim_y, dim_hidden), n_power_iterations=5),
            )
            self.out = nn.utils.spectral_norm(nn.Linear(dim_hidden, 1), n_power_iterations=5)
        # Other cases; need to do noting
        else:
            """
            self.main = nn.Sequential(
                nn.Linear(dim_x + dim_y, dim_hidden//2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.15),
                nn.Linear(dim_hidden//2, dim_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(inplace=True),
                #nn.Dropout(0.2),
                nn.Linear(dim_hidden, dim_hidden//2),
                nn.ReLU(inplace=True),
            )
            """
            self.main = nn.Sequential(
                nn.Linear(dim_x + dim_y, dim_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(dim_hidden, dim_hidden//2),
                nn.ReLU(inplace=True)
            )
            self.out = nn.Linear(dim_hidden//2, 1)
   
    def forward(self, x, y):
        h = torch.cat((x,y), dim=1)
        h = self.norm_in(h)
        h = self.main(h) 
        #h = torch.tanh(h)
        out = self.out(h)
        return out 
    

class ResBlockUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        # Input simple rescaling
        self.upscale = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Output upsample via deconvolution
        self.upsample = nn.ConvTranspose2d(out_ch,out_ch,kernel_size=4,stride=2,padding=1) #nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        shortcut = self.shortcut(self.upscale(x))
        #x = self.upsample(x)
        x = self.conv1(F.relu(self.bn1(x)))
        x = self.conv2(F.relu(self.bn2(x)))
        x = self.upsample(F.relu(self.bn3(x)))
        #return F.relu(x + shortcut)
        return x + shortcut  
        
class EncodeLayer(nn.Module):
    def __init__(self, architecture, dim_y, hyperparams=None):
        super().__init__()
        self.type = 'plain' if not hasattr(hyperparams, 'type') or hyperparams is None else hyperparams.type 
        self.dropout = False if not hasattr(hyperparams, 'dropout') or hyperparams is None else hyperparams.dropout 
        self.main = nn.Sequential( 
           *(nn.Linear(architecture[i+1], architecture[i+2], bias=True) for i in range(len(architecture)-3)),
        )  ### <<< Check this
        print(f"Out front end: {architecture[1]}")
        if self.type == 'plain':
            self.plain = nn.Linear(architecture[0], architecture[1], bias=True)
        if self.type == 'iid':
            self.enn = nn.Sequential(
                 nn.Conv1d(in_channels=1, out_channels=50, kernel_size=1, stride=1),
                 nn.ReLU(),
                 nn.Conv1d(in_channels=50, out_channels=architecture[1], kernel_size=1, stride=1),
            )
        if self.type == 'cnn1d':
            self.cnn = nn.Sequential(
                 nn.Conv1d(in_channels=1, out_channels=50, kernel_size=3, stride=1),
                 nn.ReLU(),
                 nn.Conv1d(in_channels=50, out_channels=architecture[1], kernel_size=3, stride=1),
            )
        # Change this summary statistics, it is too shallow
        if self.type == 'cnn2d':
            self.cnn2d = nn.Sequential(
                 nn.Conv2d(in_channels=1, out_channels=50, kernel_size=2, stride=1),
                 nn.ReLU(),
                 nn.Flatten(),
                 nn.Linear(50*(int(architecture[0]**0.5)-1)**2, architecture[1])# d = D - (K-1)L
            )
        """
        if self.type == 'cnnimg':
            ## Assuming the inputs are rgb images (256 -> 8)
            self.cnnimg = nn.Sequential(
                 nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1,padding=1),
                 nn.ReLU(),
                 nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5, stride=2,padding=2),
                 nn.ReLU(),
                 nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, stride=2,padding=2),
                 nn.ReLU(),
                 nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, stride=2,padding=2),
                 nn.ReLU(),
                 nn.MaxPool2d(kernel_size=(4,4),stride=4),
                 nn.Flatten(),
                 nn.Linear(4*8*8, architecture[1])# d = D - (K-1)L
            )
        
        self.cnnimg = nn.Sequential(
                 nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1,padding=1),
                 nn.ReLU(),
                 nn.BatchNorm2d(32),
                 nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2,padding=2),
                 nn.ReLU(),
                 nn.BatchNorm2d(32),
                 nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=2,padding=2),
                 nn.ReLU(),
                 nn.BatchNorm2d(16),
                 nn.Conv2d(in_channels=16, out_channels=4, kernel_size=5, stride=2,padding=2),
                 nn.ReLU(),
                 nn.MaxPool2d(kernel_size=(4,4),stride=4),
                 nn.Flatten(),
                 nn.Linear(4*8*8, architecture[1])# d = D - (K-1)L
            )
        """
        if self.type == 'cnnimg':
            self.cnnimg = nn.Sequential(
                # 256x256x3 -> 256x256x32
                # Depthwise: 3 -> 3, then Pointwise: 3 -> 32
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, groups=3),
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(32),

                # 256x256x32 -> 128x128x32
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2, groups=32),  # depthwise
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),            # pointwise
                nn.ReLU(),
                nn.BatchNorm2d(32),

                # 128x128x32 -> 64x64x16
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2, groups=32),  # depthwise
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0),            # pointwise
                nn.ReLU(),
                nn.BatchNorm2d(16),

                # 64x64x16 -> 32x32x4
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2, padding=2, groups=16),  # depthwise
                nn.Conv2d(in_channels=16, out_channels=4, kernel_size=1, stride=1, padding=0),             # pointwise
                nn.ReLU(),

                # 32x32x4 -> 8x8x4
                nn.MaxPool2d(kernel_size=(4, 4), stride=4),

                nn.Flatten(),
                nn.Linear(4 * 8 * 8, architecture[1])  # same as before
            )
            
        self.drop = nn.Dropout(p=0.20)
        if self.type == 'cnnimg':
            self.out = nn.Sequential(
                nn.Linear(architecture[-2], 5*architecture[-1], bias=True),
            )
        else:
            self.out = nn.Sequential(
                nn.Linear(architecture[-2], 2*architecture[-1], bias=True),
            )
        self.N_layers = len(architecture) - 1
        self.architecture = architecture
            
    def front_end(self, x):
        # i.i.d data
        if self.type == 'iid':
            n, d = x.size()
            x = x.view(n, 1, d)  # n*1*d
            x = self.enn(x)      # n*k*d
            x = x.sum(dim=2)     # n*k
        # Time-series data
        if self.type == 'cnn1d':
            n, d = x.size()
            x = x.view(n, 1, d)  # n*1*d
            x = self.cnn(x)      # n*k*d'
            x = x.sum(dim=2)     # n*k
        # Image data
        if self.type == 'cnn2d':
            n, d = x.size()
            x = x.view(n, 1, int(d**0.5), int(d**0.5)) ## The Image is reshaped to their original version
            x = self.cnn2d(x)    # n*k
        # default
        if self.type == 'plain':
            x = self.plain(x)
        if self.type == 'cnnimg':
            ## Convert to channels first
            imgs = torch.transpose(x,-1,1).clone()
            means = torch.mean(imgs,dim=[2,3],keepdim=True)
            stds = torch.std(imgs,dim=[2,3],keepdim=True)
            imgs = (imgs-means)/(stds+1e-6)
            #print(f"Front end: {x.size()} -> {imgs.size()}")
            x = self.cnnimg(imgs) 
        return x
        
    def forward(self, x):
        x = self.front_end(x)
        for layer in self.main: x = F.relu(layer(x))
        x = self.drop(x) if self.dropout else x
        x = self.out(x)
        return x
    