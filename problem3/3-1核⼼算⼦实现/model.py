import torch
from torch import nn
from torch.nn import functional as F
device = ('cuda' if torch.cuda.is_available() else 'cpu')
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim,dtype = torch.float32))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Conv1dProj(nn.Module):
    def __init__(self,dim,kernel_size):
        super(Conv1dProj,self).__init__()
        self.conv = nn.Conv1d(dim,dim,kernel_size,padding = kernel_size//2,groups = dim)
    
    def forward(self,x):
        return self.conv(x.transpose(1,2)).transpose(1,2)
    
class GDR(nn.Module):
    def __init__(self,dim = 64,hidden_dim = 1024,kernel_size = 3,
                 eps = 1e-6):
        super(GDR,self).__init__()
        self.in_proj = nn.Linear(dim,dim*4,bias = False)
        self.q_conv = Conv1dProj(dim,kernel_size)
        self.k_conv = Conv1dProj(dim,kernel_size)
        self.dropout = nn.Dropout(0.125)
        self.gate_proj = nn.Linear(dim,dim*2,bias = False)
        self.activate = nn.SiLU()
        self.out_proj = nn.Linear(dim,dim,bias = False)
        self.Norm = RMSNorm(dim,eps)
        self.residul = nn.Linear(dim,dim,bias = False)
    def forward(self,x,state = None):
        B,L,D = x.shape
        residul = self.residul(x)
        qkv = self.in_proj(x).to(device)
        q,k,v,gate = torch.split(qkv,D,dim = -1)

        q = self.q_conv(q)
        q = F.sigmoid(q)
        q = F.normalize(q,dim = -1,p = 2)

        k = self.k_conv(k)
        k = F.sigmoid(k)
        k = F.normalize(k,dim = -1,p = 2)

        gate = self.gate_proj(gate).to(device)
        alpha,beta = torch.split(gate,D,dim = -1)
        alpha = F.sigmoid(alpha)
        beta = F.sigmoid(beta)
        if state is None:
            state = torch.zeros(B,D,D,dtype = x.dtype,device = x.device)
        
        new_state = state.clone().to(device)
        outputs = []

        for t in range(L):
            q_t = q[:,t:t+1]
            v_t = v[:,t:t+1]
            k_t = k[:,t:t+1]
            alpha_t = alpha[:,t:t+1]
            beta_t = beta[:,t:t+1]
            
            k_kT = torch.bmm(k_t.transpose(-1,-2),k_t).to(device)
            update_mask = torch.eye(D,device = device).unsqueeze(0) - beta_t*k_kT
            update_mask = update_mask.to(device)
            new_state = alpha_t.to(device)*torch.bmm(new_state,update_mask).to(device)
            new_state = new_state + beta_t*torch.bmm(v_t.transpose(-1,-2),k_t).to(device)
            out_t = self.activate(torch.bmm(q_t,new_state))
            outputs.append(out_t)

        out = torch.cat(outputs,dim = 1)
        out = self.out_proj(out)
        out = self.dropout(out)
        out = self.Norm(out)
        out = out + residul
        return out,new_state
    
class Patchembed(nn.Module):
    def __init__(self,in_dim = 1,out_dim = 64,patch_size = 2,img_size = 28):
        super(Patchembed,self).__init__()
        self.proj = nn.Conv2d(in_dim,out_dim,patch_size,patch_size,bias = False)
        self.num_patches = (img_size//patch_size)**2
        self.pos = nn.Parameter(torch.randn(1,self.num_patches,out_dim))

    def forward(self,x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1,2)
        x = x + self.pos      #这里去掉self.pos就可以改成无位置编码的patchembed了
        return x
    
class MLP(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x) 
        return x
    

class GDRBlocks(nn.Module):
    def __init__(self,dim,kernel_size,num_layers,
                 dropout = 0.1):
        super(GDRBlocks,self).__init__()
        self.module = nn.ModuleList([GDR(dim = dim,kernel_size = kernel_size) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,state = None):
        for module in self.module:
            x,state = module(x,state)
            x = self.dropout(x)
        return x,state

        

class Model(nn.Module):
    def __init__(self,dim,patch_size,kernel_size,img_size,num_layers,num_classes,hidden_dim = 512):
        super(Model,self).__init__()
        self.conv = nn.Conv2d(1,1,kernel_size = 1,stride = 1,padding = 0,bias = False)
        self.patch = Patchembed(1,patch_size = patch_size,img_size = img_size)
        self.mlp = MLP(dim,hidden_dim,num_classes)
        self.layernorm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim,dim,kernel_size,padding = kernel_size//2,groups = dim)
        self.gdrblocks = GDRBlocks(dim,kernel_size,num_layers) if num_layers > 0 else nn.Sequential()
        self.layernorm2 = nn.LayerNorm(dim)
        self.conv2 = nn.Conv1d(dim,dim,kernel_size,padding = kernel_size//2,groups = dim)
        self.linear = MLP(dim,hidden_dim,num_classes)
        self.num_layers = num_layers
    
    def forward(self,x):
        x = self.conv(x)
        x = self.patch(x)
        x = self.layernorm1(x)
        x = self.conv1(x.transpose(1,2)).transpose(1,2)
        if self.num_layers > 0:
            x,state = self.gdrblocks(x)
        else:
            x = self.gdrblocks(x)
        x = self.layernorm2(x)
        x = self.conv2(x.transpose(1,2)).transpose(1,2)
        x = self.linear(x)
        x = x.mean(dim = 1)
        return x
    
def build_model(config):
    model = Model(
        dim = config['dim'],
        patch_size = config['patch_size'],
        kernel_size = config['kernel_size'],
        img_size = config['img_size'],  
        num_layers=config['num_layers'],
        num_classes=config['num_classes']
    ).to(config['device'])
    return model
