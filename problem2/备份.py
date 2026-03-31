import torch
from torch import nn
from torch.nn import functional as F
device = ('cpu')
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiheadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            # mask shape: (batch_size, seq_len) or (batch_size, seq_len, seq_len)
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
            
            attn_scores = attn_scores.masked_fill_(mask == mask, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to original shape
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final linear projection
        output = self.out_proj(attn_output)
        
        if return_attention:
            return output, attn_weights
        return output


x = torch.tensor(torch.randn(16,10,16))
model = MultiheadAttention(embed_dim=16, num_heads=4,dropout = 0.1)
y = model(x,mask = None,return_attention=False)
print(x.shape)
print(y.shape)

class MultiheadAttention_cavhe(nn.Module):
    def __init__(self,d_model,num_heads):
        super(MultiheadAttention_cavhe,self).__init__()
        assert d_model%num_heads==0
        self.d_model = d_model
        self.num_heads = num_heads
        self.scale = d_model // num_heads
        self.q = nn.Linear(d_model,d_model)
        self.k = nn.Linear(d_model,d_model)
        self.v = nn.Linear(d_model,d_model)
        self.out_proj = nn.Linear(d_model,d_model)
        self.k_cache = None
        self.v_cache = None

    def scaled_dot_attn_causal_mask(self,q,k,v,mask = None):
        attn = torch.matmul(q,k.transpose(-2,-1))/torch.sqrt(torch.tensor(self.scale,dtype = torch.float32))
        Tq,Tk = q.size(2),k.size(2)
        if Tq == Tk:
            causal_mask = torch.tril(torch.ones(Tq,Tq,dtype = torch.bool))
            attn = attn.masked_fill(~causal_mask, -1e9)
        if mask is not None:
            attn = attn.masked_fill(~mask, -1e9)
        
        attn = F.softmax(attn,dim=-1)
        output = torch.matmul(attn,v)
        return output
    
    def reset(self):
        self.k_cache = None
        self.v_cache = None

    def forward(self,x,use_cache = True,mask = None):
        batch_size,seq_len,_ = x.shape
        q,k,v = self.q(x),self.k(x),self.v(x)
        q = q.view(batch_size,seq_len,self.num_heads,self.scale).transpose(1,2)
        k = k.view(batch_size,seq_len,self.num_heads,self.scale).transpose(1,2)
        v = v.view(batch_size,seq_len,self.num_heads,self.scale).transpose(1,2)

        if use_cache:
            if self.k_cache is not None:
                k = torch.cat([self.k_cache,k],dim = 2)
                v = torch.cat([self.v_cache,v],dim = 2)
            self.k_cache = k
            self.v_cache = v
        
        attn = self.scaled_dot_attn_causal_mask(q,k,v,mask = mask)
        attn = attn.transpose(1,2).contiguous().view(batch_size,seq_len,self.d_model)
        attn = self.out_proj(attn)
        return attn
class GQA(nn.Module):
    def __init__(self,d_model,num_heads,num_kv_heads):
        super(GQA,self).__init__()
        assert d_model%num_heads==0
        assert d_model%num_kv_heads==0
        assert num_heads % num_kv_heads == 0
        self.d_model=d_model
        self.num_heads=num_heads
        self.scale = d_model // num_heads
        self.num_kv_heads=num_kv_heads
        self.q=nn.Linear(d_model,d_model)
        self.v = nn.Linear(d_model,num_kv_heads*self.scale)
        self.k = nn.Linear(d_model,num_kv_heads*self.scale)
        self.num_q_kv_heads = num_heads // num_kv_heads
        self.out_proj = nn.Linear(d_model,d_model)
        self.k_cache = None
        self.v_cache = None

    def reset(self):
        self.k_cache = None
        self.v_cache = None

    def forward(self,x,use_cache = True,mask = None):
        batch_size,seq_len,_ = x.shape
        q = self.q(x).view(batch_size,seq_len,self.num_heads,self.scale).transpose(1,2)
        v = self.v(x).view(batch_size,seq_len,self.num_kv_heads,self.scale).transpose(1,2)
        k = self.k(x).view(batch_size,seq_len,self.num_kv_heads,self.scale).transpose(1,2)

        if use_cache:
            if self.k_cache is not None:
                k = torch.cat([self.k_cache,k],dim = 2)
                v = torch.cat([self.v_cache,v],dim = 2)
            self.k_cache = k
            self.v_cache = v
        k = k.repeat_interleave(self.num_q_kv_heads,dim = 1)
        v = v.repeat_interleave(self.num_q_kv_heads,dim = 1)
        attn = torch.matmul(q,k.transpose(-2,-1))/torch.sqrt(torch.tensor(self.scale,dtype = torch.float32))
        Tq,Tk = q.size(2),k.size(2)
        if Tq == Tk:
            causal_mask = torch.tril(torch.ones(Tq,Tk,dtype = torch.bool))
            attn.masked_fill_(~causal_mask,-1e9)
        if mask is not None:
            attn.masked_fill_(~mask,-1e9)

        attn = F.softmax(attn,dim = -1)
        attn = torch.matmul(attn,v)
        attn = attn.transpose(1,2).contiguous().view(batch_size,seq_len,self.d_model)
        attn = self.out_proj(attn)

        return attn

x = torch.randn(3,9,9).to(device)
model = MultiheadAttention_cavhe(d_model = 9,num_heads = 3).to(device)

y = model(x,use_cache=False,mask = None)
print(y.shape)

model = GQA(9,9,3)
y = model(x)
print(y.shape)

  
