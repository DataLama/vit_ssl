import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
  """MLP Block"""
  def __init__(self, dim, hidden_dim, dropout = 0.):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout)
    )
  def forward(self, x):
    return self.net(x)

class Attention(nn.Module):
  """MHA Block"""
  def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
      super().__init__()
      """ 모든 attention head의 dimension size. input embedding에서 projection되는 차원."""
      inner_dim = dim_head * heads 
      project_out = not (heads == 1 and dim_head == dim)

      """ num_heads, scaled, attention function, dropout """
      self.heads = heads # attention head의 갯수
      self.scale = dim_head ** -0.5 # self attention에서 scaled 되는 값.

      self.attend = nn.Softmax(dim = -1) # attnetion score 구하는 함수.
      self.dropout = nn.Dropout(dropout)

      """ (N, seq_len, dim) -> (N, seq_len, inner_dim * 3)"""
      self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

      """ Linear + drop out"""
      self.to_out = nn.Sequential(
          nn.Linear(inner_dim, dim),
          nn.Dropout(dropout)
      ) if project_out else nn.Identity()

  def forward(self, x):
      """Get Query, Key, Value"""
      qkv = self.to_qkv(x).chunk(3, dim = -1) 

      """list of tensor에 각 tensor를 순회하면서, multi-attention head와 head별 dim으로 reshape함."""
      q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) # n == seq_len, h == num_heads, d == head_dims

      """(b h n d) * (b h d n) -> (b h n n); """
      dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # scaled dot product.

      attn = self.attend(dots) # attention map
      attn = self.dropout(attn) # attention score에 대한 dropout

      out = torch.matmul(attn, v) # value에 attention score로 reweighting을 해줌.
      """multi attention을 reshape을 통해 풀어줌."""
      out = rearrange(out, 'b h n d -> b n (h d)')
      return self.to_out(out) # linear, dropout...


class Transformer(nn.Module):
  """ViT Transformer Block
  ViT Transformer Block에서 눈여겨 볼 것.
  * Layer Norm의 위치.
    * 초기에 제안된 transformer encdoer를 학습한느 모델의 LayerNorm은 residual connection 이후에 LayerNorm을 수행함.
    * 다음 논문에서 (https://aclanthology.org/P19-1176.pdf) LayerNorm을 sub-layer를 통과하기 이전에 사용하는 것이 더 좋음을 보여줌. (학습이 안정적, 더 깊게 가능함, 성능이 좀 더 좋아짐.)
    * ViT에서도 PreNorm을 적용함.
  """
  def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
      super().__init__()
      self.layers = nn.ModuleList([])
      for _ in range(depth):
          self.layers.append(nn.ModuleList([
              PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
              PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
          ]))
  def forward(self, x):
      for attn, ff in self.layers:
          x = attn(x) + x
          x = ff(x) + x
      return x

class ViT(nn.Module):
  def __init__(
      self, 
      *, 
      image_size: int, 
      patch_size: int, 
      num_classes: int, 
      dim: int, 
      depth: int, 
      heads: int, 
      mlp_dim: int, 
      channels:int = 3, 
      dim_head = 64, 
      dropout: float  = 0., 
      emb_dropout: float = 0.,
      pool:str = 'cls'
  ):
      """Vision Transformers
      Args: 
        image_size(int) 
          * If you have rectangular images, make sure your image size is the maximum of the width and height.
        patch_size(int): 
          * `image_size` must be divisible by patch_size. 
          * The number of patches is: n = (image_size // patch_size) ** 2 and n must be greater than 16.
        num_classes(int):
          * Number of classes to classify.
        dim(int):
          * Dimension of Embeddings.
        depth(int):
          * Number of Transformer blocks
        heads(int):
          * Number of heads in Multi-head attention layer.
        mlp_dim(int):
          * Dimension of the MLP (FeedForward) layer.
        channels(int):
          * Number of image's channels.
          * default 3
        dim_head(int):
          * dimension for multihead attention
        dropout(float):
          * float between `[0, 1]` default 0.
        emb_dropout(float):
          * float between `[0, 1]` default 0.
        pool(str):
          * `cls` token pooling or `mean` pooling.
      """
      super().__init__()
      
      """ Define values and assertion. """
      image_height, image_width = pair(image_size)
      patch_height, patch_width = pair(patch_size)
      num_patches = (image_height // patch_height) * (image_width // patch_width) # e.g.) 224 X 224 = (14*16) X (14*16)
      patch_dim = channels * patch_height * patch_width # patch image를 flatten한 차원.
      
      assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
      assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
      """ -------------------------------------------------------------------------------------------------------------------------- """ 
      
      """ Patch Embedding 1 - 논문에 쓰인대로 implementation
      
      * step 1) Image to Patches
      `from einops.layers.torch import Rearrange`를 활용하여 Image를 patch 쪼개고 flatten 하기.

      * step 2) Linear Transformation.
      flatten된 이미지 패치에 대한 linear transformation.
      """ 
      self.to_patch_embedding = nn.Sequential(
          Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
          nn.Linear(patch_dim, dim),
      )
      
      """ Patch Embedding 2 - Conv2d를 사용하기.
      * ViT 공식 jax코드나 timm(Pytorch Image Models)의 implementation 코드를 보면 Conv2d로 patch embedding을 구현함.
      
      # init...
      >>> # (N, C, H, W) -> (N, D, H//p, W//p)
      >>> self.projection = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)

      # forward...
      >>> # (N, D, H//p, W//p) -> (N, D, (H*W)//p**2) -> (N, (H*W)//p**2, D)
      >>> self.projection(img).flatten(2).transpose(1, 2)
      """
      """ -------------------------------------------------------------------------------------------------------------------------- """ 

      """ Construct Input Embedding
      * Input Embedding = [cls_token, Patch Embedding] + Positional Embedding
      """
      self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
      self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
      self.dropout = nn.Dropout(emb_dropout)
      """ -------------------------------------------------------------------------------------------------------------------------- """ 
      
      """ Vit Model
      """
      self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

      self.pool = pool
      # self.to_latent = nn.Identity() - byol implemenation purpose.

      self.mlp_head = nn.Sequential(
          nn.LayerNorm(dim),
          nn.Linear(dim, num_classes)
      )

  def forward(self, img):
      """ Patch Embedding """
      x = self.to_patch_embedding(img)
      b, n, _ = x.shape
      
      """ Input Representation """
      cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b) # broad casting.
      x = torch.cat((cls_tokens, x), dim=1) # seq의 차원으로 concat하기.
      x += self.pos_embedding[:, :(n + 1)] 
      x = self.dropout(x)

      """ transformer block """
      x = self.transformer(x)

      """ mean pooling or cls token """
      x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

      # x = self.to_latent(x) - byol implemenation purpose.
      """ mlp_head """
      return self.mlp_head(x)