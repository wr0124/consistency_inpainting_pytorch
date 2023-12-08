from typing import Tuple
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from dataclasses import asdict, dataclass
from torch import Tensor, nn
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from einops import rearrange
import torch
from typing import Callable
import os
import json
from torchinfo import summary

from UNet import UNet
import torch.optim as optim
from consistency_generator import ImprovedConsistencyTraining, pseudo_huber_loss
from inference import ConsistencySamplingAndEditing
import sys
from torchvision.utils import make_grid
from visdom import Visdom
import torchvision.utils as vutils
viz = Visdom(env="consistency_trainpy_test")

#end import for visu
@dataclass
class ImageDataModuleConfig:
    data_dir: str
    image_size: Tuple[int, int]
    batch_size: int
    num_workers: int
    pin_memory: bool = True
    persistent_workers: bool = True

class ImageDataModule:
    def __init__(self, config: ImageDataModuleConfig) -> None:
        self.config = config
        self.dataset = None

    def setup(self) -> None:
        transform = T.Compose(
            [
                T.Resize(self.config.image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Lambda(lambda x: (x * 2) - 1),
            ]
        )
        self.dataset = ImageFolder(self.config.data_dir, transform=transform)

    def get_dataloader(self, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )

# # Usage example
# config = ImageDataModuleConfig(
#     data_dir='/data3/juliew/datasets/butterflies/',
#     image_size=(32, 32),
#     batch_size=32,
#     num_workers=12
# )
# data_module = ImageDataModule(config)
# data_module.setup()
# train_loader = data_module.get_dataloader(shuffle=True)
# # Now you can use train_loader in your training loop
# def show_images(images, nrow=8, padding=2):
#     """Display a batch of images using Visdom."""
#     grid = make_grid(images, nrow=nrow, padding=padding, normalize=True)
#     viz.image(grid, opts=dict(title='Batch of Images', caption='From DataLoader'))
#
# for batch in train_loader:
#     images, _ = batch  # Assuming each batch returns images and labels
#     show_images(images)
#     break  # Just show the first batch
# # end for the dataload usage
# Modules
# def GroupNorm(channels: int) -> nn.GroupNorm:
#     return nn.GroupNorm(num_groups=min(32, channels // 4), num_channels=channels)
#
#
# class SelfAttention(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         n_heads: int = 8,
#         dropout: float = 0.3,
#     ) -> None:
#         super().__init__()
#
#         self.dropout = dropout
#
#         self.qkv_projection = nn.Sequential(
#             GroupNorm(in_channels),
#             nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1, bias=False),
#             Rearrange("b (i h d) x y -> i b h (x y) d", i=3, h=n_heads),
#         )
#         self.output_projection = nn.Sequential(
#             Rearrange("b h l d -> b l (h d)"),
#             nn.Linear(in_channels, out_channels, bias=False),
#             Rearrange("b l d -> b d l"),
#             GroupNorm(out_channels),
#             nn.Dropout1d(dropout),
#         )
#         self.residual_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#
#     def forward(self, x: Tensor) -> Tensor:
#         q, k, v = self.qkv_projection(x).unbind(dim=0)
#
#         output = F.scaled_dot_product_attention(
#             q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False
#         )
#         output = self.output_projection(output)
#         output = rearrange(output, "b c (x y) -> b c x y", x=x.shape[-2], y=x.shape[-1])
#
#         return output + self.residual_projection(x)
#
#
# class UNetBlock(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         noise_level_channels: int,
#         dropout: float = 0.3,
#     ) -> None:
#         super().__init__()
#
#         self.input_projection = nn.Sequential(
#             GroupNorm(in_channels),
#             nn.SiLU(),
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
#             nn.Dropout2d(dropout),
#         )
#         self.noise_level_projection = nn.Sequential(
#             nn.SiLU(),
#             nn.Conv2d(noise_level_channels, out_channels, kernel_size=1),
#         )
#         self.output_projection = nn.Sequential(
#             GroupNorm(out_channels),
#             nn.SiLU(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
#             nn.Dropout2d(dropout),
#         )
#         self.residual_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#
#     def forward(self, x: Tensor, noise_level: Tensor) -> Tensor:
#         h = self.input_projection(x)
#         h = h + self.noise_level_projection(noise_level)
#
#         return self.output_projection(h) + self.residual_projection(x)
#
#
# class UNetBlockWithSelfAttention(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         noise_level_channels: int,
#         n_heads: int = 8,
#         dropout: float = 0.3,
#     ) -> None:
#         super().__init__()
#
#         self.unet_block = UNetBlock(
#             in_channels, out_channels, noise_level_channels, dropout
#         )
#         self.self_attention = SelfAttention(
#             out_channels, out_channels, n_heads, dropout
#         )
#
#     def forward(self, x: Tensor, noise_level: Tensor) -> Tensor:
#         return self.self_attention(self.unet_block(x, noise_level))
#
#
# class Downsample(nn.Module):
#     def __init__(self, channels: int) -> None:
#         super().__init__()
#
#         self.projection = nn.Sequential(
#             Rearrange("b c (h ph) (w pw) -> b (c ph pw) h w", ph=2, pw=2),
#             nn.Conv2d(4 * channels, channels, kernel_size=1),
#         )
#
#     def forward(self, x: Tensor) -> Tensor:
#         return self.projection(x)
#
#
# class Upsample(nn.Module):
#     def __init__(self, channels: int) -> None:
#         super().__init__()
#
#         self.projection = nn.Sequential(
#             nn.Upsample(scale_factor=2.0, mode="nearest"),
#             nn.Conv2d(channels, channels, kernel_size=3, padding="same"),
#         )
#
#     def forward(self, x: Tensor) -> Tensor:
#         return self.projection(x)
#
#
# class NoiseLevelEmbedding(nn.Module):
#     def __init__(self, channels: int, scale: float = 0.02) -> None:
#         super().__init__()
#
#         self.W = nn.Parameter(torch.randn(channels // 2) * scale, requires_grad=False)
#
#         self.projection = nn.Sequential(
#             nn.Linear(channels, 4 * channels),
#             nn.SiLU(),
#             nn.Linear(4 * channels, channels),
#             Rearrange("b c -> b c () ()"),
#         )
#
#     def forward(self, x: Tensor) -> Tensor:
#         h = x[:, None] * self.W[None, :] * 2 * torch.pi
#         h = torch.cat([torch.sin(h), torch.cos(h)], dim=-1)
#
#         return self.projection(h)
#
#
# # Unet
# @dataclass
# class UNetConfig:
#     channels: int = 3
#     noise_level_channels: int = 256
#     noise_level_scale: float = 0.02
#     n_heads: int = 8
#     top_blocks_channels: Tuple[int, ...] = (128, 128)
#     top_blocks_n_blocks_per_resolution: Tuple[int, ...] = (2, 2)
#     top_blocks_has_resampling: Tuple[bool, ...] = (True, True)
#     top_blocks_dropout: Tuple[float, ...] = (0.0, 0.0)
#     mid_blocks_channels: Tuple[int, ...] = (256, 512)
#     mid_blocks_n_blocks_per_resolution: Tuple[int, ...] = (4, 4)
#     mid_blocks_has_resampling: Tuple[bool, ...] = (True, False)
#     mid_blocks_dropout: Tuple[float, ...] = (0.0, 0.3)
#
#
#
# class UNet(nn.Module):
#     def __init__(self, config: UNetConfig, checkpoint_dir: str = "") -> None:
#         super().__init__()
#         self.config = config
#         self.checkpoint_dir = checkpoint_dir
#
#         self.input_projection = nn.Conv2d(
#             config.channels,
#             config.top_blocks_channels[0],
#             kernel_size=3,
#             padding="same",
#         )
#         self.noise_level_embedding = NoiseLevelEmbedding(
#             config.noise_level_channels, config.noise_level_scale
#         )
#         self.top_encoder_blocks = self._make_encoder_blocks(
#             self.config.top_blocks_channels + self.config.mid_blocks_channels[:1],
#             self.config.top_blocks_n_blocks_per_resolution,
#             self.config.top_blocks_has_resampling,
#             self.config.top_blocks_dropout,
#             self._make_top_block,
#         )
#         self.mid_encoder_blocks = self._make_encoder_blocks(
#             self.config.mid_blocks_channels + self.config.mid_blocks_channels[-1:],
#             self.config.mid_blocks_n_blocks_per_resolution,
#             self.config.mid_blocks_has_resampling,
#             self.config.mid_blocks_dropout,
#             self._make_mid_block,
#         )
#         self.mid_decoder_blocks = self._make_decoder_blocks(
#             self.config.mid_blocks_channels + self.config.mid_blocks_channels[-1:],
#             self.config.mid_blocks_n_blocks_per_resolution,
#             self.config.mid_blocks_has_resampling,
#             self.config.mid_blocks_dropout,
#             self._make_mid_block,
#         )
#         self.top_decoder_blocks = self._make_decoder_blocks(
#             self.config.top_blocks_channels + self.config.mid_blocks_channels[:1],
#             self.config.top_blocks_n_blocks_per_resolution,
#             self.config.top_blocks_has_resampling,
#             self.config.top_blocks_dropout,
#             self._make_top_block,
#         )
#         self.output_projection = nn.Conv2d(
#             config.top_blocks_channels[0],
#             config.channels,
#             kernel_size=3,
#             padding="same",
#         )
#
#     def forward(self, x: Tensor, noise_level: Tensor) -> Tensor:
#         h = self.input_projection(x)
#         noise_level = self.noise_level_embedding(noise_level)
#
#         top_encoder_embeddings = []
#         for block in self.top_encoder_blocks:
#             if isinstance(block, UNetBlock):
#                 h = block(h, noise_level)
#                 top_encoder_embeddings.append(h)
#             else:
#                 h = block(h)
#
#         mid_encoder_embeddings = []
#         for block in self.mid_encoder_blocks:
#             if isinstance(block, UNetBlockWithSelfAttention):
#                 h = block(h, noise_level)
#                 mid_encoder_embeddings.append(h)
#             else:
#                 h = block(h)
#
#         for block in self.mid_decoder_blocks:
#             if isinstance(block, UNetBlockWithSelfAttention):
#                 h = torch.cat((h, mid_encoder_embeddings.pop()), dim=1)
#                 h = block(h, noise_level)
#             else:
#                 h = block(h)
#
#         for block in self.top_decoder_blocks:
#             if isinstance(block, UNetBlock):
#                 h = torch.cat((h, top_encoder_embeddings.pop()), dim=1)
#                 h = block(h, noise_level)
#             else:
#                 h = block(h)
#
#         return self.output_projection(h)
#
#     def _make_encoder_blocks(
#         self,
#         channels: Tuple[int, ...],
#         n_blocks_per_resolution: Tuple[int, ...],
#         has_resampling: Tuple[bool, ...],
#         dropout: Tuple[float, ...],
#         block_fn: Callable[[], nn.Module],
#     ) -> nn.ModuleList:
#         blocks = nn.ModuleList()
#
#         channel_pairs = list(zip(channels[:-1], channels[1:]))
#         for idx, (in_channels, out_channels) in enumerate(channel_pairs):
#             for _ in range(n_blocks_per_resolution[idx]):
#                 blocks.append(block_fn(in_channels, out_channels, dropout[idx]))
#                 in_channels = out_channels
#
#             if has_resampling[idx]:
#                 blocks.append(Downsample(out_channels))
#
#         return blocks
#
#     def _make_decoder_blocks(
#         self,
#         channels: Tuple[int, ...],
#         n_blocks_per_resolution: Tuple[int, ...],
#         has_resampling: Tuple[bool, ...],
#         dropout: Tuple[float, ...],
#         block_fn: Callable[[], nn.Module],
#     ) -> nn.ModuleList:
#         blocks = nn.ModuleList()
#
#         channel_pairs = list(zip(channels[:-1], channels[1:]))[::-1]
#         for idx, (out_channels, in_channels) in enumerate(channel_pairs):
#             if has_resampling[::-1][idx]:
#                 blocks.append(Upsample(in_channels))
#
#             inner_blocks = []
#             for _ in range(n_blocks_per_resolution[::-1][idx]):
#                 inner_blocks.append(
#                     block_fn(in_channels * 2, out_channels, dropout[::-1][idx])
#                 )
#                 out_channels = in_channels
#             blocks.extend(inner_blocks[::-1])
#
#         return blocks
#
#     def _make_top_block(
#         self, in_channels: int, out_channels: int, dropout: float
#     ) -> UNetBlock:
#         return UNetBlock(
#             in_channels,
#             out_channels,
#             self.config.noise_level_channels,
#             dropout,
#         )
#
#     def _make_mid_block(
#         self,
#         in_channels: int,
#         out_channels: int,
#         dropout: float,
#     ) -> UNetBlockWithSelfAttention:
#         return UNetBlockWithSelfAttention(
#             in_channels,
#             out_channels,
#             self.config.noise_level_channels,
#             self.config.n_heads,
#             dropout,
#         )
#
#     def save_pretrained(self, pretrained_path: str) -> None:
#         os.makedirs(pretrained_path, exist_ok=True)
#
#         with open(os.path.join(pretrained_path, "config.json"), mode="w") as f:
#             json.dump(asdict(self.config), f)
#
#         torch.save(self.state_dict(), os.path.join(pretrained_path, "model.ckpt"))
#
#     @classmethod
#     def from_pretrained(cls, pretrained_path: str) -> "UNet":
#         with open(os.path.join(pretrained_path, "config.json"), mode="r") as f:
#             config_dict = json.load(f)
#         config = UNetConfig(**config_dict)
#
#         model = cls(config)
#
#         state_dict = torch.load(
#             os.path.join(pretrained_path, "model.ckpt"), map_location=torch.device("cpu")
#         )
#         model.load_state_dict(state_dict)
#
#         return model
summary(UNet(UNetConfig()), input_size=((1, 3, 32, 32), (1,)))
#sys.exit()


# Usage example
config = ImageDataModuleConfig(
    data_dir='/data3/juliew/datasets/butterflies/',
    image_size=(32, 32),
    batch_size=32,
    num_workers=12
)
data_module = ImageDataModule(config)
data_module.setup()
train_loader = data_module.get_dataloader(shuffle=True)
#def show_images(images, nrow=8, padding=2):
#    """Display a batch of images using Visdom."""
#    grid = make_grid(images, nrow=nrow, padding=padding, normalize=True)
#    viz.image(grid, opts=dict(title='Batch of Images', caption='From DataLoader'))

#for batch in train_loader:
#    images, _ = batch  # Assuming each batch returns images and labels
#    show_images(images)
#    break  # Just show the first batch

# Decide which device we want to run on
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
model = UNet(UNetConfig()).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.995) )
scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-5, total_iters=10_000
        )

def train_one_epoch(epoch_index, epoch_length, device):
    running_loss = 0.
    last_loss = 0.
    interval_loss = 10
    for batch_idx, batch in enumerate(train_loader):
        #print(f"batch_idx is {batch_idx}")
        if isinstance(batch, list):
            batch = batch[0]
        viz.images( vutils.make_grid( batch, normalize=True, nrow=8 ), win="consistency_batch", 
                   opts=dict( title="train_batch image", caption="train_batch image",width=300, height=300,) )

        optimizer.zero_grad()
 
        consistency_training = ImprovedConsistencyTraining()
        
        # epoch_length: number of batches in one epoch (number of batches in the dataset)
        global_step = epoch_index * epoch_length + batch_idx
        max_steps = EPOCHS * epoch_length
        output = consistency_training( model, batch.to(device), global_step, max_steps)       
        #print( f"global_step is {global_step}")
        loss = ( pseudo_huber_loss(output.predicted, output.target) * output.loss_weights ).mean()
        loss.backward()
        
        optimizer.step()
        running_loss +=loss.item()
        
        if batch_idx % interval_loss == (interval_loss - 1):
            last_loss = running_loss /interval_loss #loss per batch
            #print(f"batch: {batch_idx + 1} loss: {last_loss}")
            viz.line( [last_loss], [global_step ],win="consis_loss_iteration", opts=dict(title=f"loss over {interval_loss} iteration time"), update="append")
            running_loss = 0.
    return last_loss




EPOCHS = 10000

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    

    avg_loss = train_one_epoch(epoch, len(train_loader) ,device)

    
    print(f'trainning loss {avg_loss} at epoch {epoch+1}')
    
    
    viz.line( [avg_loss], [ epoch ],win="consis_loss_epoch", opts=dict(title="loss over epoch time"), update="append")
    if epoch % 1000 == 0 or epoch == 0 or epoch == EPOCHS:
        model_path = f"checkpoints/model_epoch_{epoch}"
        model.save_pretrained(model_path)



   # Specify the path where the model is saved
model_path = "checkpoints/"

# Load the model
loaded_model = UNet.from_pretrained(model_path)
###inpainting
batch, _ = next(iter(train_loader))
# Now you can use `loaded_model` for inference or further training
consistency_inpainting = ConsistencySamplingAndEditing()

inpainting = consistency_inpainting( loaded_model,  batch.cpu(), (80.0, 24.4, 5.84, 0.9, 0.661))
