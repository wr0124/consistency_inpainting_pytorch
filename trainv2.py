from typing import Tuple, List, Union
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
import numpy as np
import random
from options import parse_opts
from UNet import UNet
from UNet import UNetConfig
import torch.optim as optim
from consistency_generator import ImprovedConsistencyTraining, pseudo_huber_loss
from inference import ConsistencySamplingAndEditing
from torchvision.utils import make_grid
from visdom import Visdom
import torchvision.utils as vutils
# Decide which device we want to run on
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
args = parse_opts()
viz = Visdom(env=args.env)
#end import for visu

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_value = 42  
set_seed(seed_value)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

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
            worker_init_fn=worker_init_fn
        )


# Usage example
config = ImageDataModuleConfig(
    data_dir = args.data_dir,
    image_size = args.image_size,
    batch_size = args.batch_size,
    num_workers = args.num_workers
)
data_module = ImageDataModule(config)
data_module.setup()
train_loader = data_module.get_dataloader(shuffle=True)
print(f"train_loader len is {len(train_loader)}")
# Decide which device we want to run on
model = UNet(UNetConfig()).to(device)
summary(UNet(UNetConfig()), input_size=((1, 3, 32, 32), (1,)))


optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.995) )
scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-5, total_iters=1000
        )


@dataclass
class TrainingConfig:
    lr = args.lr
    betas: Tuple[float, float] = (0.9, 0.995)
    lr_scheduler_start_factor = args.lr_scheduler_start_factor
    lr_scheduler_iters = args.lr_scheduler_iters
    sample_every_n_steps = args.sample_every_n_steps
    num_samples = args.num_samples
    sampling_sigmas: Tuple[Tuple[int, ...], ...] = (
        (80,),
        (80.0, 0.661),
        (80.0, 24.4, 5.84, 0.9, 0.661),
    )

# Assuming your TrainingConfig class is already defined with default values
training_config = TrainingConfig()


@torch.no_grad()
def __sample_and_log_samples(batch: Union[Tensor, List[Tensor]] , config:TrainingConfig, global_step ) -> None:
    if isinstance(batch, list):
        batch = batch[0]

        # Ensure the number of samples does not exceed the batch size
    num_samples = min(config.num_samples, batch.shape[0])

        # Log ground truth samples
    __log_images(
            batch[:num_samples].detach(),
            "ground_truth",
            global_step,
            "ground_truth_window",
            window_size=(200, 200),
            )
        # option1 for sampling
        # self.ema_model.load_state_dict(self.model.state_dict())
    inpainting = ConsistencySamplingAndEditing()
    for sigmas in config.sampling_sigmas:
        samples = inpainting(
                model,
                batch.to(device),
                sigmas,
                clip_denoised=True,
                verbose=True,
            )  # Generated samples
        samples = samples.clamp(min=-1.0, max=1.0)

            # Generated samples
        __log_images(
                samples,
                f"generated_samples-sigmas={sigmas}",
                global_step,
                f"generated_samples_window_{sigmas}",
                window_size=(400, 400),
            )

@torch.no_grad()
def __log_images(
        images: Tensor,
        title: str,
        global_step: int,
        window_name: str,
        window_size: Tuple[int, int],
    ) -> None:
    images = images.detach().float()

    grid = make_grid(
            images.clamp(-1.0, 1.0), value_range=(-1.0, 1.0), normalize=True, nrow=8
            )

    grid = grid.cpu().numpy()
    if grid.min() < 0:
        grid = (grid + 1) / 2
    width, height = window_size
    viz.images(
            grid,
            nrow=4,
            opts=dict(
                title=f"{title} at step {global_step}",
                caption=f"{title}",
                width=width,
                height=height,
            ),
            win=window_name,
        )

def train_one_epoch(epoch_index, epoch_length, device, config:TrainingConfig):
    running_loss = 0.
    last_loss = 0.
    global global_step
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
        global_step +=1
        optimizer.step()
        running_loss +=loss.item()
        
        viz.line( [loss.detach().cpu().numpy()], [global_step ],win="consis_loss_iteration", opts=dict(title="loss over iteration"), update="append")
        if batch_idx %  len(train_loader) == 0:
            last_loss = running_loss / len(train_loader) #loss per batch
            #print(f"batch: {batch_idx + 1} loss: {last_loss}")
            #viz.line( [last_loss], [global_step ],win="consis_loss_iteration", opts=dict(title=f"loss over {interval_loss} iteration time"), update="append")
            running_loss = 0.
            #__sample_and_log_samples(batch.to(device), training_config, global_step)


    return last_loss




EPOCHS = int( args.max_steps / len(train_loader) )
global_step = 0  
for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    

    avg_loss = train_one_epoch(epoch, len(train_loader) ,device, training_config)

    
    print(f'trainning loss {avg_loss} at epoch {epoch+1}')
    
    
    viz.line( [avg_loss], [ epoch ],win="consis_loss_epoch", opts=dict(title="loss over epoch time"), update="append")
    if epoch % args.sample_every_n_steps == 0 or epoch == 0 or epoch == EPOCHS:
        model_path = f"checkpoints/model_epoch_{epoch}"
        model.save_pretrained(model_path)

        batch, _ = next(iter(train_loader))
        __sample_and_log_samples(batch.to(device), training_config, global_step)
    scheduler.step()
#
#    # Specify the path where the model is saved
# model_path = "checkpoints/"
#
# # Load the model
# loaded_model = UNet.from_pretrained(model_path)
#
# ###inpainting
# batch, _ = next(iter(train_loader))
# # Now you can use `loaded_model` for inference or further training
# # Create an instance of the class, optionally specify sigma_min and sigma_data
# inpainting = ConsistencySamplingAndEditing()
#
# # Now call the instance with the appropriate arguments for the __call__ method
# result = inpainting(loaded_model, batch.cpu(), (80.0, 24.4, 5.84, 0.9, 0.661))
#
