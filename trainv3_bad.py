from typing import Tuple, List, Union, Optional
from torch import Tensor
import logging
import os
import glob
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import torchvision.utils as vutils
from torchinfo import summary
from visdom import Visdom
from dataclasses import dataclass
from options import parse_opts
from UNetv3 import UNet, UNetConfig
from consistency_generator import ImprovedConsistencyTraining, pseudo_huber_loss
from inference import ConsistencySamplingAndEditing
from tqdm import tqdm
from torchvision.datasets.folder import default_loader
# Set device
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
args = parse_opts()
viz = Visdom(env=args.env)

# Set seed for reproducibility
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

# Worker initialization function for DataLoader
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# CustomImageFolder class
class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None,
                 image_folder="img", mask_folder="bbox"):
        super(CustomImageFolder, self).__init__(
            root, transform=transform, target_transform=target_transform,
            loader=loader, is_valid_file=is_valid_file
        )
        self.image_folder = image_folder
        self.mask_folder = mask_folder

    def __getitem__(self, index):
        file_path, _ = self.samples[index]
        filename = os.path.basename(file_path)
        image_path = os.path.join(self.root, self.image_folder, filename)
        mask_path = os.path.join(self.root, self.mask_folder, filename)

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            raise FileNotFoundError(f"Image or mask not found for file: {filename}")

        image = self.loader(image_path)
        mask = self.loader(mask_path)

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# ImageDataModuleConfig dataclass
@dataclass
class ImageDataModuleConfig:
    data_dir: str
    image_size: Tuple[int, int]
    batch_size: int
    num_workers: int
    pin_memory: bool = True
    persistent_workers: bool = True

# ImageDataModule class
class ImageDataModule:
    def __init__(self, config: ImageDataModuleConfig) -> None:
        super().__init__()
        self.config = config

    def train_dataloader(self) -> DataLoader:
        transform = T.Compose(
                [
                    T.Resize(self.config.image_size),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Lambda(lambda x: (x * 2) - 1),
                    ]
                )
        self.dataset = CustomImageFolder(
                self.config.data_dir,
                transform=transform,
                image_folder=args.data_dir+"/img/",  # Specify the folder for images
                mask_folder=args.data_dir+"/bbox/" ,    # Specify the folder for masks
                )


        return DataLoader(
                self.dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                persistent_workers=self.config.persistent_workers,
                worker_init_fn=worker_init_fn
                )

  # TrainingConfig dataclass
@dataclass
class TrainingConfig:
    lr: float = args.lr
    betas: Tuple[float, float] = (0.9, 0.995)
    lr_scheduler_start_factor = args.lr_scheduler_start_factor
    lr_scheduler_iters = args.lr_scheduler_iters
    sample_every_n_epochs = args.sample_every_n_epochs
    num_samples = args.num_samples
    sampling_sigmas: Tuple[Tuple[int, ...], ...] = (
        (80,),
        (80.0, 0.661),
        (80.0, 24.4, 5.84, 0.9, 0.661),
    )
    model_ckpt_path: str="checkpoints/"
    resume_ckpt_path: Optional[str] = None

    def __post_init__(self):
        if args.train_continue:
            self.resume_ckpt_path = self._find_last_checkpoint()

    def _find_last_checkpoint(self) -> Optional[str]:
        dataset_name = os.path.basename(self.config.data_dir.strip("/"))
        checkpoint_folder = os.path.join(self.model_ckpt_path, dataset_name)

        # List all checkpoint files in the checkpoint directory
        checkpoint_files = glob.glob(os.path.join(checkpoint_folder, 'model_epoch_*'))

         # If no checkpoints found, return None
        if not checkpoint_files:
            logging.info(
                    f"No checkpoints found for dataset '{dataset_name}', starting training from scratch."
                    )
            return None

        # Find the checkpoint file with the highest epoch number
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1]))
        return latest_checkpoint if latest_checkpoint else None




# Main training loop
config = ImageDataModuleConfig(
    data_dir=args.data_dir,
    image_size=args.image_size,
    batch_size=args.batch_size,
    num_workers=args.num_workers
)
data_module = ImageDataModule(config)
train_loader = data_module.train_dataloader()
print(f"train_loader len is {len(train_loader)}")

# Extract the dataset name from the data_dir
dataset_name = os.path.basename(args.data_dir.strip("/"))
checkpoint_dir = os.path.join("checkpoints", dataset_name)
print(f"checkpoint_dir is {checkpoint_dir}")

model = UNet(UNetConfig()).to(device)
# Create the dataset-specific checkpoint folder if it doesn't exist
os.makedirs(checkpoint_dir, exist_ok=True)
if config.resume_ckpt_path:
    checkpoint = torch.load(
            config.resume_ckpt_path, map_location=torch.device("cpu")
            )
    state_dict = checkpoint[
            "state_dict"
            ]  # Access the state_dict key within the checkpoint
        # Remove 'model.' prefix
    adapted_state_dict = {
            k[len("model.") :]: v
            for k, v in state_dict.items()
            if k.startswith("model.")
            }
        # Now try loading the adapted state dict
    model.load_state_dict(adapted_state_dict)
else:
    logging.info("No checkpoint specified, starting training from scratch.")

      # Model configuration and summary
summary(model, input_size=((1, 3, 32, 32), (1,)))

# Optimizer and scheduler setup
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.995))
scheduler = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=args.lr_scheduler_start_factor, total_iters=args.lr_scheduler_iters
)
# Training configuration instance
training_config = TrainingConfig()

# Function for sampling and logging samples
@torch.no_grad()
def __sample_and_log_samples(batch: Union[Tensor, List[Tensor]], masks: Union[Tensor,List[Tensor]],config: TrainingConfig, global_step) -> None:
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

    inpainting = ConsistencySamplingAndEditing()
    for sigmas in config.sampling_sigmas:
        samples = inpainting(
            model,
            batch.to(device),
            sigmas,
            masks,
            clip_denoised=True,
            verbose=True,
        )
        samples = samples.clamp(min=-1.0, max=1.0)

        __log_images(
            samples,
            f"generated_samples-sigmas={sigmas}",
            global_step,
            f"generated_samples_window_{sigmas}",
            window_size=(400, 400),
        )

# Function for logging images
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

# Function for training one epoch
def train_one_epoch(epoch_index, epoch_length, device, config: TrainingConfig, train_loader):
    running_loss = 0.
    last_loss = 0.
    global global_step
    # Wrap the DataLoader with tqdm for the progress bar
    train_loader = tqdm(train_loader, desc=f'Epoch {epoch_index + 1}', dynamic_ncols=True)

    for batch_idx, (images, masks) in enumerate(train_loader):
        #print(f"batch_idx is {batch_idx}")
        binary_masks = (masks != -1 ).float()
        if isinstance(images, list):
            images = images[0]

        viz.images( vutils.make_grid( images, normalize=True, nrow=8 ), win="consistency_batch", 
                   opts=dict( title="train_batch image", caption="train_batch image",width=300, height=300,) )

        optimizer.zero_grad()

        consistency_training = ImprovedConsistencyTraining()

        global_step = epoch_index * epoch_length + batch_idx
        max_steps = EPOCHS * epoch_length
        output = consistency_training(model, images.to(device), global_step, max_steps, binary_masks.to(device))
        loss = (pseudo_huber_loss(output.predicted, output.target) * output.loss_weights).mean()
        loss.backward()
        global_step += 1
        optimizer.step()
        running_loss += loss.item()

        viz.line(
            [loss.detach().cpu().numpy()],
            [global_step],
            win="consis_loss_iteration",
            opts=dict(title="loss over iteration"),
            update="append"
        )

        if batch_idx % len(train_loader) == 0:
            last_loss = running_loss / len(train_loader)
            running_loss = 0.
        # if batch_idx % 2 == 0:
        #     __sample_and_log_samples(images.to(device), binary_masks.to(device), training_config, global_step)
        #


    return last_loss

# Main training loop
def save_model(model, global_step, epoch):
    model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}")
    model.save_pretrained(model_path, global_step=global_step)


EPOCHS = int(args.max_steps / len(train_loader))

for epoch in range(  EPOCHS ):
    print(f'EPOCH {epoch + 1}:')

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    
    avg_loss = train_one_epoch(epoch, len(train_loader), device, training_config, train_loader)

    # Log average loss over epoch
    viz.line([avg_loss], [epoch], win="consis_loss_epoch", opts=dict(title="loss over epoch time"), update="append")

    if epoch % args.sample_every_n_epochs == 0 or epoch == 0 or epoch == EPOCHS - 1:
        save_model(model, global_step, epoch)

        batch, masks = next(iter(train_loader))
        binary_masks = (masks != -1).float()
        __sample_and_log_samples(batch.to(device), binary_masks.to(device), training_config, global_step)

    scheduler.step()

