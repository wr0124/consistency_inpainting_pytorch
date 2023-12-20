from typing import Tuple, List, Union
from tqdm import tqdm
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from dataclasses import dataclass
from torch import Tensor
from options import parse_opts
from UNet import UNet, UNetConfig
import torch.optim as optim
from consistency_generator import ImprovedConsistencyTraining, pseudo_huber_loss
from inference import ConsistencySamplingAndEditing
from torchvision.utils import make_grid
from visdom import Visdom
import torchvision.utils as vutils
import torch
import os
from torchvision.datasets.folder import default_loader
import glob
import numpy as np

# Decide which device to run on
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
args = parse_opts()
viz = Visdom(env=args.env)

# Set seed
seed_value = 42  
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Custom Image Folder class
class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None,
                 image_folder="img", mask_folder="bbox"):
        super().__init__(root, transform=transform, target_transform=target_transform,
                         loader=loader, is_valid_file=is_valid_file)
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

# Data module configuration
@dataclass
class ImageDataModuleConfig:
    data_dir: str
    image_size: Tuple[int, int]
    batch_size: int
    num_workers: int
    pin_memory: bool = True
    persistent_workers: bool = True

# Data module class
class ImageDataModule:
    def __init__(self, config: ImageDataModuleConfig) -> None:
        self.config = config
        self.dataset = None

    def setup(self) -> None:
        transform = T.Compose([
            T.Resize(self.config.image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Lambda(lambda x: (x * 2) - 1),
        ])
        self.dataset = CustomImageFolder(
            self.config.data_dir,
            transform=transform,
            image_folder=os.path.join(args.data_dir, "img"),
            mask_folder=os.path.join(args.data_dir, "bbox")
        )

    def get_dataloader(self, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )

# Training configuration
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

# Main training loop
def train_one_epoch(epoch_index, epoch_length, device, config: TrainingConfig, train_loader, model, optimizer, scheduler, viz):
    #model.train(True)
    running_loss = 0.
    global_step = 0

    train_loader = tqdm(train_loader, desc=f'Epoch {epoch_index + 1}', dynamic_ncols=True)
    EPOCHS = int(args.max_steps / len(train_loader))
    for batch_idx, (images, masks) in enumerate(train_loader):
        binary_masks = (masks != -1).float()

        optimizer.zero_grad()
        consistency_training = ImprovedConsistencyTraining()
        global_step = epoch_index * epoch_length + batch_idx
        max_steps = EPOCHS * epoch_length

        output = consistency_training(model, images.to(device), global_step, max_steps, binary_masks.to(device))
        loss = (pseudo_huber_loss(output.predicted, output.target) * output.loss_weights).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()

        viz.line([loss.detach().cpu().numpy()], [global_step],
                 win="consis_loss_iteration", opts=dict(title="loss over iteration"), update="append")

        if batch_idx % len(train_loader) == 0:
            running_loss = 0.

        if batch_idx % 50 == 0:
            __sample_and_log_samples(images.to(device), binary_masks.to(device), config, global_step)

    return running_loss / len(train_loader)

# Main training loop
def main():
    EPOCHS = int(args.max_steps / len(train_loader))
    global_step = 0

    for epoch in range(EPOCHS):
        print(f'EPOCH {epoch + 1}:')

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch, len(train_loader), device, training_config, train_loader, model, optimizer, scheduler, viz)

        viz.line([avg_loss], [epoch], win="consis_loss_epoch", opts=dict(title="loss over epoch time"), update="append")

        if epoch % args.sample_every_n_epochs == 0 or epoch == 0 or epoch == EPOCHS - 1:
            model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}")
            model.save_pretrained(model_path)

            batch, masks = next(iter(train_loader))
            binary_masks = (masks != -1).float()
            __sample_and_log_samples(batch.to(device), binary_masks.to(device), training_config, global_step)

        scheduler.step()

if __name__ == "__main__":
    # Model configuration and summary
    model = UNet(UNetConfig()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.995))
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=args.lr_scheduler_start_factor, total_iters=args.lr_scheduler_iters)

    # Training configuration
    training_config = TrainingConfig()

    # Data module
    dm_config = ImageDataModuleConfig(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    data_module = ImageDataModule(dm_config)
    data_module.setup()
    train_loader = data_module.get_dataloader(shuffle=True)
    print(f"train_loader len is {len(train_loader)}")

    # Extract the dataset name from the data_dir
    dataset_name = os.path.basename(args.data_dir)
    checkpoint_dir = os.path.join("checkpoints", dataset_name)
    print(f"checkpoint_dir is {checkpoint_dir}")

    # Create the dataset-specific checkpoint folder if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Check if train_continue option is provided
    if args.train_continue:
        # Get the list of model paths
        model_paths = glob.glob(os.path.join(checkpoint_dir, "model_epoch_*"))

        if model_paths:
            # Get the latest model path
            latest_model_path = max(model_paths, key=os.path.getctime)
            print(f"Model loaded from {latest_model_path}")

            # Load the latest model
            model = UNet.from_pretrained(latest_model_path)
        else:
            print(f"No existing models found in {checkpoint_dir}. Train from scratch.")
    else:
        print(f"No latest models found for {dataset_name}. Train from scratch.")

    main()

