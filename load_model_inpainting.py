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
from torchinfo import summary
import numpy as np
import random
import torch
import os
from torchvision.datasets.folder import default_loader
import glob
# Decide which device we want to run on
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
args = parse_opts()
viz = Visdom(env=args.env)

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
       # print(f"filename is {file_path}")
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
            image_folder=args.data_dir+"/img/",  # Specify the folder for images
            mask_folder=args.data_dir+"/bbox/" ,    # Specify the folder for masks
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

# Extract the dataset name from the data_dir
dataset_name = os.path.basename(args.data_dir)
checkpoint_dir = os.path.join("checkpoints", dataset_name)
print(f"checkpoint_dir is {checkpoint_dir}")

# Check if train_continue option is provided
if args.train_continue:
    # Get the list of model paths
    model_paths = glob.glob(os.path.join(checkpoint_dir, "model_epoch_*"))

    if model_paths:
        # Get the latest model path
        latest_model_path = max(model_paths, key=os.path.getctime)
        latest_model_epochs = os.path.basename(latest_model_path).split("_")[-1]
        print(f"latest_model_epoch is {latest_model_epochs}")
        print(f"Model loaded from {latest_model_path}")

        # Load the latest model
        model = UNet.from_pretrained(latest_model_path).to(device)
    else:
        print(f"No existing models found in {checkpoint_dir}. Train from scratch.")
        # If not continuing, create a new model
        model = UNet(UNetConfig()).to(device)

else:
    print(f"No latest models found for {dataset_name}. Train from scratch.")
    # If not continuing, create a new model
    model = UNet(UNetConfig()).to(device)


# Model configuration and summary
#summary(model, input_size=((1, 3, 32, 32), (1,)))


# sampling configuration
@dataclass
class SamplingConfig:
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

# Training configuration instance
Sampling_config = SamplingConfig()

# Function for sampling and logging samples
@torch.no_grad()
def __sample_and_log_samples(batch: Union[Tensor, List[Tensor]], masks: Union[Tensor,List[Tensor]],config: SamplingConfig, latest_model_epoch) -> None:

    # if isinstance(batch, list):
    #     batch = batch[0]
    #
    # Ensure the number of samples does not exceed the batch size
    num_samples = min(config.num_samples, batch.shape[0])

    # Log ground truth samples
    __log_images(
        batch[:num_samples].detach(),
        "ground_truth",
        latest_model_epoch,
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
            latest_model_epoch,
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
    

# Usage example
config = ImageDataModuleConfig(
    data_dir=args.data_dir,
    image_size=args.image_size,
    batch_size=args.batch_size,
    num_workers=args.num_workers
)
data_module = ImageDataModule(config)
data_module.setup()
train_loader = data_module.get_dataloader(shuffle=True)
print(f"train_loader len is {len(train_loader)}")

batch, masks = next(iter(train_loader))
binary_masks = (masks != -1).float()
__sample_and_log_samples(batch.to(device), binary_masks.to(device),Sampling_config, latest_model_epochs)
