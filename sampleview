import os
import tqdm
from typing import Dict
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50

import torch

AGENT_MOTION_CONFIG_PATH = 'config/agent_motion_config.yaml'

# Set environment variable (or pass it as an argument for LocalDataManager)
os.environ['L5KIT_DATA_FOLDER'] = 'data'
dataManager = LocalDataManager()

# Load agent motion config
cfg = load_config_data(AGENT_MOTION_CONFIG_PATH)

# Set path to training dataset
# val_data_loader key features: {key: 'scenes/train.zarr', batch_size: 12, shuffle: False, num_workers: 0})
# Batch size - number of samples processed during single epoch(=iteration)
# (Batch size) < (number of training samples) allows us not to overload machine memory +
# + frequently update weights matrix.
# Note that batch size has an ambiguous effect on gradient estimation accuracy.
# Data shuffling (in combination with small batch size) is applied to prevent SGD getting stuck in local minima(s)
# num_workers: GPU(s) and CPU(s) running the model (synchr-ly or asynchr-ly).
datasetPath = dataManager.require(cfg['val_data_loader']['key'])


# TODO: Building a model
def build_model(cfg: Dict) -> torch.nn.Module:
    # Loading pretrained 2D Convolutional Model
    model = resnet50(pretrained=True)

    # Changing input channels number to match the l5kit rasterizer's output
    # Channels representing previous frames used
    num_previous_channels = cfg['model_params']['history_num_frames']
    # Add current frame
    # EGO frames and Agents' frames go into separate channels, hence multiplication by 2
    num_current_channels = (cfg['model_params']['history_num_frames'] + 1) * 2
    # Add channels for semantic map R, G, B
    num_input_channels = num_current_channels + 3

    # For more information google: "CNN", "Kernel"
    # (e.g. https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148)
    # (https://en.wikipedia.org/wiki/Convolution)
    model.conv1 = torch.nn.Conv2d(
        num_input_channels,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False,
    )

    # Changing output size to (X, Y) * number of future states (50 as set in current configuration).
    num_targets = 2 * cfg["model_params"]["future_num_frames"]  # 2 stands for x, y coordinates

    # Applying linear transformation (A * W^T + b), for more information google: "CNN Fully-connected layer"
    model.fc = nn.Linear(in_features=2048, out_features=num_targets)

    return model


def forward(data, model, device, criterion):
    inputs = data['image'].to(device) # ".to(device)": CUDA (GPU) if possible else CPU
    target_availabilities = data['target_availabilities'].unsqueeze(-1).to(device)
    targets = data['target_position'].to(device)  # Real agents' positions

    # Forward pass
    outputs = model(inputs).reshape(targets.shape)  # Predicted agents' posistions
    loss = criterion(outputs, targets)  # Cost function

    # Not all output steps will be valid, filter required:
    loss = (loss * target_availabilities).mean()

    return loss, outputs


# INITIALIZING DATASET

# train_data_loader key features: {key: 'scenes/train.zarr', batch_size: 12, shuffle: True, num_workers: 0})
train_cfg = cfg['train_data_loader']

rasterizer = build_rasterizer(cfg, dataManager)

# "require" checks whether file with the given key is present in the local data folder.
# If it isn't raises FileNotFoundError, returns the path to the file otherwise.
# To clarify ChunkedDataset object structure look into commentary file (Strongly advised).
train_zarr = ChunkedDataset(dataManager.require(train_cfg['key'])).open()

# Wrapping the ChunkedDataset into AgentDataset, which inherist from torch Dataset abstract class.
train_dataset = AgentDataset(cfg, train_zarr, rasterizer)

# Passing data to torch DataLoader
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg['shuffle'], batch_size=train_cfg['batch_size'],
                              num_workers=train_cfg['num_workers'])

#print(train_dataset)


# INITITALIZING THE MODEL

device = torch.device('cuda:0' if torch.cuda.is_avaliable() else 'cpu')
model = build_model(cfg).to(device)

# https://pytorch.org/docs/stable/optim.html
# https://habr.com/ru/post/318970/
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Мужыки, крепитес!

criterion = torch.nn.MSELoss(reduction=None)

# TODO: TRAIN LOOP

train_iterator = iter(train_dataloader)
progress_bar = tqdm(range(cfg['train_params']['max_num_steps']))
train_losses = list()

for _ in progress_bar:
    try:
        data = next(train_iterator)
    except StopIteration:
        train_iterator = iter(train_dataloader)
        data = next(train_iterator)
    model.train()
    torch.set_grad_enabled(True)
    loss, _ = forward(data, model, device, criterion)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    progress_bar.set_description(f'Loss: {loss.item()}, AVG Loss: {np.mean(train_losses)}')


# Visualization

plt.plot(np.arange(len(train_losses)), train_losses, label='Train loss')
plt.legend()
plt.show()

# GENERATE AND LOAD CHOPPED DATASET

chopped_frames = 100  # Number of frames to chop
eval_cfg = cfg['val_data_loader']  # Evaluation config
eval_base_path = create_chopped_dataset(dm.require(eval_cfg['key']), cfg['raster_params']['filter_agents_threshold'],
                                        chopped_frames, cfg['model_params']['future_num_frames'], MIN_FUTURE_STEPS)

eval_zarr_path = str(Path(eval_base_path) / Path(dm.require(eval_cfg['key'])).name)
eval_mask_path = str(Path(eval_base_path) / 'mask.npz')
eval_gt_path = str(Path(eval_base_path) / 'gt.csv')







