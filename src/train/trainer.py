import torch
from tqdm import tqdm
import os
import numpy as np
import torchvision.transforms as transforms
import random
import cv2
from src.models.RESURF_MISRGRU import MISRGRU as cnn
import json
import cupy as cp
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from src.losses.losses import fourier_space_loss

class ModelTrainer:
    def __init__(
        self,
        frames,
        model="MISRGRU",
        config_path="configs/config.json",
        base_path_input = "data/input",
        base_path_target = "data/target",
        input_token="_input_",
        target_token="_target_",
        save_dir_models="checkpoints",
        epochs=400,
        batch_size=10,
        k=5,
        seed=100,
        schedular=False,
        optimizer="ADAM"
    ):
        self.frames = frames
        self.model = model
        self.config_path = config_path
        self.base_path = base_path
        self.tensor_prefix_input = tensor_prefix_input
        self.tensor_prefix_target = tensor_prefix_target
        self.save_dir_models = save_dir_models
        self.epochs = epochs
        self.batch_size = batch_size
        self.k = k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.schedular = schedular
        self.optimizer = optimizer
        self.seed = seed
        # Load config
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Dataset placeholders
        self.all_tensors = None
        self.all_targets = None
        self.all_indexes = None
        self.dataset_size = None
        self.dataset = None

        # Set random seeds for reproducibility
        self._set_seed(self.seed)

    def _set_seed(self, seed = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True # Slower but endures reproducibilty
        torch.backends.cudnn.benchmark = False # Prevents dynamic algorithm selection - (normally cude picks the fastest alg depending on hardware and env) 

    def _get_loss_fn(self):
        if self.loss_type == "L1":
            return torch.nn.L1Loss()
        elif self.loss_type == "Fourier":
            return fourier_space_loss
        elif self.loss_type == "L1_Fourier":
            return None #l1_fourier_space_loss
        else:
            return torch.nn.L1Loss()