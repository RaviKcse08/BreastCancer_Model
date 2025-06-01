# config/settings.py

import torch

# --- Global FL Settings ---
GLOBAL_ROUNDS = 50          # Number of federated learning rounds
NUM_CLIENTS = 10            # Total number of hospitals participating
CLIENT_PARTICIPATION_RATE = 0.5 # Percentage of clients participating per round (if global, not HPO)
GLOBAL_MODEL_SAVE_PATH = "global_model.pth"

# --- Common Model Settings ---
NUM_CLASSES = 2             # e.g., Benign/Malignant for breast cancer
IMAGE_SIZE = (224, 224)     # Input size for the DL model
PRETRAINED_IMAGENET = True  # Use ImageNet pre-trained weights for base models

# --- HPO/NAS Search Space (Define for Bilevel Optimizer) ---
# Each element corresponds to a parameter in the order expected by the optimizer
HPO_NAS_SEARCH_SPACE = [
    # HPO Parameters for the inner BO loop
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.01)},
    {'name': 'momentum', 'type': 'continuous', 'domain': (0.5, 0.95)},
    {'name': 'epochs_per_round_local', 'type': 'discrete', 'domain': tuple(range(5, 51))}, # Local epochs
    # Note: 'num_neurons' and 'num_layers' might be part of NAS encoding,
    # or general HPO for blocks. Clarify where they apply in your design.
    # For now, let's assume 'num_neurons' refers to a flexible width within blocks, and 'num_layers' the total number of blocks.
    # If these are purely NAS decisions, they should be in the NAS encoding.
    {'name': 'block_width_factor', 'type': 'discrete', 'domain': tuple(range(1, 5))}, # e.g., for ResNet/DenseNet widths
    
    # NAS Parameters for the outer GWO loop (encoded as integers/categories)
    # Example: A sequence of block types, say 3 blocks.
    # 0: ResNetBlock, 1: DenseNetBlock, 2: CustomBlock
    # The actual NAS representation is crucial here (e.g., fixed length sequence of block types)
    {'name': 'block_type_1', 'type': 'discrete', 'domain': (0, 1)}, # 0=ResNet, 1=DenseNet
    {'name': 'block_type_2', 'type': 'discrete', 'domain': (0, 1)},
    {'name': 'block_type_3', 'type': 'discrete', 'domain': (0, 1)},
    # And potentially connection choices, skip connections, etc.
]

# Mapping for discrete NAS parameters (e.g., for block types)
NAS_BLOCK_MAPPING = {
    0: 'ResNetBlock',
    1: 'DenseNetBlock',
    # 2: 'CustomBlock'
}

# --- Optimization Settings ---
GWO_POP_SIZE = 15           # Number of wolves in GWO (candidate architectures)
GWO_ITERATIONS = 5          # Number of GWO updates (architecture search rounds)
BO_ITERATIONS_PER_GWO_EVAL = 10 # Number of BO evaluations per architecture candidate
LOCAL_VALIDATION_SPLIT = 0.2 # % of local data for validation during HPO/NAS

# --- Device Settings ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")