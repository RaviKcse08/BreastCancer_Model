# hpo_nas/architecture_encoding.py

# This file defines how an architecture is encoded and decoded.
# The HGW-BO optimizer will work with this encoding.

# Example: Simple fixed-length sequence of block types
# e.g., genotype = [block_type_id_1, block_type_id_2, ..., block_type_id_N]
# where block_type_id_X maps to a specific block class (ResNetBlock, DenseNetBlock)

# Function to decode genotype to a model instance
from models.dynamic_nas_model import DynamicNASModel
from config import settings

def decode_architecture_genotype(genotype_array, num_classes, width_factor):
    """
    Decodes a numpy array representing the architecture genotype
    into a PyTorch model instance.
    """
    # Assuming genotype_array's last elements are the block type IDs
    # and the first element is the block_width_factor (if it's a HPO param)
    
    # This mapping must be consistent with settings.HPO_NAS_SEARCH_SPACE
    # and settings.NAS_BLOCK_MAPPING
    
    # Example: If HPO_NAS_SEARCH_SPACE has 4 HPO params + 3 NAS block types
    # then genotype_array[0-3] are HPO and genotype_array[4-6] are NAS.
    # We need to extract the architecture genotype part.
    
    # For our example where HPO_NAS_SEARCH_SPACE includes block_width_factor
    # and then block_type_1, block_type_2, block_type_3 for NAS
    # The block_width_factor is actually a HPO param, which will be passed separately.
    # The NAS genotype part comes from the GWO's search for architecture.
    
    # Let's adjust: The GWO in outer loop will search for discrete architecture choices.
    # The inner BO will search for continuous HPO + discrete block_width_factor.

    # Revised conceptualization:
    # GWO searches for: [block_type_1_id, block_type_2_id, ...] (integer/discrete values)
    # BO searches for: [learning_rate, momentum, epochs_per_round_local, block_width_factor] (continuous/discrete)

    # So, the genotype_array from GWO would directly be the architecture.
    architecture_sequence = [int(round(x)) for x in genotype_array]
    
    # Now, DynamicNASModel needs the architecture sequence and a width_factor.
    # The width_factor comes from the BO inner loop.
    # This means the objective function will receive *both* arch and hpo parameters.
    # This will be handled in the hgw_bo_optimizer's objective.

    # For now, this decoder will just take the architecture and assume width_factor is passed in.
    # This function is more about mapping integer IDs to block names.
    return architecture_sequence

# Function to encode architecture for GWO (reverse of decode)
def encode_architecture_for_gwo(arch_sequence):
    """
    Encodes an architecture sequence back into a numpy array for GWO.
    """
    # Simply return the sequence as a numpy array of integers.
    return np.array(arch_sequence)