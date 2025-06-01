
# server.py

import flwr as fl
import torch
from collections import OrderedDict
from typing import Dict, List, Tuple
from flwr.common import Parameters, Scalar

from models.dynamic_nas_model import DynamicNASModel
from config import settings

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, initial_model_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_model_weights = initial_model_weights
        self.current_global_model = DynamicNASModel(
            architecture_genotype=[0,0,0], # Dummy initial arch
            num_classes=settings.NUM_CLASSES,
            width_factor=1
        ).to(settings.DEVICE)
        self.current_global_model.load_state_dict(initial_model_weights)
        print("Server: Initial global model loaded.")

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        # This method controls which clients participate.
        # Use settings.CLIENT_PARTICIPATION_RATE here.
        config = {"server_round": server_round}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        # Sample clients based on participation rate
        sample_size = int(client_manager.num_available() * settings.CLIENT_PARTICIPATION_RATE)
        sample_size = max(1, sample_size) # Ensure at least one client
        
        min_fit_clients = self.min_fit_clients # From FedAvg strategy, ensure enough clients
        
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_fit_clients
        )
        
        return [(client, fl.common.FitIns(parameters, config)) for client in clients]


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
    ) -> Tuple[Parameters, Dict[str, Scalar]]:
        # Aggregate client model updates using FedAvg
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # Convert aggregated parameters back to PyTorch state_dict
        params_dict = zip(self.current_global_model.state_dict().keys(), aggregated_parameters.tensors)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.current_global_model.load_state_dict(state_dict)

        # Save the global model at the end of each round (optional)
        torch.save(self.current_global_model.state_dict(), settings.GLOBAL_MODEL_SAVE_PATH)
        print(f"Server: Global model saved after round {server_round}.")

        # Log aggregated metrics
        print(f"Server: Aggregated metrics after round {server_round}: {aggregated_metrics}")

        return aggregated_parameters, aggregated_metrics

    def evaluate(self, server_round: int, parameters: Parameters) -> Tuple[float, Dict[str, Scalar]]:
        # Evaluate the global model on the server-side (if server has a test set)
        # Or, let clients evaluate the global model via the `evaluate` client method
        # Here, we'll rely on client evaluations for simplicity.
        return 0.0, {} # No server-side evaluation in this simplified example


# --- Global Model Initialization ---
def initialize_global_model():
    # Define a default initial architecture for the global model
    # (e.g., a simple ResNet-like structure with default width)
    initial_arch = [0, 0, 0] # Three ResNet blocks
    initial_width_factor = 1

    model = DynamicNASModel(
        architecture_genotype=initial_arch,
        num_classes=settings.NUM_CLASSES,
        width_factor=initial_width_factor
    )
    # Optionally load pre-trained weights if not training from scratch
    # For example, if you want to use ImageNet pre-trained ResNet as base
    # models.resnet18(pretrained=True) to extract state_dict
    
    return model.state_dict()

# --- Main Server Logic ---
if __name__ == "__main__":
    initial_weights = initialize_global_model()
    
    # Convert initial weights to Flower Parameters object
    initial_parameters = fl.common.parameters_to_weights(initial_weights.values())
    initial_parameters = fl.common.Parameters(tensors=[p.cpu().numpy() for p in initial_parameters], tensor_type="numpy.ndarray")

    # Define strategy
    strategy = SaveModelStrategy(
        initial_model_weights=initial_weights,
        fraction_fit=settings.CLIENT_PARTICIPATION_RATE, # Clients participating in fit
        fraction_evaluate=1.0, # All clients evaluate
        min_fit_clients=int(settings.NUM_CLIENTS * settings.CLIENT_PARTICIPATION_RATE),
        min_evaluate_clients=settings.NUM_CLIENTS,
        min_available_clients=settings.NUM_CLIENTS,
        initial_parameters=initial_parameters
    )

    # Start the Flower server
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=settings.GLOBAL_ROUNDS),
        strategy=strategy,
    )