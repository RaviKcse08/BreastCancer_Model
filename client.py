# client.py

import flwr as fl
import torch
import collections

from hpo_nas.hgw_bo_optimizer import HGWBO_Optimizer
from models.dynamic_nas_model import DynamicNASModel
from config import settings

class HospitalClient(fl.client.NumPyClient):
    def __init__(self, client_id, data_csv_path, data_root_dir):
        self.client_id = client_id
        self.data_csv_path = data_csv_path
        self.data_root_dir = data_root_dir
        self.current_global_model_weights = None
        self.current_global_architecture = None # If the global architecture also evolves

        # Initialize an empty model to get the structure for initial weights request
        # This will be replaced by the dynamically built model after HPO/NAS
        self.model = DynamicNASModel(
            architecture_genotype=[0,0,0], # Dummy initial arch
            num_classes=settings.NUM_CLASSES,
            width_factor=1
        ).to(settings.DEVICE)

    def get_parameters(self, config):
        # This function is called by the server to get initial model parameters.
        # We send the parameters of our *current* best local model.
        # If it's the very first round, this will be the initialized dummy model's params.
        # After HPO/NAS, it will be the newly trained best local model's params.
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        # This function is called by the server to send updated global model parameters.
        # Store them to be used as initial weights for local HPO/NAS.
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = collections.OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
        # Store these weights. They will be used as the starting point for local HPO/NAS.
        self.current_global_model_weights = state_dict
        print(f"Client {self.client_id}: Received global model parameters.")

    def fit(self, parameters, config):
        # This is the core training function for each federated round.
        self.set_parameters(parameters) # Update local model with global parameters

        print(f"Client {self.client_id}: Starting local HPO/NAS and training...")
        
        # Instantiate and run the HGW-BO optimizer
        hgw_bo_optimizer = HGWBO_Optimizer(
            client_id=self.client_id,
            local_data_csv_path=self.data_csv_path,
            local_data_root_dir=self.data_root_dir,
            initial_global_model_weights=self.current_global_model_weights
        )

        # This call orchestrates GWO for NAS and BO for HPO, then retrains best model
        best_local_model_state_dict, best_hpo, best_architecture, final_fitness = hgw_bo_optimizer.find_best_local_model_and_params()

        # Update the client's current model with the best found local model
        self.model = DynamicNASModel(
            architecture_genotype=best_architecture,
            num_classes=settings.NUM_CLASSES,
            width_factor=best_hpo['block_width_factor']
        ).to(settings.DEVICE)
        self.model.load_state_dict(best_local_model_state_dict)

        # Return the updated model parameters, number of local examples, and metrics
        # The metrics can be used by the server for logging or performance-based aggregation.
        num_local_examples = len(hgw_bo_optimizer.local_train_dataset)
        metrics = {"loss": final_fitness, "accuracy": 1 - final_fitness} # Store 1-ROC_AUC as accuracy metric
        
        # If the server also needs to adapt its global architecture,
        # you might need to send `best_architecture` back in the metrics.
        # However, for basic FedAvg, only weights are aggregated.
        return [val.cpu().numpy() for val in self.model.state_dict().values()], num_local_examples, metrics

    def evaluate(self, parameters, config):
        # The server calls this to evaluate the global model on client's local test data.
        # Here, we evaluate the *received global model* (not our locally tuned model).
        self.set_parameters(parameters)

        print(f"Client {self.client_id}: Evaluating global model...")
        self.model.eval() # Use the model with global weights
        
        # You'd need a separate test loader for evaluation, not the validation loader
        # created in HGWBO_Optimizer. For simplicity, reusing val_loader for demonstration.
        # In a real setup, load a distinct test dataset.
        
        all_labels = []
        all_preds_probs = [] # For ROC AUC
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            # Use the val_loader from the HGWBO_Optimizer for this client
            hgw_bo_temp = HGWBO_Optimizer(self.client_id, self.data_csv_path, self.data_root_dir)
            
            for images, labels in hgw_bo_temp.val_loader: # Use the local validation set for evaluation
                if images.size(0) == 0: continue
                images, labels = images.to(settings.DEVICE), labels.to(settings.DEVICE)
                outputs = self.model(images)
                
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_preds_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()) # Probs for positive class

        avg_loss = total_loss / total if total > 0 else 1.0
        accuracy = correct / total if total > 0 else 0.0
        
        roc_auc = 0.5 # Default if no data
        if len(all_labels) > 0 and len(np.unique(all_labels)) > 1: # Need at least two classes for ROC AUC
            roc_auc = roc_auc_score(all_labels, all_preds_probs)
        elif len(all_labels) > 0: # Only one class present
            print(f"Client {self.client_id}: Only one class present in evaluation data. Cannot compute ROC AUC. Accuracy: {accuracy:.4f}")

        print(f"Client {self.client_id}: Global model evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")
        return avg_loss, total, {"accuracy": accuracy, "roc_auc": roc_auc}


# --- Function to start a client (for simulation) ---
def start_client(cid: str, data_csv_path: str, data_root_dir: str):
    client = HospitalClient(cid, data_csv_path, data_root_dir)
    # Start the client with Flower, connecting to the server
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)