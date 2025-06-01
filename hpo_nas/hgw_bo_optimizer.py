# hpo_nas/hgw_bo_optimizer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from tqdm import tqdm

import GPyOpt # For Bayesian Optimization
from mealpy.swarm_based.GWO import OriginalGWO # For Grey Wolf Optimizer

from config import settings
from data.custom_dataset import HospitalMedicalDataset, collate_fn_remove_none
from data.transforms import get_train_transforms, get_val_transforms
from models.dynamic_nas_model import DynamicNASModel
from hpo_nas.architecture_encoding import decode_architecture_genotype #, encode_architecture_for_gwo

class HGWBO_Optimizer:
    def __init__(self, client_id, local_data_csv_path, local_data_root_dir, initial_global_model_weights=None):
        self.client_id = client_id
        self.local_data_csv_path = local_data_csv_path
        self.local_data_root_dir = local_data_root_dir
        self.initial_global_model_weights = initial_global_model_weights # For loading pre-trained weights
        
        self.device = settings.DEVICE

        self._load_and_split_local_data()
        
        # Define search spaces for BO and GWO
        # For simplicity, let's represent the combined search space as flat array for GWO,
        # and parse it within the objective function.
        # This requires careful mapping of indices.
        
        # HPO params (continuous/discrete for BO)
        # NAS params (discrete for GWO)
        
        # Bounds for GWO (architecture search)
        # Let's assume architecture is defined by a fixed sequence of N block types.
        # E.g., for 3 blocks: [block_type_1, block_type_2, block_type_3]
        # Where 0=ResNetBlock, 1=DenseNetBlock (from settings.NAS_BLOCK_MAPPING)
        num_nas_blocks = 3 # Fixed number of architecture blocks to search
        self.gwo_bounds = np.array([
            [0, len(settings.NAS_BLOCK_MAPPING) - 1] for _ in range(num_nas_blocks) # For block types
        ])
        
        # Bounds for BO (hyperparameters + block_width_factor)
        # Note: 'epochs_per_round_local' and 'block_width_factor' are discrete
        self.bo_bounds = [
            {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.01)},
            {'name': 'momentum', 'type': 'continuous', 'domain': (0.5, 0.95)},
            {'name': 'epochs_per_round_local', 'type': 'discrete', 'domain': tuple(range(5, 51))},
            {'name': 'block_width_factor', 'type': 'discrete', 'domain': tuple(range(1, 5))}, # Assuming integer factors
        ]
        self.bo_param_names = [b['name'] for b in self.bo_bounds]

    def _load_and_split_local_data(self):
        full_dataset = HospitalMedicalDataset(
            data_csv_path=self.local_data_csv_path,
            root_dir=self.local_data_root_dir,
            transform=get_train_transforms(settings.IMAGE_SIZE), # Train transforms for main data
            use_cropped_roi=True # Or False, depending on your data
        )

        # Remove None entries before splitting
        filtered_indices = [i for i, item in enumerate(full_dataset) if item is not None]
        filtered_dataset = torch.utils.data.Subset(full_dataset, filtered_indices)

        val_size = int(len(filtered_dataset) * settings.LOCAL_VALIDATION_SPLIT)
        train_size = len(filtered_dataset) - val_size
        
        if train_size == 0 or val_size == 0:
            raise ValueError(f"Client {self.client_id} has insufficient data for training/validation split.")

        self.local_train_dataset, self.local_val_dataset = random_split(filtered_dataset, [train_size, val_size])
        
        # Apply validation transforms to the validation subset
        # This requires creating a new dataset for the validation split
        self.local_val_dataset.dataset.transform = get_val_transforms(settings.IMAGE_SIZE)


        self.train_loader = DataLoader(
            self.local_train_dataset,
            batch_size=32, # Example batch size
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn_remove_none
        )
        self.val_loader = DataLoader(
            self.local_val_dataset,
            batch_size=32, # Example batch size
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn_remove_none
        )
        print(f"Client {self.client_id}: Local train samples: {len(self.local_train_dataset)}, Val samples: {len(self.local_val_dataset)}")


    def _evaluate_model_performance(self, model, optimizer_params, local_epochs):
        """
        Inner loop: Trains a specific model architecture with given hyperparameters
        on local data and returns validation performance.
        """
        model.to(self.device)
        model.train()

        # Extract specific optimizer params
        lr = optimizer_params['learning_rate']
        momentum = optimizer_params['momentum']
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        
        # Basic training loop
        for epoch in range(local_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                if images.size(0) == 0: continue # Skip empty batches
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # print(f"  Batch {i}/{len(self.train_loader)} Loss: {loss.item():.4f}") # Too verbose

        # Evaluation
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for images, labels in self.val_loader:
                if images.size(0) == 0: continue # Skip empty batches
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        # Calculate ROC AUC for binary classification
        # Ensure all_labels and all_preds are not empty for metric calculation
        if len(all_labels) == 0:
            print(f"Client {self.client_id} - No validation data processed. Returning high error.")
            return 1.0 # High error if no data

        try:
            # Need probabilities for ROC AUC, not just predicted class
            # For CrossEntropyLoss, take softmax of outputs to get probabilities
            softmax_outputs = torch.softmax(outputs, dim=1)
            # Assuming binary classification, take probability of the positive class (index 1)
            positive_class_probs = softmax_outputs[:, 1].cpu().numpy()
            
            roc_auc = roc_auc_score(all_labels, positive_class_probs)
            # We want to minimize, so return 1 - ROC AUC
            return 1 - roc_auc
        except ValueError as e:
            print(f"Client {self.client_id} - Error calculating ROC AUC: {e}. Returning high error.")
            # This can happen if only one class is present in a validation batch
            return 1.0


    def _objective_function_for_hgw_bo(self, hparams_and_arch_array):
        """
        The main objective function called by both GWO and BO.
        Takes a flat array containing both HPO and NAS parameters.
        Returns the error (1 - ROC AUC).
        """
        # Parse the flat array into HPO dict and architecture genotype
        # Assuming hparams_and_arch_array = [lr, momentum, epochs, width_factor, block1_id, block2_id, block3_id]
        
        hpo_values = hparams_and_arch_array[:len(self.bo_bounds)]
        hpo_dict = {name: val for name, val in zip(self.bo_param_names, hpo_values)}
        
        # Ensure discrete params are integers
        hpo_dict['epochs_per_round_local'] = int(round(hpo_dict['epochs_per_round_local']))
        hpo_dict['block_width_factor'] = int(round(hpo_dict['block_width_factor']))

        architecture_genotype = [int(round(x)) for x in hparams_and_arch_array[len(self.bo_bounds):]]
        
        print(f"Client {self.client_id} - Evaluating Arch: {architecture_genotype}, HPO: {hpo_dict}")

        # Decode architecture genotype into a PyTorch model
        model = DynamicNASModel(
            architecture_genotype=architecture_genotype,
            num_classes=settings.NUM_CLASSES,
            width_factor=hpo_dict['block_width_factor']
        )
        
        # Load initial global model weights if provided (applies to base layers)
        if self.initial_global_model_weights:
            model.load_state_dict(self.initial_global_model_weights, strict=False) # strict=False if arch changed

        # Train and evaluate this specific model with these hyperparameters
        error = self._evaluate_model_performance(
            model=model,
            optimizer_params={'learning_rate': hpo_dict['learning_rate'], 'momentum': hpo_dict['momentum']},
            local_epochs=hpo_dict['epochs_per_round_local']
        )
        print(f"Client {self.client_id} - Result Arch: {architecture_genotype}, HPO: {hpo_dict} -> Error: {error:.4f}")
        return error

    def find_best_local_model_and_params(self):
        """
        Orchestrates the HGW-BO optimization.
        Returns the best model, best hyperparameters, and best architecture found locally.
        """
        print(f"\n--- Client {self.client_id}: Starting HGW-BO HPO/NAS ---")

        # Combine bounds for GWO (it searches over the entire space including HPO and NAS)
        # GWO will optimize a flat array, so we need to concatenate all bounds.
        # Order: [LR, Momentum, Epochs, WidthFactor, Block1, Block2, Block3]
        combined_bounds_min = [b['domain'][0] for b in self.bo_bounds if b['type'] == 'continuous'] + \
                              [b['domain'][0] for b in self.bo_bounds if b['type'] == 'discrete'] + \
                              [b[0] for b in self.gwo_bounds] # NAS bounds
        combined_bounds_max = [b['domain'][1] for b in self.bo_bounds if b['type'] == 'continuous'] + \
                              [b['domain'][-1] for b in self.bo_bounds if b['type'] == 'discrete'] + \
                              [b[1] for b in self.gwo_bounds] # NAS bounds
        
        gwo_problem_dict = {
            "fit_func": self._objective_function_for_hgw_bo,
            "lb": np.array(combined_bounds_min),
            "ub": np.array(combined_bounds_max),
            "minmax": "min",
            "log_to": None,
            "save_population": False,
            "obj_weights": np.array([1])
        }

        # --- Outer Loop: GWO for broad exploration / initial candidates ---
        gwo_model = OriginalGWO(epoch=settings.GWO_ITERATIONS, pop_size=settings.GWO_POP_SIZE)
        gwo_best_position, gwo_best_fitness = gwo_model.solve(gwo_problem_dict)

        print(f"Client {self.client_id} - GWO Best Position: {gwo_best_position}")
        print(f"Client {self.client_id} - GWO Best Fitness: {gwo_best_fitness:.4f}")

        # --- Inner Loop: BO for local refinement (centered around GWO's best) ---
        # Initialize BO with GWO's best solution and its fitness
        initial_X_bo = np.array([gwo_best_position])
        initial_Y_bo = np.array([[gwo_best_fitness]])

        # We need to map the full combined search space for GPyOpt as well
        # and ensure types are correct.
        bo_full_bounds = []
        for b in self.bo_bounds: # Add HPO bounds
            bo_full_bounds.append(b)
        for i in range(self.gwo_bounds.shape[0]): # Add NAS bounds (as discrete for BO)
            bo_full_bounds.append({'name': f'block_type_{i+1}', 'type': 'discrete', 'domain': tuple(range(int(self.gwo_bounds[i,0]), int(self.gwo_bounds[i,1])+1))})

        bo_optimizer = GPyOpt.methods.BayesianOptimization(
            f=lambda x: self._objective_function_for_hgw_bo(x[0]),
            domain=bo_full_bounds,
            acquisition_type='EI',
            exact_feval=True,
            X=initial_X_bo,
            Y=initial_Y_bo,
            maximize=False
        )

        print(f"Client {self.client_id} - Starting BO Local Refinement...")
        bo_optimizer.run_optimization(max_iter=settings.BO_ITERATIONS_PER_GWO_EVAL, report_file=f"bo_report_client_{self.client_id}.txt")

        final_best_position_array = bo_optimizer.x_opt
        final_best_fitness = bo_optimizer.fx_opt

        # Parse the final best parameters
        best_hpo_values = final_best_position_array[:len(self.bo_bounds)]
        best_hpo_dict = {name: val for name, val in zip(self.bo_param_names, best_hpo_values)}
        best_hpo_dict['epochs_per_round_local'] = int(round(best_hpo_dict['epochs_per_round_local']))
        best_hpo_dict['block_width_factor'] = int(round(best_hpo_dict['block_width_factor']))

        best_architecture_genotype = [int(round(x)) for x in final_best_position_array[len(self.bo_bounds):]]
        
        print(f"Client {self.client_id} - HGW-BO Final Best Architecture: {best_architecture_genotype}")
        print(f"Client {self.client_id} - HGW-BO Final Best HPO: {best_hpo_dict}")
        print(f"Client {self.client_id} - HGW-BO Final Best Objective Value (1-ROC AUC): {final_best_fitness:.4f}")

        # Retrain the final best model to return its state_dict
        best_model = DynamicNASModel(
            architecture_genotype=best_architecture_genotype,
            num_classes=settings.NUM_CLASSES,
            width_factor=best_hpo_dict['block_width_factor']
        )
        if self.initial_global_model_weights:
            best_model.load_state_dict(self.initial_global_model_weights, strict=False)

        # Train for the final time with the best found hyperparameters
        print(f"Client {self.client_id} - Retraining final best model...")
        self._evaluate_model_performance( # This function actually trains the model
            model=best_model,
            optimizer_params={'learning_rate': best_hpo_dict['learning_rate'], 'momentum': best_hpo_dict['momentum']},
            local_epochs=best_hpo_dict['epochs_per_round_local']
        )
        
        return best_model.state_dict(), best_hpo_dict, best_architecture_genotype, final_best_fitness