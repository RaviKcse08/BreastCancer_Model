# data/custom_dataset.py

from torch.utils.data import Dataset
import os
import pandas as pd
from data.transforms import dicom_to_pil # Import our DICOM utility

class HospitalMedicalDataset(Dataset):
    def __init__(self, data_csv_path, root_dir, transform=None, use_cropped_roi=True):
        self.annotations = pd.read_csv(data_csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.use_cropped_roi = use_cropped_roi
        self.label_map = {'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0, 'MALIGNANT': 1} # For DDSM

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_relative_path = self.annotations.iloc[idx]['cropped image file path'] if self.use_cropped_roi \
                            else self.annotations.iloc[idx]['image file path']
        
        # Adjust path construction based on your DDSM organization (e.g., './CBIS-DDSM/Mass-Training/Mass-Training_P_00001_LEFT_CC/1-1.dcm')
        img_path = os.path.join(self.root_dir, img_relative_path).replace("\\", "/")

        if not os.path.exists(img_path):
            print(f"Warning: Image file not found at {img_path}. Skipping sample {idx}.")
            return None # Signal to collate_fn to skip

        try:
            image = dicom_to_pil(img_path)
            label = self.label_map[self.annotations.iloc[idx]['pathology']]

            if self.transform:
                image = self.transform(image)
            
            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None

# Custom collate_fn to filter out None values from the dataset
def collate_fn_remove_none(batch):
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if not batch: # Return empty tensors if batch is empty after filtering
        return torch.empty(0, 3, settings.IMAGE_SIZE[0], settings.IMAGE_SIZE[1]), torch.empty(0, dtype=torch.long)
    return torch.utils.data.dataloader.default_collate(batch)