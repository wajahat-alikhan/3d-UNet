import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import nibabel as nib
import numpy as np
from pathlib import Path
import re
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet3D, self).__init__()
        
        features = init_features
        
        # Encoder path
        self.encoder1 = DoubleConv3D(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        
        self.encoder2 = DoubleConv3D(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        
        self.encoder3 = DoubleConv3D(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        
        # Bridge
        self.bridge = DoubleConv3D(features * 4, features * 8)
        
        # Decoder path
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, 
            kernel_size=2, stride=2,
            padding=0
        )
        self.decoder3 = DoubleConv3D(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2,
            kernel_size=2, stride=2,
            padding=0
        )
        self.decoder2 = DoubleConv3D(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features,
            kernel_size=2, stride=2,
            padding=0
        )
        self.decoder1 = DoubleConv3D(features * 2, features)
        
        # Final convolution
        self.final_conv = nn.Conv3d(features, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Store input size for later use
        input_size = x.size()
        
        # Encoder
        enc1 = self.encoder1(x)
        x = self.pool1(enc1)
        
        enc2 = self.encoder2(x)
        x = self.pool2(enc2)
        
        enc3 = self.encoder3(x)
        x = self.pool3(enc3)
        
        # Bridge
        x = self.bridge(x)
        
        # Decoder with size checking and adjustment
        # Upsample 1
        x = self.upconv3(x)
        # Adjust sizes if necessary
        enc3 = self._adjust_size(enc3, x)
        x = torch.cat((x, enc3), dim=1)
        x = self.decoder3(x)
        
        # Upsample 2
        x = self.upconv2(x)
        enc2 = self._adjust_size(enc2, x)
        x = torch.cat((x, enc2), dim=1)
        x = self.decoder2(x)
        
        # Upsample 3
        x = self.upconv1(x)
        enc1 = self._adjust_size(enc1, x)
        x = torch.cat((x, enc1), dim=1)
        x = self.decoder1(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Adjust final output size if necessary
        if x.size() != input_size:
            x = self._adjust_size(x, target_tensor=None, target_size=input_size)
            
        return x
    
    def _adjust_size(self, x, target_tensor=None, target_size=None):
        """
        Adjusts the size of tensor x to match the target tensor's size
        or the specified target size
        """
        if target_tensor is not None:
            target_size = target_tensor.size()
        
        if x.size() == target_size:
            return x
            
        # Calculate necessary padding
        pad_dims = []
        for i, (src, dst) in enumerate(zip(x.shape[2:], target_size[2:])):
            pad_amt = dst - src
            if pad_amt > 0:
                pad_dims.extend([pad_amt//2, pad_amt - pad_amt//2])
            elif pad_amt < 0:
                # If source is larger, we need to crop
                crop = -pad_amt
                x = x.narrow(i+2, crop//2, src - crop)
                pad_dims.extend([0, 0])
            else:
                pad_dims.extend([0, 0])
                
        if any(p != 0 for p in pad_dims):
            x = F.pad(x, pad_dims)
            
        return x

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class BrainModalityDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Verify directory exists
        if not self.data_dir.exists():
            raise ValueError(f"Directory not found: {self.data_dir}")
            
        print(f"Scanning directory: {self.data_dir}")
        self.pairs = self._find_pairs()
        
        if len(self.pairs) == 0:
            raise ValueError("No valid MRI-PET pairs found in the directory")
            
        print(f"Successfully initialized dataset with {len(self.pairs)} pairs")
    
    def _find_pairs(self) -> list:
        """Find and match MRI-PET pairs based on subject ID and date"""
        pairs = []
        nii_files = list(self.data_dir.glob("*.nii"))
        print(f"Found {len(nii_files)} total .nii files")
        
        # Group files by subject and date
        pairs_dict = {}
        for file_path in nii_files:
            # Print each file being processed for debugging
            print(f"Processing file: {file_path.name}")
            
            pattern = r'(\d+_S_\d+)_(\d{4}[-_]\d{2})'
            match = re.search(pattern, file_path.name)
            if not match:
                print(f"Warning: Could not parse filename pattern for {file_path.name}")
                continue
            
            subject_id, date = match.groups()
            key = f"{subject_id}_{date.replace('-', '_')}"
            
            # Sort files into MRI and PET based on filename
            if 'MPRAGE' in file_path.name:
                if key not in pairs_dict:
                    pairs_dict[key] = {'mri': None, 'pet': None}
                pairs_dict[key]['mri'] = file_path
                print(f"Found MRI for {key}")
            elif 'PET_FDG_Coreg_Ave' in file_path.name:
                if key not in pairs_dict:
                    pairs_dict[key] = {'mri': None, 'pet': None}
                pairs_dict[key]['pet'] = file_path
                print(f"Found PET for {key}")
        
        # Create validated pairs
        for key, pair_dict in pairs_dict.items():
            if pair_dict['mri'] and pair_dict['pet']:
                pairs.append((pair_dict['mri'], pair_dict['pet']))
                print(f"Created pair for {key}")
        
        return pairs

    def _load_and_preprocess(self, file_path: Path) -> torch.Tensor:
        """
        Load and preprocess a NIfTI file.
        Handles NaN values and normalizes the data.
        """
        try:
            img = nib.load(file_path)
            data = img.get_fdata()
            
            # Handle NaN values
            if np.isnan(data).any():
                data = np.nan_to_num(data, nan=np.nanmedian(data))
            
            # Normalize to zero mean and unit variance
            data = (data - np.mean(data)) / (np.std(data) + 1e-8)
            
            # Convert to tensor and add channel dimension
            tensor = torch.FloatTensor(data).unsqueeze(0)
            
            return tensor
        except Exception as e:
            raise RuntimeError(f"Error loading file {file_path}: {str(e)}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a pair of MRI and PET images"""
        mri_path, pet_path = self.pairs[idx]
        
        mri_tensor = self._load_and_preprocess(mri_path)
        pet_tensor = self._load_and_preprocess(pet_path)
        
        if self.transform:
            mri_tensor = self.transform(mri_tensor)
            pet_tensor = self.transform(pet_tensor)
        
        return mri_tensor, pet_tensor

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 device, num_epochs=100, save_dir='checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Create visualization directory
        self.vis_dir = self.save_dir / 'visualizations'
        self.vis_dir.mkdir(exist_ok=True)
        
        # Initialize tracking variables
        self.train_losses = []
        self.val_losses = []
        self.psnr_scores = []
        self.ssim_scores = []
        self.best_val_loss = float('inf')
        
        # Add learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10, verbose=True
        )
    
    def calculate_metrics(self, pred, target):
        """Calculate PSNR and SSIM metrics"""
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Calculate metrics for each image in batch
        psnr_vals = []
        ssim_vals = []
        
        for i in range(pred_np.shape[0]):
            # Take central slice for 2D metrics
            slice_idx = pred_np.shape[2] // 2
            pred_slice = pred_np[i, 0, slice_idx]
            target_slice = target_np[i, 0, slice_idx]
            
            # Calculate PSNR
            psnr_val = psnr(target_slice, pred_slice, 
                           data_range=target_slice.max() - target_slice.min())
            psnr_vals.append(psnr_val)
            
            # Calculate SSIM
            ssim_val = ssim(target_slice, pred_slice, 
                           data_range=target_slice.max() - target_slice.min())
            ssim_vals.append(ssim_val)
        
        return np.mean(psnr_vals), np.mean(ssim_vals)

    def save_prediction_examples(self, mri, pred_pet, true_pet, epoch):
        """Save example predictions as images"""
        # Take middle slices
        slice_idx = mri.shape[2] // 2
        
        # Get the first image from the batch
        mri_slice = mri[0, 0, slice_idx].cpu().numpy()
        pred_slice = pred_pet[0, 0, slice_idx].cpu().numpy()
        true_slice = true_pet[0, 0, slice_idx].cpu().numpy()
        
        # Create figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot MRI
        axes[0].imshow(mri_slice, cmap='gray')
        axes[0].set_title('Input MRI')
        axes[0].axis('off')
        
        # Plot Predicted PET
        axes[1].imshow(pred_slice, cmap='hot')
        axes[1].set_title('Predicted PET')
        axes[1].axis('off')
        
        # Plot True PET
        axes[2].imshow(true_slice, cmap='hot')
        axes[2].set_title('Ground Truth PET')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.vis_dir / f'prediction_epoch_{epoch}.png')
        plt.close()

    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'psnr_scores': self.psnr_scores,
            'ssim_scores': self.ssim_scores
        }
        torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{epoch}.pt')
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, self.save_dir / 'best_model.pt')
            print(f"Saved new best model with validation loss: {val_loss:.4f}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc='Training', leave=False) as pbar:
            for mri, pet in pbar:
                mri = mri.to(self.device)
                pet = pet.to(self.device)
                
                self.optimizer.zero_grad()
                predicted_pet = self.model(mri)
                loss = self.criterion(predicted_pet, pet)
                
                # Calculate PSNR and SSIM
                psnr_val, ssim_val = self.calculate_metrics(predicted_pet, pet)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                epoch_psnr += psnr_val
                epoch_ssim += ssim_val
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'psnr': f'{psnr_val:.2f}',
                    'ssim': f'{ssim_val:.4f}'
                })
        
        return (epoch_loss / num_batches, 
                epoch_psnr / num_batches,
                epoch_ssim / num_batches)

    def validate(self):
        """Run validation"""
        self.model.eval()
        val_loss = 0
        val_psnr = 0
        val_ssim = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for mri, pet in self.val_loader:
                mri = mri.to(self.device)
                pet = pet.to(self.device)
                
                predicted_pet = self.model(mri)
                loss = self.criterion(predicted_pet, pet)
                
                # Calculate PSNR and SSIM
                psnr_val, ssim_val = self.calculate_metrics(predicted_pet, pet)
                
                val_loss += loss.item()
                val_psnr += psnr_val
                val_ssim += ssim_val
                
                # Save example predictions for the last batch
                if len(self.val_losses) % 5 == 0:  # Save every 5 epochs
                    self.save_prediction_examples(mri, predicted_pet, pet, len(self.val_losses))
        
        return (val_loss / num_batches,
                val_psnr / num_batches,
                val_ssim / num_batches)

    def plot_learning_curves(self):
        """Plot and save learning curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot PSNR and SSIM
        ax2.plot(self.psnr_scores, label='PSNR', color='green')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(self.ssim_scores, label='SSIM', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('PSNR (dB)')
        ax2_twin.set_ylabel('SSIM')
        ax2.set_title('PSNR and SSIM Metrics')
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'learning_curves.png')
        plt.close()

    def train(self):
        """Main training loop"""
        print("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            # Training phase
            train_loss, train_psnr, train_ssim = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation phase
            val_loss, val_psnr, val_ssim = self.validate()
            self.val_losses.append(val_loss)
            self.psnr_scores.append(val_psnr)
            self.ssim_scores.append(val_ssim)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss)
            
            # Print progress
            print(f"\nEpoch [{epoch+1}/{self.num_epochs}]")
            print(f"Training Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f}, SSIM: {train_ssim:.4f}")
            print(f"Validation Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Plot learning curves
            self.plot_learning_curves()
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/60:.2f} minutes")

def create_model(in_channels=1, out_channels=1, init_features=32):
    """Create and initialize the 3D UNet model"""
    model = UNet3D(in_channels, out_channels, init_features)
    
    # Initialize model weights
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    return model

def train_model(data_dir, batch_size=2, num_epochs=100, learning_rate=0.001):
    """Main function to setup and start training"""
    
    # Create dataset
    dataset = BrainModalityDataset(data_dir)
    
    # Create train/val split
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    # Calculate split sizes - 80% train, 20% validation
    split = int(np.floor(0.2 * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Create samplers for train and validation sets
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    # Create data loaders with the samplers
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=0
    )
    
    print(f"\nTraining set size: {len(train_indices)}")
    print(f"Validation set size: {len(val_indices)}")
    
    # Setup device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = create_model(in_channels=1, out_channels=1, init_features=32)
    print("Model created with 3D convolutions")
    
    # Initialize loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create trainer instance
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs
    )
    
    return trainer

# Main execution block with error handling
if __name__ == "__main__":
    # Define the path to your data directory containing .nii files
    data_dir = 'D:\Wajahat Ali Khan\KHU Gangdong Hospital Data\PET1_FDG\ADNI3_new'  # Replace with your actual data path
    
    try:
        # Create and start training with carefully chosen parameters
        trainer = train_model(
            data_dir=data_dir,
            batch_size=2,  # Small batch size for 3D data to manage memory
            num_epochs=100,  # Adjust based on convergence needs
            learning_rate=0.001  # Standard learning rate for Adam optimizer
        )
        
        print("\nInitializing training process...")
        # Begin the training process
        trainer.train()
        
    except Exception as e:
        print(f"\nError occurred during training process: {str(e)}")
        # Print the full traceback for debugging
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()
        raise