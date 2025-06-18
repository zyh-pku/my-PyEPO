import time
from tqdm import tqdm
import torch
from torch.amp import GradScaler, autocast

import pyepo

from config import DEVICE, BATCH_SIZE, NUM_EPOCHS
from data_loader import device_loader

from test_regret import sequential_regret

from config import MARKET_MODEL_DIR,MARKET_MODEL_DIR_TESTING
from model_factory import build_market_neutral_model_testing
import pickle


with open(MARKET_MODEL_DIR_TESTING, "rb") as f:
    params_testing = pickle.load(f)
    
import os


# os.environ['GUROBI_HOME'] = '/usr/licensed/gurobi/12.0.0/linux64'
# os.environ['GRB_LICENSE_FILE'] = '/usr/licensed/gurobi/license/gurobi.lic'

# # 清除个人WLS许可证
# for var in ['WLSACCESSID', 'WLSSECRET']:
#     if var in os.environ:
#         del os.environ[var]



def trainModel(model, loss_func, method_name, loader_train, loader_test, market_neutral_model, params_testing, loss_log, loss_log_regret, num_epochs=1000, lr=1e-3, initial=False, scaler= False):
    """
    Enhanced training function with:
    - Mixed precision for faster GPU training
    - Learning rate scheduling
    - Progress bars
    - Detailed logging
    - Memory-efficient tensor handling
    """
    # Set up optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1
    )
    
    if  scaler:
        # Enable mixed precision training
        scaler = GradScaler(enabled=(DEVICE.type in ["cuda", "mps"]))

    else:
        scaler = None

    # Set model to training mode
    model.train()
    
    # Initialize logs
    
    if initial: # evaluate loss on whole test data
        ## 系统Gurobi
        market_neutral_model_testing= build_market_neutral_model_testing(**params_testing)# need to initialize the testing Grb 
        regret = sequential_regret(model, market_neutral_model_testing, device_loader(loader_test))
        #loss_log_regret = [pyepo.metric.regret(model, market_neutral_model, device_loader(loader_test))]
        
        print(f"Initial regret: {regret*100:.4f}%")
    
    # Initialize elapsed time tracking
    training_start = time.time()
    total_elapsed = 0
    
    # Verbosity control - set to false for production
    debug_mode = False
    log_interval = 10  # Log every N batches
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start = time.time()
        
        # Progress bar for this epoch
        progress_bar = tqdm(loader_train, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, data in enumerate(progress_bar):
            x, c, w, z = data
            
            # Move data to GPU (once, not in every batch)
            x, c, w, z = x.to(DEVICE), c.to(DEVICE), w.to(DEVICE), z.to(DEVICE)
            
            # Record batch start time for accurate timing
            batch_start = time.time()
            
            # Clear gradients for each batch
            optimizer.zero_grad()
            
            # Use mixed precision where appropriate
            with autocast(device_type=DEVICE.type, enabled=(DEVICE.type in ["cuda", "mps"])):
                # Forward pass
                cp = model(x)
                
                # Compute loss based on method
                if method_name == "spo+":
                    loss = loss_func(cp, c, w, z)
                elif method_name in ["ptb", "pfy", "imle", "aimle", "nce", "cmap"]:
                    loss = loss_func(cp, w)
                elif method_name in ["dbb", "nid"]:
                    loss = loss_func(cp, c, z)
                elif method_name in ["pg", "ltr"]:
                    loss = loss_func(cp, c)
            
            # Backward pass with mixed precision handling
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # Gradient clipping to prevent gradient explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Track batch elapsed time
            batch_elapsed = time.time() - batch_start
            total_elapsed += batch_elapsed
            
            # Update loss tracking
            current_loss = loss.item()
            epoch_loss += current_loss
            loss_log.append(current_loss)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{current_loss:.4f}", 
                'batch time': f"{batch_elapsed:.4f}s"
            })
            
            # Debug logging (limited to avoid overwhelming output)
            if debug_mode and i % log_interval == 0:
                print(f"\n[Debug] Batch {i} stats:")
                print(f"Loss: {current_loss:.6f}")
                print(f"Pred shape: {cp.shape}, values: {cp[0,:5].detach().cpu().numpy()}")
                
                # Monitor memory usage
                if DEVICE.type == 'cuda':
                    mem_allocated = torch.cuda.memory_allocated() / 1024**2
                    mem_reserved = torch.cuda.memory_reserved() / 1024**2
                    print(f"GPU Memory: {mem_allocated:.1f}MB allocated, {mem_reserved:.1f}MB reserved")
        
        # Compute regret on test set after each epoch
        with torch.no_grad():
            model.eval()  # Set model to evaluation mode
            market_neutral_model_testing= build_market_neutral_model_testing(**params_testing)# need to reinitialize the testing Grb 
            regret = sequential_regret(model, market_neutral_model_testing, device_loader(loader_test))
            #regret = pyepo.metric.regret(model, market_neutral_model, device_loader(loader_test, device))
            model.train()  # Set back to training mode
            loss_log_regret.append(regret)
        
        # Update learning rate scheduler
        scheduler.step(epoch_loss)
        
        # End of epoch reporting
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}: Loss={epoch_loss/len(loader_train):.6f}, "
              f"Regret={regret*100:.4f}%, Time={epoch_time:.2f}s")
    
    # Report total training time
    total_training_time = time.time() - training_start
    print(f"Total training time: {total_training_time:.2f}s, "
          f"Effective computation time: {total_elapsed:.2f}s")
    
    return loss_log, loss_log_regret


def trainModel_with_log(model, loss_func, method_name, loader_train, loader_test, market_neutral_model, params_testing, loss_log, loss_log_regret, num_epochs=1000, lr=1e-3, initial=False, scaler=False, tensorboard_log_dir=None):
    """
    Enhanced training function with:
    - Mixed precision for faster GPU training
    - Learning rate scheduling
    - Progress bars
    - Detailed logging
    - Memory-efficient tensor handling
    - TensorBoard logging for metrics and gradients
    """
    from torch.utils.tensorboard import SummaryWriter
    import os
    
    # Set up TensorBoard writer
    writer = None
    if tensorboard_log_dir is not None:
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        writer = SummaryWriter(tensorboard_log_dir)
        print(f"TensorBoard logging to: {tensorboard_log_dir}")
    
    # Set up optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1
    )
    
    if scaler:
        # Enable mixed precision training
        scaler = GradScaler(enabled=(DEVICE.type in ["cuda", "mps"]))
    else:
        scaler = None

    # Set model to training mode
    model.train()
    
    # Initialize logs
    if initial: # evaluate loss on whole test data
        ## 系统Gurobi
        market_neutral_model_testing = build_market_neutral_model_testing(**params_testing)# need to initialize the testing Grb 
        regret = sequential_regret(model, market_neutral_model_testing, device_loader(loader_test))
        
        print(f"Initial regret: {regret*100:.4f}%")
        
        # Log initial regret to TensorBoard
        if writer:
            writer.add_scalar('Regret/Initial', regret, 0)
    
    # Initialize elapsed time tracking
    training_start = time.time()
    total_elapsed = 0
    global_step = 0  # For TensorBoard logging
    
    # Verbosity control - set to false for production
    debug_mode = False
    log_interval = 10  # Log every N batches
    
    # Helper function to compute gradient norm
    def compute_gradient_norm(model):
        total_norm = 0.0
        param_count = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        total_norm = total_norm ** (1. / 2)
        return total_norm, param_count
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start = time.time()
        
        # Progress bar for this epoch
        progress_bar = tqdm(loader_train, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, data in enumerate(progress_bar):
            x, c, w, z = data
            
            # Move data to GPU (once, not in every batch)
            x, c, w, z = x.to(DEVICE), c.to(DEVICE), w.to(DEVICE), z.to(DEVICE)
            
            # Record batch start time for accurate timing
            batch_start = time.time()
            
            # Clear gradients for each batch
            optimizer.zero_grad()
            
            # Use mixed precision where appropriate
            with autocast(device_type=DEVICE.type, enabled=(DEVICE.type in ["cuda", "mps"])):
                # Forward pass
                cp = model(x)
                
                # Compute loss based on method
                if method_name == "spo+":
                    loss = loss_func(cp, c, w, z)
                elif method_name in ["ptb", "pfy", "imle", "aimle", "nce", "cmap"]:
                    loss = loss_func(cp, w)
                elif method_name in ["dbb", "nid"]:
                    loss = loss_func(cp, c, z)
                elif method_name in ["pg", "ltr"]:
                    loss = loss_func(cp, c)
            
            # Backward pass with mixed precision handling
            if scaler is not None:
                scaler.scale(loss).backward()
                
                # Unscale for gradient clipping and norm computation
                scaler.unscale_(optimizer)
                
                # Compute gradient norm before clipping
                grad_norm_before, param_count = compute_gradient_norm(model)
                
                # Gradient clipping to prevent gradient explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Compute gradient norm after clipping
                grad_norm_after, _ = compute_gradient_norm(model)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                
                # Compute gradient norm before clipping
                grad_norm_before, param_count = compute_gradient_norm(model)
                
                # Gradient clipping to prevent gradient explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Compute gradient norm after clipping
                grad_norm_after, _ = compute_gradient_norm(model)
                
                optimizer.step()
            
            # Track batch elapsed time
            batch_elapsed = time.time() - batch_start
            total_elapsed += batch_elapsed
            
            # Update loss tracking
            current_loss = loss.item()
            epoch_loss += current_loss
            loss_log.append(current_loss)
            
            # TensorBoard logging
            if writer:
                # Log every batch
                writer.add_scalar('Loss/Train_Batch', current_loss, global_step)
                writer.add_scalar('Gradients/Norm_Before_Clipping', grad_norm_before, global_step)
                writer.add_scalar('Gradients/Norm_After_Clipping', grad_norm_after, global_step)
                writer.add_scalar('Training/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('Training/Batch_Time', batch_elapsed, global_step)
                
                # Log gradient clipping ratio
                clipping_ratio = grad_norm_after / max(grad_norm_before, 1e-8)
                writer.add_scalar('Gradients/Clipping_Ratio', clipping_ratio, global_step)
                
                # Log parameter statistics every 100 steps
                if global_step % 100 == 0:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            writer.add_histogram(f'Parameters/{name}', param.data, global_step)
                            writer.add_histogram(f'Gradients/{name}', param.grad.data, global_step)
                            writer.add_scalar(f'Param_Stats/{name}_mean', param.data.mean(), global_step)
                            writer.add_scalar(f'Param_Stats/{name}_std', param.data.std(), global_step)
                
                # Log memory usage if on CUDA
                if DEVICE.type == 'cuda' and global_step % 50 == 0:
                    mem_allocated = torch.cuda.memory_allocated() / 1024**2
                    mem_reserved = torch.cuda.memory_reserved() / 1024**2
                    writer.add_scalar('Memory/Allocated_MB', mem_allocated, global_step)
                    writer.add_scalar('Memory/Reserved_MB', mem_reserved, global_step)
            
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{current_loss:.4f}", 
                'grad_norm': f"{grad_norm_after:.4f}",
                'batch time': f"{batch_elapsed:.4f}s"
            })
            
            # Debug logging (limited to avoid overwhelming output)
            if debug_mode and i % log_interval == 0:
                print(f"\n[Debug] Batch {i} stats:")
                print(f"Loss: {current_loss:.6f}")
                print(f"Gradient norm (before/after clipping): {grad_norm_before:.6f}/{grad_norm_after:.6f}")
                print(f"Pred shape: {cp.shape}, values: {cp[0,:5].detach().cpu().numpy()}")
                
                # Monitor memory usage
                if DEVICE.type == 'cuda':
                    mem_allocated = torch.cuda.memory_allocated() / 1024**2
                    mem_reserved = torch.cuda.memory_reserved() / 1024**2
                    print(f"GPU Memory: {mem_allocated:.1f}MB allocated, {mem_reserved:.1f}MB reserved")
        
        # Compute regret on test set after each epoch
        with torch.no_grad():
            model.eval()  # Set model to evaluation mode
            market_neutral_model_testing = build_market_neutral_model_testing(**params_testing)# need to reinitialize the testing Grb 
            regret = sequential_regret(model, market_neutral_model_testing, device_loader(loader_test))
            model.train()  # Set back to training mode
            loss_log_regret.append(regret)
        
        # Update learning rate scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # TensorBoard logging for epoch metrics
        if writer:
            avg_epoch_loss = epoch_loss / len(loader_train)
            writer.add_scalar('Loss/Train_Epoch', avg_epoch_loss, epoch)
            writer.add_scalar('Regret/Test_Epoch', regret, epoch)
            writer.add_scalar('Training/Learning_Rate_Epoch', new_lr, epoch)
            
            # Log if learning rate changed
            if old_lr != new_lr:
                print(f"Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")
        
        # End of epoch reporting
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}: Loss={epoch_loss/len(loader_train):.6f}, "
              f"Regret={regret*100:.4f}%, Time={epoch_time:.2f}s")
    
    # Report total training time
    total_training_time = time.time() - training_start
    print(f"Total training time: {total_training_time:.2f}s, "
          f"Effective computation time: {total_elapsed:.2f}s")
    
    # Close TensorBoard writer
    if writer:
        writer.close()
        print(f"TensorBoard logs saved to: {tensorboard_log_dir}")
        print("To view logs, run: tensorboard --logdir=" + tensorboard_log_dir)
    
    return loss_log, loss_log_regret