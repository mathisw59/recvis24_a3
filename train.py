from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import argparse
import os

from model_factory import ModelFactory

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    SpinnerColumn
)
from rich.console import Console

def plot_confusion_matrix(cm, class_names):
    """Helper function to plot a confusion matrix for TensorBoard logging."""
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, format(cm[i, j], ".2f"), horizontalalignment="center", color=color)
    
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    return figure

def plot_feature_importance(model, inputs, targets, writer, epoch):
    """Fixed version with proper memory cleanup"""
    inputs = inputs.to(next(model.parameters()).device)
    inputs.requires_grad_()
    
    try:
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets.to(outputs.device))
        loss.backward()
        grad_norm = inputs.grad.norm(dim=1)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(range(len(grad_norm)), grad_norm.detach().cpu().numpy())
        ax.set_xlabel('Feature')
        ax.set_ylabel('Importance')
        ax.set_title('Feature Importance')
        writer.add_figure('Feature Importance', fig, global_step=epoch)
        plt.close(fig)  # Close figure to prevent memory leak
    finally:
        # Cleanup
        inputs.grad.data.zero_()
        del inputs.grad
        torch.cuda.empty_cache()

def train(
    modelFactory: ModelFactory
):
    model = modelFactory.model
    optimizer = modelFactory.optimizer
    criterion = modelFactory.criterion
    train_loader = modelFactory.train_loader
    val_loader = modelFactory.val_loader
    scheduler = modelFactory.scheduler

    ### LOGGING SETUP ###
    console = Console()

    # Create TensorBoard writer with a unique directory for each model
    console.print(f"[bold blue]Log Dir:[/] {modelFactory.log_dir}")
    console.print(f"[bold yellow]Model Checkpoint:[/] {modelFactory.checkpoint_path}")
    console.print(f"[bold green]Configuration File:[/] {modelFactory.config_path}")

    writer = SummaryWriter(log_dir=modelFactory.log_dir)
    writer.add_hparams({
        'lr': modelFactory.config.optimizer.lr,
        'batch_size': modelFactory.config.data.batch_size,
        'model config file': modelFactory.config_path
    }, {})

    best_val_loss = np.inf
    no_improvement_epochs = 0

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

    with Progress(*progress_columns, console=console) as progress:
        # Add tasks for epoch, batch, and validation progress
        epoch_task = progress.add_task(
            "[cyan bold]Training Epochs",
            total=modelFactory.epochs
        )
        batch_task = progress.add_task(
            "[yellow bold]Current Epoch Progress",
            total=len(train_loader),
            visible=False
        )
        validation_task = progress.add_task(
            "[green bold]Validating Batches",
            total=len(val_loader),
            visible=False
        )

        # We add a try/finally block to empty CUDA cache in case of errors
        try:
            ### TRAINING LOOP ###
            for epoch in range(1, modelFactory.epochs + 1):

                # Reset and show batch progress bar for new epoch
                progress.reset(batch_task)
                progress.update(batch_task, visible=True, description=f"[yellow bold]Epoch {epoch}/{modelFactory.epochs}", completed=0, total=len(train_loader))


                model.train()
                total_loss = 0
                correct = 0 if modelFactory.supervised else None
                ssl_acc = 0 if not modelFactory.supervised else None

                ### BATCH LOOP ###
                for batch_idx, (data, target) in enumerate(train_loader):

                    ### TRAINING STEP ###
                    if modelFactory.use_cuda:
                        data, target = data.cuda(), target.cuda()

                    optimizer.zero_grad()

                    if modelFactory.mixed_precision:
                        with torch.cuda.amp.autocast():
                            # torch.cuda.empty_cache()
                            output = model(data)
                            if modelFactory.supervised:
                                loss = criterion(output, target)
                            else:
                                batch_accuracy, loss = criterion(output, target)
                                ssl_acc += batch_accuracy.item()

                        modelFactory.scaler.scale(loss).backward()
                        if modelFactory.gradient_clipping:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), modelFactory.gradient_clipping) 
                        modelFactory.scaler.step(optimizer)
                        modelFactory.scaler.update()
                    
                    else:
                        # torch.cuda.empty_cache() # Clear CUDA cache to prevent OOM errors
                        output = model(data)
                        if modelFactory.supervised:
                            loss = criterion(output, target)
                        else:
                            batch_accuracy, loss = criterion(output, target)
                            ssl_acc += batch_accuracy.item()

                        loss.backward()
                        if modelFactory.gradient_clipping:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), modelFactory.gradient_clipping)
                        optimizer.step()

                    ### PER-BATCH LOGGING / LOSS,ACC CALCULATION ###
                    total_loss += loss.item()

                    if modelFactory.supervised:
                        pred = output.argmax(dim=1, keepdim=True)
                        n_correct = pred.eq(target.view_as(pred)).sum().item()
                        correct += n_correct
                        batch_accuracy = n_correct / len(target)

                    # Update batch progress with current loss
                    progress.update(
                        batch_task,
                        completed=batch_idx + 1,
                        description=f"[yellow bold]Epoch {epoch}/{modelFactory.epochs} - "
                                f"Loss: {loss.item():.4f} - "
                                f"Acc: {100 * batch_accuracy:.2f}%"
                    )

                    # Log batch-level training metrics
                    writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_loader) + batch_idx)
                    if modelFactory.supervised:
                        writer.add_scalar('Accuracy/train_batch', 100 * batch_accuracy, epoch * len(train_loader) + batch_idx)
                    else:
                        writer.add_scalar('SSL Accuracy/train_batch', 100 * batch_accuracy, epoch * len(train_loader) + batch_idx)
                    writer.flush()

                ### VALIDATION STEP ###
                # if epoch == modelFactory.epochs:
                val_loss = validation(modelFactory, writer, epoch, progress, validation_task)
                # else:
                #     val_loss = 0
                ## MODEL SAVING ##
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improvement_epochs = 0
                    # Save model
                    checkpoint_path = f"{modelFactory.checkpoint_path}_best.pth"
                    if modelFactory.supervised:
                        torch.save(model[0].state_dict(), checkpoint_path)
                    else:
                        torch.save(model.model[0].state_dict(), checkpoint_path)
                    console.print(f"[green]Model saved:[/] {checkpoint_path}")
                else:
                    no_improvement_epochs += 1
                
                if epoch % modelFactory.save_interval == 0:
                    checkpoint_path = os.path.join(modelFactory.checkpoint_path, f"{modelFactory.save_name}_{epoch}.pth")
                    if modelFactory.supervised:
                        torch.save(model[0].state_dict(), checkpoint_path)
                    else:
                        torch.save(model.model[0].state_dict(), checkpoint_path)
                    console.print(f"[green]Model saved:[/] {checkpoint_path}")

                ## SCHEDULER STEP ##
                if scheduler:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
                    current_lr = scheduler.get_last_lr()[0]
                    writer.add_scalar('Hyperparameters/Learning Rate', current_lr, epoch)

                ### EPOCH LOGGING ###
                avg_train_loss = total_loss / len(train_loader)
                writer.add_scalar('Loss/train', avg_train_loss, epoch)
                if modelFactory.supervised:
                    avg_train_accuracy = 100. * correct / len(train_loader.dataset)
                    writer.add_scalar('Accuracy/train', avg_train_accuracy, epoch)
                else:
                    ssl_acc /= len(train_loader)
                    writer.add_scalar('SSL Accuracy/train', 100. * ssl_acc, epoch)

                writer.add_scalar('Validation/Best Loss', best_val_loss, epoch)
                writer.add_scalar('Validation/No Improvement Epochs', no_improvement_epochs, epoch)

                progress.update(
                    epoch_task,
                    completed=epoch,
                    description=f"[cyan bold]Training Epochs - "
                            f"Train Loss: {avg_train_loss:.4f} - "
                            f"Val Loss: {val_loss:.4f}"
                )

                if modelFactory.log_interval is not None and (epoch % modelFactory.log_interval == 0 or epoch == modelFactory.epochs):
                    console.print(f"[bold]Epoch [{epoch}/{modelFactory.epochs}][/], "
                                f"[blue]Train Loss:[/] {avg_train_loss:.4f}, "
                                f"[yellow]Validation Loss:[/] {val_loss:.4f}")
                
                # if modelFactory.supervised:
                #     plot_feature_importance(model, next(iter(val_loader))[0], next(iter(val_loader))[1], writer, epoch)

                ## EARLY STOPPING ##
                # if modelFactory.early_stopping and no_improvement_epochs >= modelFactory.early_stopping.patience:
                #     console.print("[bold red]Early stopping triggered[/]")
                #     break

        finally:
            # Cleanup
            writer.close()
            torch.cuda.empty_cache()



def validation(
    modelFactory: ModelFactory,
    writer: SummaryWriter,
    epoch: int,
    progress: Progress,
    validation_task: TaskProgressColumn,
):
    model = modelFactory.model
    criterion = modelFactory.criterion
    val_loader = modelFactory.val_loader
    
    model.eval()

    validation_loss = 0
    correct = 0
    all_preds = []
    all_targets = []

    class_names = val_loader.dataset.classes

    with torch.no_grad():
        for data, target in val_loader:
            if modelFactory.use_cuda:
                data, target = data.cuda(), target.cuda()

            # torch.cuda.empty_cache()
            output = model(data)
            
            if modelFactory.supervised:
                validation_loss += criterion(output, target).item()
            else:
                acc, loss = criterion(output, target)
                validation_loss += loss.item()
                correct += acc.item()

            if modelFactory.supervised:
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                all_preds.extend(pred.view(-1).cpu().numpy())
                all_targets.extend(target.view(-1).cpu().numpy())

            progress.update(validation_task, advance=1)


    # Log validation loss
    validation_loss /= len(val_loader)
    writer.add_scalar('Loss/validation', validation_loss, epoch)

    if modelFactory.supervised:
        accuracy = 100. * correct / len(val_loader)
        writer.add_scalar('Accuracy/validation', accuracy, epoch)

        if class_names:
            cm = confusion_matrix(all_targets, all_preds)
            cm_fig = plot_confusion_matrix(cm, class_names)
            writer.add_figure("Confusion Matrix", cm_fig, global_step=epoch)

    else:
        writer.add_scalar('SSL Accuracy/validation', 100. * correct / len(val_loader), epoch)

    return validation_loss