# MLP Handwriting Recognition and Localization
# Author: Muhamad Faez Abdullah

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics  # Import torchmetrics
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import itertools

class HandWritingMLPModelTrainerV3:
    
    def __init__(self, model, train_loader, val_loader, test_loader, epochs, lr, device,patience = 10):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        self.writer = SummaryWriter()
        self.patience = patience
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stop = False
        self.best_model_path = 'best_model.pth'  # Path to save the best model
        
        # Initialize and move metrics to the correct device
        self.accuracy_train = torchmetrics.Accuracy(num_classes=62, average='macro', task='multiclass').to(device)
        self.precision_train = torchmetrics.Precision(num_classes=62, average='macro', task='multiclass').to(device)
        self.recall_train = torchmetrics.Recall(num_classes=62, average='macro', task='multiclass').to(device)
        self.f1_train = torchmetrics.F1Score(num_classes=62, average='macro', task='multiclass').to(device)

        self.accuracy_val = torchmetrics.Accuracy(num_classes=62, average='macro', task='multiclass').to(device)
        self.precision_val = torchmetrics.Precision(num_classes=62, average='macro', task='multiclass').to(device)
        self.recall_val = torchmetrics.Recall(num_classes=62, average='macro', task='multiclass').to(device)
        self.f1_val = torchmetrics.F1Score(num_classes=62, average='macro', task='multiclass').to(device)
        
        self.accuracy_test = torchmetrics.Accuracy(num_classes=62, average='macro', task='multiclass').to(device)
        self.precision_test = torchmetrics.Precision(num_classes=62, average='macro', task='multiclass').to(device)
        self.recall_test = torchmetrics.Recall(num_classes=62, average='macro', task='multiclass').to(device)
        self.f1_test = torchmetrics.F1Score(num_classes=62, average='macro', task='multiclass').to(device)


    def plot_confusion_matrix(self, cm, class_names, epoch, phase):
        """
        Plots the confusion matrix and logs it to TensorBoard.

        Args:
        - cm: The confusion matrix to plot.
        - class_names: List of class names for the axis labels.
        - epoch: Current epoch for logging.
        - phase: The phase ('train', 'validate', 'test') for logging.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names)

        # Loop over data dimensions and create text annotations.
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title(f'Confusion Matrix - {phase.capitalize()} Epoch: {epoch}')

        # Instead of directly logging to TensorBoard here, return the figure
        return fig

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            self.accuracy_train.reset()
            self.precision_train.reset()
            self.recall_train.reset()
            self.f1_train.reset()
            # Initialize variables to store predictions and true labels
            all_preds = []
            all_labels = []
            
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
                # Store predictions and true labels
                _, preds = torch.max(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                
                self.accuracy_train.update(preds, labels)
                self.precision_train.update(preds, labels)
                self.recall_train.update(preds, labels)
                self.f1_train.update(preds, labels)
            
            # At the end of the epoch, compute the metrics
            avg_loss = total_loss / len(self.train_loader)
            train_accuracy = self.accuracy_train.compute()
            precision = self.precision_train.compute()
            recall = self.recall_train.compute()
            f1_score = self.f1_train.compute()
            
            # Concatenate all predictions and true labels
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            
            # Compute the confusion matrix
            # Inside your train, validate, or test methods, where you compute the confusion matrix:
            cm = torchmetrics.functional.confusion_matrix(all_preds, all_labels, num_classes=62, task='multiclass')

            
            # Plot and log the confusion matrix
            fig = self.plot_confusion_matrix(cm.numpy(), ['class_'+str(i) for i in range(62)], epoch, 'train')
            self.writer.add_figure('Confusion Matrix/train', fig, global_step=epoch)
            
            print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f} Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}')
            
            self.writer.add_scalars('Loss', {'train': avg_loss}, epoch)
            self.writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            self.writer.add_scalars('Precision', {'train': precision}, epoch)
            self.writer.add_scalars('Recall', {'train': recall}, epoch)
            self.writer.add_scalar('F1 Score/train', f1_score, epoch)
            
            val_loss = self.validate(epoch)  # Now validate returns avg_val_loss

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                print(f"Validation loss decreased to {val_loss:.4f}, saving model...")
                torch.save(self.model.state_dict(), self.best_model_path)
            else:
                self.epochs_no_improve += 1
                print(f"No improvement in validation loss for {self.epochs_no_improve} epochs.")
            
            if self.epochs_no_improve >= self.patience:
                print("Early stopping triggered.")
                self.early_stop = True
                break

        # Load the best model before testing
        if self.early_stop:
            print("Loading the best model for testing...")
            self.model.load_state_dict(torch.load(self.best_model_path))
        
        self.test()
        
        self.writer.close()

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        self.accuracy_val.reset()
        self.precision_val.reset()
        self.recall_val.reset()
        self.f1_val.reset()
        # Initialize lists to store predictions and true labels for the entire validation phase
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                _, preds = torch.max(outputs, dim=1)
                # Store predictions and true labels
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                
                self.accuracy_val.update(preds, labels)
                self.precision_val.update(preds, labels)
                self.recall_val.update(preds, labels)
                self.f1_val.update(preds, labels)
        
        # Concatenate all predictions and true labels
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        # Compute the confusion matrix
        cm = torchmetrics.functional.confusion_matrix(all_preds, all_labels, num_classes=62, task='multiclass')
        
        # Plot and log the confusion matrix
        fig = self.plot_confusion_matrix(cm.numpy(), ['class_'+str(i) for i in range(62)], epoch, 'validate')
        self.writer.add_figure('Confusion Matrix/validate', fig, global_step=epoch)
        
        avg_loss = total_loss / len(self.val_loader)
        val_accuracy = self.accuracy_val.compute()
        precision = self.precision_val.compute()
        recall = self.recall_val.compute()
        f1_score = self.f1_val.compute()
        print(f'Validation Loss: {avg_loss:.4f}, Validation Accuracy:{val_accuracy:.4f}, Validation Precision: {precision:.4f}, Validation Recall: {recall:.4f}, Validation F1 Score: {f1_score:.4f}')
        
        self.writer.add_scalars('Loss', {'validation': avg_loss}, epoch)
        self.writer.add_scalar('Accuracy/validation', val_accuracy, epoch)
        self.writer.add_scalars('Precision', {'validation': precision}, epoch)
        self.writer.add_scalars('Recall', {'validation': recall}, epoch)
        self.writer.add_scalar('F1 Score/validation', f1_score, epoch)

         # Return the average validation loss for early stopping comparison
        return avg_loss

    def test(self):
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0
        self.accuracy_test.reset()
        self.precision_test.reset()
        self.recall_test.reset()
        self.f1_test.reset()
        # Initialize lists to store predictions and true labels for the entire test phase
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, preds = torch.max(outputs, dim=1)
                # Store predictions and true labels
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                
                self.accuracy_test.update(preds, labels)
                self.precision_test.update(preds, labels)
                self.recall_test.update(preds, labels)
                self.f1_test.update(preds, labels)

        # Concatenate all predictions and true labels
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Compute the confusion matrix
        cm = torchmetrics.functional.confusion_matrix(all_preds, all_labels, num_classes=62, task='multiclass')

        # Plot and log the confusion matrix
        fig = self.plot_confusion_matrix(cm.numpy(), ['class_'+str(i) for i in range(62)], epoch=None, phase='test')
        self.writer.add_figure('Confusion Matrix/test', fig, global_step=0)

        avg_loss = total_loss / len(self.test_loader)
        
        test_accuracy = self.accuracy_test.compute()
        precision = self.precision_test.compute()
        recall = self.recall_test.compute()
        f1_score = self.f1_test.compute()
        print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {precision:.4f}, Test Recall: {recall:.4f}, Test F1 Score: {f1_score:.4f}')

        # Optionally log test metrics to TensorBoard
        self.writer.add_scalars('Loss', {'test': avg_loss}, global_step=0)
        self.writer.add_scalar('Accuracy/test', test_accuracy, global_step=0)
        self.writer.add_scalars('Precision', {'test': precision}, global_step=0)
        self.writer.add_scalars('Recall', {'test': recall}, global_step=0)
        self.writer.add_scalar('F1 Score/test', f1_score, global_step=0)



