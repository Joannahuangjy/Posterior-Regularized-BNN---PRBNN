import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchbnn as bnn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import hydrostats.ens_metrics as em
class BayesianNet(nn.Module):
    def __init__(self):
        super(BayesianNet, self).__init__()
        self.hid1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.01, in_features=11, out_features=64)
        self.hid2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.02, in_features=64, out_features=32)
        self.oupt = bnn.BayesLinear(prior_mu=0, prior_sigma=0.01, in_features=32, out_features=1)
        self.loss_func = torch.nn.MSELoss()
        self.optimizer = None

    def forward(self, x):
        y = F.relu(self.hid1(x))
        z = F.relu(self.hid2(y))
        z = self.oupt(z)
        return z

    def train_model(self, trainloader, derate_loader, off_loader, epochs=400, device='cuda:0',save_path='./model_checkpoint.pth'):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0002)
        s1 = s2 = 1.0
        rho1 = rho2 = 1.0

        for epoch in range(epochs):
            self.train()
            fit_loss_total = complexity_loss_total = 0
            c1_total = c2_total = 0
            
            # Adjust learning rate
            if epoch < 40:
                lr = 0.0002
            elif epoch < 90:
                lr = 0.0001
            else:
                lr = 0.00005
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            for (x, y), (d0, d1), (f0, f1) in zip(trainloader, derate_loader, off_loader):
                x, y, d0, d1, f0, f1 = x.to(device), y.to(device), d0.to(device), d1.to(device), f0.to(device), f1.to(device)
                optimizer.zero_grad()
                
                # Standard prediction and loss
                prediction = self(x)
                loss = self.loss_func(prediction, y)
                fit_loss_total += loss.item()
                complexity_loss = self.kl_loss(self)
                complexity_loss_total += complexity_loss.item()
                
                # Constraint predictions and losses
                derate0_output, derate1_output = self(d0), self(d1)
                off0_output, off1_output = self(f0), self(f1)
                con_derate = torch.mean(torch.clamp(derate1_output - derate0_output, min=1e-6))
                con_off = torch.mean(torch.clamp(off1_output - off0_output, min=1e-6))
                
                c1_total += con_derate.item()
                c2_total += con_off.item()

                total_loss = 50 * (loss + 0.009 * complexity_loss) - con_derate - con_off
                total_loss.backward()
                optimizer.step()

            # Compute average losses and constraints
            avg_fit_loss = fit_loss_total / len(trainloader)
            avg_complexity_loss = complexity_loss_total / len(trainloader)
            c1_avg = c1_total / len(trainloader)
            c2_avg = c2_total / len(trainloader)
            
            # Update s1 and s2 based on constraints
            s1 = max(0, s1 - rho1 * c1_avg)
            s2 = max(0, s2 - rho2 * c2_avg)

            if (epoch+1) % 20 == 0:  # Adjust this condition as per your requirement
                print(f'Epoch {epoch+1}, c1: {c1_avg}, c2: {c2_avg}, s1: {s1}, s2: {s2}, '
                      f'Fit: {avg_fit_loss * 10}, Complexity: {avg_complexity_loss * 0.005}, Total Loss: {avg_fit_loss}')
        save_directory = os.path.dirname(save_path)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory, exist_ok=True)
    
        torch.save(self.state_dict(), save_path)
        print(f'Model saved to {save_path}')
    def load_and_test(self, testloader, save_path='./model_checkpoint.pth', n_samples=50, device='cuda:0'):
        if os.path.exists(save_path):
            self.load_state_dict(torch.load(save_path, map_location=device))
            self.to(device)
            self.eval()
            print(f'Model loaded from {save_path}')
        else:
            print(f'No saved model found at {save_path}. Please check the path.')
            return None, None

        test_results = torch.zeros((len(testloader.dataset), n_samples), device=device)
        test_targets = torch.zeros(len(testloader.dataset), device=device)
        
        with torch.no_grad():
            batch_start = 0
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                batch_size = data.size(0)
                for i in range(n_samples):
                    output = self(data).squeeze()
                    test_results[batch_start:batch_start+batch_size, i] = output
                test_targets[batch_start:batch_start+batch_size] = target
                batch_start += batch_size
        
        return test_results.cpu(), test_targets.cpu()
    
    def evaluate(self, predictions, targets):
        # Convert tensors to numpy arrays for compatibility with scikit-learn metrics
        y_true = targets.cpu().numpy()
        y_pred = predictions.mean(dim=1).cpu().numpy()  # Assuming predictions is a 2D tensor with samples x n_samples

        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f'Mean Squared Error: {mse}')
        print(f'R-squared: {r2}')
        return mse, r2

    
    def visualize_and_save(self, predictions, targets, figure_name, visualization_period=(10, 30)):
        epistemic_uncertainty = predictions.std(dim=1)
        mean_predictions = predictions.mean(dim=1)

        # Make directory if it does not exist
        save_directory = os.path.dirname(figure_name)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        plt.figure(figsize=(30, 8))
        plt.title('BNN With Hard Constraint', fontsize=18)

        # Plot predictions
        plt.plot(mean_predictions.cpu().numpy(), label='Predicted Mean')
        plt.fill_between(np.arange(len(targets)), 
                         (mean_predictions - 3 * epistemic_uncertainty).cpu().numpy(), 
                         (mean_predictions + 3 * epistemic_uncertainty).cpu().numpy(), 
                         alpha=0.3, color='blue', label='Epistemic uncertainty')
        
        # Plot ground truth
        plt.plot(targets.cpu().numpy(), label='Ground Truth', linestyle='--')

        # Add legend
        plt.legend(fontsize=18)
        
        # Save figure
        plt.savefig(figure_name)
        plt.close()
        print(f'Visualization saved as {figure_name}')
