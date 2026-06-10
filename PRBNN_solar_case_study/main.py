import torch

try:
    from .data_utils import load_data, preprocess_data, normalize_data, split_data, create_dataloaders, cons_data
    from .model import BayesianNet
except ImportError:
    from data_utils import load_data, preprocess_data, normalize_data, split_data, create_dataloaders, cons_data
    from model import BayesianNet

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load and preprocess data
    raw_data = load_data(file_path="Know_regression.pkl")
    processed_data = preprocess_data(raw_data)
    normalized_data = normalize_data(processed_data)
    train_data, valid_data, test_data = split_data(normalized_data)
    
    # Create data loaders
    train_loader, _ , test_loader = create_dataloaders(train_data, valid_data, test_data, batch_size=32, device=device)
    derate_loader, off_loader = cons_data(train_data, device=device)  # Adjust as needed
    
    # Initialize model
    model = BayesianNet().to(device)
    
    # Train model
    model.train_model(train_loader, derate_loader, off_loader, epochs=400, device=device)
    
    # Evaluate model
    predictions, targets = model.load_and_test(test_loader, device=device)
    mse, r2 = model.evaluate(predictions, targets)
    print('The MSE is ', mse,'. The r2 is ',r2)
    # Visualize results
    model.visualize_and_save(predictions=predictions, targets=targets, figure_name='./results_with_hard_constraint.png')

if __name__ == "__main__":
    main()
