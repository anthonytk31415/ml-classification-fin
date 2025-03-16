# Best parameters: {
# 'colsample_bytree': 0.9, 
# 'learning_rate': 0.2, 
# 'max_depth': 3, 
# 'n_estimators': 200, 
# 'subsample': 0.8}


# Step 1: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train XGBoost model directly
print("Training standard XGBoost model...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.2,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.9,
    objective='binary:logistic',
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Step 3: Make predictions and evaluate the model
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 4: For PyTorch integration, we can use XGBoost's output as features for a PyTorch model
# or use PyTorch's DataLoader for handling the data going into XGBoost

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Option 1: Use XGBoost with PyTorch DataLoader
print("\nTraining XGBoost with PyTorch DataLoader...")
def train_xgb_with_pytorch_loader(train_loader):
    # Collect all batches
    all_X = []
    all_y = []
    for X_batch, y_batch in train_loader:
        all_X.append(X_batch.numpy())
        all_y.append(y_batch.numpy())
    
    # Concatenate batches
    X_train_combined = np.vstack(all_X)
    y_train_combined = np.concatenate(all_y)
    
    # Train XGBoost
    xgb_model_from_loader = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        objective='binary:logistic',
        random_state=42
    )
    xgb_model_from_loader.fit(X_train_combined, y_train_combined)
    return xgb_model_from_loader

xgb_model_pytorch = train_xgb_with_pytorch_loader(train_loader)

# Option 2: Define a simple PyTorch model that uses XGBoost features
class LoanDefaultClassifier(torch.nn.Module):
    def __init__(self, input_size):
        super(LoanDefaultClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Extract feature importance from XGBoost
print("\nFeature Importance from XGBoost:")
feature_importance = xgb_model.feature_importances_
for i, importance in enumerate(feature_importance):
    print(f"Feature {i}: {importance:.4f}")

# Function to evaluate both models on test data
def evaluate_models(xgb_model, test_loader):
    # Evaluate XGBoost
    X_test_np = X_test
    y_test_np = y_test
    y_pred_xgb = xgb_model.predict(X_test_np)
    xgb_accuracy = accuracy_score(y_test_np, y_pred_xgb)
    print(f"\nXGBoost Final Test Accuracy: {xgb_accuracy:.4f}")
    return xgb_accuracy
evaluate_models(xgb_model, test_loader)

# Additional code for hyperparameter tuning (optional)
def hyperparameter_tuning():
    from sklearn.model_selection import GridSearchCV
    import pandas as pd
    import matplotlib.pyplot as plt
    
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    print("\nPerforming hyperparameter tuning (this may take some time)...")
    grid_search = GridSearchCV(
        estimator=xgb.XGBClassifier(objective='binary:logistic', random_state=42),
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1,
        return_train_score=True  # This will give us training scores too
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best accuracy: {grid_search.best_score_:.4f}")
    
    # Convert results to DataFrame
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Function to analyze parameter impact
    def analyze_parameter_impact(param_name, results_df):
        param_scores = results_df.groupby(f'param_{param_name}').agg({
            'mean_test_score': ['mean', 'std', 'count'],
            'mean_train_score': ['mean', 'std']
        }).round(4)
        
        param_scores.columns = ['Test Mean', 'Test Std', 'Count', 'Train Mean', 'Train Std']
        print(f"\nPerformance analysis for {param_name}:")
        print(param_scores)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.errorbar(param_scores.index, param_scores['Test Mean'], 
                    yerr=param_scores['Test Std'], label='Test Score', 
                    marker='o', capsize=5)
        plt.errorbar(param_scores.index, param_scores['Train Mean'], 
                    yerr=param_scores['Train Std'], label='Train Score', 
                    marker='s', capsize=5)
        plt.title(f'Performance vs {param_name}')
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'hyperparameter_analysis_{param_name}.png')
        plt.close()
        
        return param_scores
    
    # Analyze each parameter
    parameter_analyses = {}
    for param in param_grid.keys():
        parameter_analyses[param] = analyze_parameter_impact(param, results)
    
    # Create interaction analysis for selected parameter pairs
    def analyze_parameter_interaction(param1, param2, results_df):
        pivot_table = pd.pivot_table(
            results_df,
            values='mean_test_score',
            index=f'param_{param1}',
            columns=f'param_{param2}',
            aggfunc='mean'
        ).round(4)
        
        print(f"\nInteraction analysis between {param1} and {param2}:")
        print(pivot_table)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(pivot_table, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Score')
        plt.title(f'Interaction between {param1} and {param2}')
        plt.xlabel(param2)
        plt.ylabel(param1)
        plt.xticks(range(len(pivot_table.columns)), pivot_table.columns)
        plt.yticks(range(len(pivot_table.index)), pivot_table.index)
        
        # Add text annotations to the heatmap
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                plt.text(j, i, f'{pivot_table.iloc[i, j]:.3f}',
                        ha='center', va='center')
        
        plt.savefig(f'interaction_analysis_{param1}_{param2}.png')
        plt.close()
        
        return pivot_table
    
    # Analyze interactions between important parameter pairs
    interaction_analyses = {}
    important_pairs = [
        ('max_depth', 'learning_rate'),
        ('n_estimators', 'learning_rate'),
        ('max_depth', 'n_estimators')
    ]
    
    for param1, param2 in important_pairs:
        interaction_analyses[(param1, param2)] = analyze_parameter_interaction(param1, param2, results)
    
    # Save all results to a comprehensive report
    with open('hyperparameter_tuning_report.txt', 'w') as f:
        f.write("XGBoost Hyperparameter Tuning Report\n")
        f.write("===================================\n\n")
        
        f.write("Best Parameters:\n")
        f.write(f"{grid_search.best_params_}\n\n")
        
        f.write("Best Score:\n")
        f.write(f"{grid_search.best_score_:.4f}\n\n")
        
        f.write("Individual Parameter Analysis:\n")
        f.write("-----------------------------\n")
        for param, analysis in parameter_analyses.items():
            f.write(f"\n{param}:\n")
            f.write(analysis.to_string())
            f.write("\n")
        
        f.write("\nParameter Interaction Analysis:\n")
        f.write("------------------------------\n")
        for (param1, param2), analysis in interaction_analyses.items():
            f.write(f"\n{param1} vs {param2}:\n")
            f.write(analysis.to_string())
            f.write("\n")
    
    return grid_search.best_estimator_

# Uncomment to run hyperparameter tuning
best_xgb_model = hyperparameter_tuning()