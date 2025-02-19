# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pickle

# %%
# Load và xem dữ liệu
full_data = pd.read_csv('amazon.csv')
print("Kích thước dữ liệu ban đầu:", full_data.shape)
full_data.head()

# %%
# Loại bỏ các cột không cần thiết
columns_to_drop = [
    'discounted_price', 'actual_price', 'discount_percentage',
    'about_product', 'user_name', 'review_id', 'review_title',
    'review_content'
]

data = full_data.drop(columns=columns_to_drop)
print("\nKích thước dữ liệu sau khi loại bỏ các cột:", data.shape)
data.head()

# %%
# Kiểm tra phân phối rating
plt.figure(figsize=(15, 8))
sns.countplot(data=data.sort_values(by='rating'), x='rating')
plt.title('Distribution of Ratings')
plt.xlabel('Rating Value')
plt.ylabel('Count')
plt.show()

# %%
# Xử lý dữ liệu
data = data.dropna()
data = data[data.rating != '|']
data['rating'] = data['rating'].astype(float)

# Encode user_id và product_id
product_id_encoder = LabelEncoder()
user_id_encoder = LabelEncoder()

data['product_id'] = product_id_encoder.fit_transform(data['product_id'])
data['user_id'] = user_id_encoder.fit_transform(data['user_id'])

print("\nThông tin dữ liệu sau khi xử lý:")
print(data.info())

# %%
# Deep Matrix Factorization Model
class DeepMatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, factors=[64, 32, 16, 8]):
        super(DeepMatrixFactorization, self).__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, factors[0])
        self.item_embedding = nn.Embedding(n_items, factors[0])
        
        # User tower
        self.user_tower = nn.Sequential(
            nn.Linear(factors[0], factors[1]),
            nn.ReLU(),
            nn.BatchNorm1d(factors[1]),
            nn.Dropout(0.2),
            nn.Linear(factors[1], factors[2]),
            nn.ReLU(),
            nn.BatchNorm1d(factors[2]),
            nn.Dropout(0.2),
            nn.Linear(factors[2], factors[3])
        )
        
        # Item tower
        self.item_tower = nn.Sequential(
            nn.Linear(factors[0], factors[1]),
            nn.ReLU(),
            nn.BatchNorm1d(factors[1]),
            nn.Dropout(0.2),
            nn.Linear(factors[1], factors[2]),
            nn.ReLU(),
            nn.BatchNorm1d(factors[2]),
            nn.Dropout(0.2),
            nn.Linear(factors[2], factors[3])
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, user_ids, item_ids):
        # Get embeddings
        user_embedded = self.user_embedding(user_ids)
        item_embedded = self.item_embedding(item_ids)
        
        # Pass through towers
        user_vector = self.user_tower(user_embedded)
        item_vector = self.item_tower(item_embedded)
        
        # Normalize embeddings
        user_vector = nn.functional.normalize(user_vector, p=2, dim=1)
        item_vector = nn.functional.normalize(item_vector, p=2, dim=1)
        
        # Compute prediction
        prediction = torch.sum(user_vector * item_vector, dim=1)
        return torch.sigmoid(prediction)

# %%
# Training Framework
class DMFTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            user_ids, item_ids, ratings = batch
            
            # Forward pass
            predictions = self.model(user_ids, item_ids)
            loss = self.criterion(predictions, ratings)
            
            # Accuracy
            predicted_labels = (predictions >= 0.5).float()
            correct_predictions += (predicted_labels == ratings).sum().item()
            total_samples += ratings.size(0)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader), correct_predictions / total_samples

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                user_ids, item_ids, ratings = batch
                predictions = self.model(user_ids, item_ids)
                loss = self.criterion(predictions, ratings)
                
                predicted_labels = (predictions >= 0.5).float()
                correct_predictions += (predicted_labels == ratings).sum().item()
                total_samples += ratings.size(0)
                
                total_loss += loss.item()
                
        return total_loss / len(val_loader), correct_predictions / total_samples

# %%
# Dataset
class RecommenderDataset(torch.utils.data.Dataset):
    def __init__(self, df, rating_range=5.0):
        self.users = torch.LongTensor(df['user_id'].values)
        self.items = torch.LongTensor(df['product_id'].values)
        self.ratings = torch.FloatTensor(df['rating'].values) / rating_range
        
    def __len__(self):
        return len(self.users)
        
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

# %%
# Split data
Train, Test = train_test_split(data, test_size=0.2, random_state=42)
print("Train size:", len(Train))
print("Test size:", len(Test))

# %%
# Create datasets and dataloaders
train_dataset = RecommenderDataset(Train)
test_dataset = RecommenderDataset(Test)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=128
)

# %%
# Initialize and train model
n_users = len(data['user_id'].unique())
n_items = len(data['product_id'].unique())
model = DeepMatrixFactorization(n_users, n_items)
trainer = DMFTrainer(model)

# Training loop
num_epochs = 50
train_losses = []
test_losses = []
train_accs = []
test_accs = []

print("Starting training...")
for epoch in range(num_epochs):
    # Train
    train_loss, train_acc = trainer.train_epoch(train_loader)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Evaluate
    test_loss, test_acc = trainer.evaluate(test_loader)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    if (epoch + 1) % 5 == 0:
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

# %%
# Plot training results
plt.figure(figsize=(12, 4))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(test_accs, label='Test Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# %%
# Save model and training info
save_info = {
    'model_state_dict': model.state_dict(),
    'train_losses': train_losses,
    'test_losses': test_losses,
    'train_accs': train_accs,
    'test_accs': test_accs,
    'n_users': n_users,
    'n_items': n_items,
    'user_encoder': user_id_encoder,
    'product_encoder': product_id_encoder
}

with open('dmf_model.pkl', 'wb') as f:
    pickle.dump(save_info, f)
print("\nModel saved to dmf_model.pkl")

# %%
def get_recommendations_with_details(model=model, data=data, n_recommendations=5, product_id=0):
    """Get recommendations for a product"""
    model.eval()
    with torch.no_grad():
        # Get original product ID
        original_product_id = product_id_encoder.inverse_transform([product_id])[0]
        
        # Get all product embeddings
        all_products = torch.arange(len(data['product_id'].unique()))
        target_product = torch.tensor([product_id])
        
        # Get predictions
        similarities = model(
            target_product.repeat(len(all_products)), 
            all_products
        ).numpy()
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[-n_recommendations-1:][::-1]
        # Remove the input product if it's in the recommendations
        top_indices = top_indices[top_indices != product_id][:n_recommendations]
        
        recommendations = []
        for idx in top_indices:
            recommended_original_id = product_id_encoder.inverse_transform([idx])[0]
            recommended_product_details = full_data[full_data['product_id'] == recommended_original_id].iloc[0]
            
            recommendation_info = {
                'product_id': recommended_original_id,
                'product_name': recommended_product_details['product_name'],
                'category': recommended_product_details['category'],
                'rating': recommended_product_details['rating'],
                'rating_count': recommended_product_details['rating_count'],
                'img_link': recommended_product_details['img_link'],
                'product_link': recommended_product_details['product_link'],
                'similarity': similarities[idx]
            }
            recommendations.append(recommendation_info)
            
    return recommendations

# Example usage
if __name__ == "__main__":
    # Load model and make recommendations
    recommendations = get_recommendations_with_details(model, data, 5, 21)
    
    # Print recommendations
    print("\nRecommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\nRecommendation {i}:")
        print(f"Product Name: {rec['product_name']}")
        print(f"Category: {rec['category']}")
        print(f"Rating: {rec['rating']} ({rec['rating_count']} ratings)")
        print(f"Similarity Score: {rec['similarity']:.4f}")