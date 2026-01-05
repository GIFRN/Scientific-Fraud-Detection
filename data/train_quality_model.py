"""
Train the quality evaluation model using generated data.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from tqdm import tqdm

from quality_model import ModelQualityPredictor, ArgumentDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_model(train_loader, val_loader, model, optimizer, criterion, epochs=10):
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
              f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model, best_val_acc


def main():
    # Load data
    data_file = "quality_training_data.csv"
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found!")
        print("Please run generate_quality_data.py first to create training data.")
        return
    
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} samples")
    print(f"  High quality: {(df['label'] == 1).sum()}")
    print(f"  Low quality: {(df['label'] == 0).sum()}")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), 
        df['label'].tolist(), 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']
    )
    
    # Initialize tokenizer and model
    print("\nInitializing model...")
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    base_model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    model = ModelQualityPredictor(base_model).to(device)
    
    # Create datasets
    train_dataset = ArgumentDataset(train_texts, train_labels, tokenizer)
    val_dataset = ArgumentDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    print("\nTraining...")
    model, best_acc = train_model(train_loader, val_loader, model, optimizer, criterion, epochs=5)
    
    # Save model
    output_path = "../quality_evaluation_model.pt"
    torch.save(model.state_dict(), output_path)
    print(f"\n✓ Model saved to {output_path}")
    print(f"✓ Best validation accuracy: {best_acc:.4f}")
    
    # Final evaluation
    print("\nFinal evaluation on validation set:")
    model.eval()
    val_preds, val_labels_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask)
            val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            val_labels_list.extend(labels.cpu().numpy())
    
    print(classification_report(val_labels_list, val_preds, 
                                target_names=['Low Quality', 'High Quality']))


if __name__ == "__main__":
    main()

