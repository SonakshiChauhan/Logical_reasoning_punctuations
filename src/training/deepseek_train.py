import os
import sys
import json
import torch
import logging
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from tqdm.auto import tqdm


from huggingface_hub import login
def setup_logging(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logging.basicConfig(filename=os.path.join(output_dir, 'training_log.txt'),
                        level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

class JSONLDataset(Dataset):
    def __init__(self, jsonl_files, tokenizer, max_length=512):
        self.inputs = []
        self.labels = []
        
        logging.info(f"Loading data from {jsonl_files}")
        for file in jsonl_files:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    context = item['theory']
                    for question_id, question_data in item['questions'].items():
                        narrative = context + f" Question: {question_data['question']}?"
                        answer = question_data['answer']
                        label = 0 if answer == True else 1 if answer == False else 2
                        
                        # Tokenize the input
                        encoding = tokenizer.encode_plus(
                            narrative,
                            add_special_tokens=True,
                            max_length=max_length,
                            padding='max_length',
                            truncation=True,
                            return_tensors='pt'
                        )
                        
                        self.inputs.append({
                            'input_ids': encoding['input_ids'].squeeze(),
                            'attention_mask': encoding['attention_mask'].squeeze()
                        })
                        self.labels.append(label)
        
        logging.info(f"Loaded {len(self.inputs)} samples.")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def collate_batch(batch):
    inputs, labels = zip(*batch)
    input_ids = torch.stack([item['input_ids'] for item in inputs])
    attention_mask = torch.stack([item['attention_mask'] for item in inputs])
    labels = torch.tensor(labels, dtype=torch.long)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}, labels

class DeepSeekLoRAClassifier:
    def __init__(self, model_name="deepseek-ai/deepseek-coder-1.3b-base", num_labels=3, 
                 lora_r=16, lora_alpha=32, lora_dropout=0.1):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load the base model
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.base_model, peft_config)
        self.model.print_trainable_parameters()
        
        logging.info(f"Model initialized: {model_name} with LoRA (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})")

    def load_data(self, data_dir, max_length=512):
        train_paths, val_paths, test_paths = [], [], []
        for subdir, dirs, files in os.walk(data_dir):
            for file in files:
                if 'meta-train.jsonl' in file:
                    train_paths.append(os.path.join(subdir, file))
                elif 'meta-dev.jsonl' in file:
                    val_paths.append(os.path.join(subdir, file))
                elif 'meta-test.jsonl' in file:
                    test_paths.append(os.path.join(subdir, file))
        
        logging.info(f"Found {len(train_paths)} train files, {len(val_paths)} validation files, and {len(test_paths)} test files")
        
        train_datasets = [JSONLDataset([path], self.tokenizer, max_length) for path in train_paths]
        val_datasets = [JSONLDataset([path], self.tokenizer, max_length) for path in val_paths]
        test_datasets = [JSONLDataset([path], self.tokenizer, max_length) for path in test_paths]
        
        # Sample information logging
        for i, dataset in enumerate(test_datasets[:1]):
            logging.info(f"Dataset {i} sample information:")
            for j in range(min(3, len(dataset))):
                inputs, label = dataset[j]
                logging.info(f"Sample {j}: Label={label}, Input length={len(inputs['input_ids'])}")
        
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
        test_dataset = ConcatDataset(test_datasets)
        
        logging.info(f"Training samples: {len(train_dataset)}")
        logging.info(f"Validation samples: {len(val_dataset)}")
        logging.info(f"Testing samples: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset

    def train(self, train_dataset, val_dataset, test_dataset, device, epochs=3, batch_size=8, 
              learning_rate=1e-4, weight_decay=0.01, gradient_accumulation_steps=2, output_dir=None):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

        # Set up optimizer - with LoRA we can use higher learning rates
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Calculate total training steps for the scheduler
        total_steps = len(train_loader) // gradient_accumulation_steps * epochs
        warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )

        self.model.to(device)
        
        train_losses, val_losses, test_losses, val_accs, test_accs = [], [], [], [], []
        best_val_acc = 0.0

        for epoch in range(epochs):
            logging.info(f"Starting epoch {epoch+1}/{epochs}")
            self.model.train()
            total_train_loss = 0
            optimizer.zero_grad()
            
            # Training loop with gradient accumulation
            for step, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                labels = labels.to(device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                # Scale the loss according to gradient accumulation steps
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                total_train_loss += loss.item()
                
                # Update weights every gradient_accumulation_steps
                if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            
            avg_train_loss = total_train_loss * gradient_accumulation_steps / len(train_loader)
            train_losses.append(avg_train_loss)
            logging.info(f"Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}")

            # Evaluate on validation set
            val_loss, val_acc = self.evaluate(val_loader, device)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Evaluate on test set
            test_loss, test_acc = self.evaluate(test_loader, device)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            
            # Save the model if validation performance improves
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logging.info(f"New best validation accuracy: {best_val_acc:.4f}")
                best_model_path = os.path.join(output_dir, 'best_model')
                if not os.path.exists(best_model_path):
                    os.makedirs(best_model_path)
                self.model.save_pretrained(best_model_path)
                self.tokenizer.save_pretrained(best_model_path)
                logging.info(f"Best model saved to {best_model_path}")
            
            # Save checkpoint for this epoch
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}')
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            self.model.save_pretrained(checkpoint_path)
            self.tokenizer.save_pretrained(checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")

            # Plot and save metrics after each epoch
            self.plot_metrics(epoch+1, train_losses, val_losses, val_accs, test_losses, test_accs, output_dir)
        
        # Save the final model
        final_model_path = os.path.join(output_dir, 'final_model')
        if not os.path.exists(final_model_path):
            os.makedirs(final_model_path)
        self.model.save_pretrained(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        logging.info(f"Final model saved to {final_model_path}")
        
        logging.info("Training complete. Final metrics saved.")
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_losses': test_losses,
            'val_accs': val_accs,
            'test_accs': test_accs,
            'best_val_acc': best_val_acc
        }

    def evaluate(self, loader, device):
        """Evaluate the model on the given dataloader"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc="Evaluating"):
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                labels = labels.to(device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        logging.info(f"Evaluation - Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy

    def plot_metrics(self, epoch, train_losses, val_losses, val_accs, test_losses, test_accs, output_dir):
        """Plot training and evaluation metrics"""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title(f'Loss Metrics after Epoch {epoch}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(val_accs, label='Validation Accuracy')
        plt.plot(test_accs, label='Test Accuracy')
        plt.title(f'Accuracy Metrics after Epoch {epoch}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt_path = os.path.join(output_dir, f'metrics_epoch_{epoch}.png')
        plt.savefig(plt_path)
        plt.close()
        logging.info(f"Metrics graph saved at {plt_path}")

    def load_and_merge_adapter(self, adapter_path):
        """Load a trained adapter and merge it with the base model for deployment"""
        from peft import PeftModel, PeftConfig
        
        # Load the adapter config
        config = PeftConfig.from_pretrained(adapter_path)
        
        # Load the model with the adapter
        model = PeftModel.from_pretrained(self.base_model, adapter_path)
        
        # Merge adapter weights with base model for faster inference
        merged_model = model.merge_and_unload()
        
        return merged_model

def main():
    parser = argparse.ArgumentParser(description="Finetune DeepSeek model for ProofWriter classification using LoRA.")
    parser.add_argument('--model_name', type=str, default="deepseek-ai/deepseek-coder-1.3b-base", help="DeepSeek model to use ")
    parser.add_argument('--dataset_path', type=str, required=True, help="Root directory containing dataset subdirectories.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to store outputs and checkpoints.")
    parser.add_argument('--epochs', type=int, default=3, help="Number of epochs to train the model.")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help="Number of gradient accumulation steps.")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for optimization.")
    parser.add_argument('--max_length', type=int, default=512, help="Maximum sequence length.")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay for AdamW optimizer.")
    parser.add_argument('--lora_r', type=int, default=16, help="LoRA attention dimension.")
    parser.add_argument('--lora_alpha', type=int, default=32, help="LoRA alpha parameter.")
    parser.add_argument('--lora_dropout', type=float, default=0.1, help="LoRA attention dropout.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    setup_logging(args.output_dir)
    logging.info(f"Arguments: {args}")

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize the classifier
    classifier = DeepSeekLoRAClassifier(
        model_name=args.model_name, 
        num_labels=3,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    # Load datasets
    train_dataset, val_dataset, test_dataset = classifier.load_data(
        args.dataset_path, 
        max_length=args.max_length
    )
    
    # Train the model
    metrics = classifier.train(
        train_dataset, 
        val_dataset, 
        test_dataset, 
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir
    )
    
    # Log final metrics
    logging.info(f"Training completed. Best validation accuracy: {metrics['best_val_acc']:.4f}")

if __name__ == '__main__':
    main()