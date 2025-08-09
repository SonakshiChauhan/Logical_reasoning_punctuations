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
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, GPT2Config, GPT2ForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

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
    def __init__(self, jsonl_files, tokenizer, block_size):
        self.samples = []
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
                        # print(answer, type(answer))
                        label = 0 if answer == True else 1 if answer == False else 2
                        tokens = tokenizer.encode(narrative, add_special_tokens=True)[:block_size]
                        if tokens:
                            self.samples.append(torch.tensor(tokens, dtype=torch.long))
                            self.labels.append(label)
        # print(self.samples)
        # print(self.labels)
        logging.info(f"Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

def collate_batch(batch):
    input_ids, labels = zip(*batch)
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return input_ids_padded, labels

class GPT2SequenceClassifier:
    def __init__(self, model_name="gpt2", num_labels=3):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Often, eos_token can be used as pad_token
        config = GPT2Config.from_pretrained(model_name, num_labels=num_labels, pad_token_id=self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token))
        self.model = GPT2ForSequenceClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
        logging.info("Model initialized")

    def load_data(self, data_dir, block_size=512):
        train_paths, val_paths, test_paths = [], [], []
        for subdir, dirs, files in os.walk(data_dir):
            for file in files:
                if 'meta-train.jsonl' in file:
                    train_paths.append(os.path.join(subdir, file))
                elif 'meta-dev.jsonl' in file:
                    val_paths.append(os.path.join(subdir, file))
                elif 'meta-test.jsonl' in file:
                    test_paths.append(os.path.join(subdir, file))
        
        train_datasets = [JSONLDataset([path], self.tokenizer, block_size) for path in train_paths]
        val_datasets = [JSONLDataset([path], self.tokenizer, block_size) for path in val_paths]
        test_datasets = [JSONLDataset([path], self.tokenizer, block_size) for path in test_paths]
        for dataset in test_datasets:
            print(f"Dataset: {dataset}")
            for i in range(min(3, len(dataset))):  # Print up to 3 samples from each dataset
                print(f"Sample {i}: {dataset[i]}")
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
        test_dataset = ConcatDataset(test_datasets)
        print(test_dataset)
        return train_dataset, val_dataset, test_dataset

    def train(self, train_dataset, val_dataset, test_dataset, device, epochs=3, batch_size=8, learning_rate=1e-5, output_dir=None):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs)
        loss_function = nn.CrossEntropyLoss()

        train_losses, val_losses, test_losses, val_accs, test_accs = [], [], [], [], []

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(input_ids=inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_train_loss += loss.item()
            train_losses.append(total_train_loss / len(train_loader))

            val_loss, val_acc = self.evaluate(val_loader, device)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            test_loss, test_acc = self.evaluate(test_loader, device)
            test_losses.append(test_loss)
            test_accs.append(test_acc)

            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}')
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            self.model.save_pretrained(checkpoint_path)
            self.tokenizer.save_pretrained(checkpoint_path)

            self.plot_metrics(epoch+1, train_losses, val_losses, val_accs, test_losses, test_accs, output_dir)
        
        final_model_path = os.path.join(output_dir, 'final_model')
        if not os.path.exists(final_model_path):
            os.makedirs(final_model_path)
        self.model.save_pretrained(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        logging.info("Training complete. Final metrics saved.")

    def evaluate(self, loader, device):
        self.model.eval()
        total_loss, total_accuracy = 0, 0
        loss_function = nn.CrossEntropyLoss()
        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc="Evaluating"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(input_ids=inputs, labels=labels)
                loss = outputs.loss
                preds = torch.argmax(outputs.logits, dim=1)
                accuracy = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
                total_loss += loss.item()
                total_accuracy += accuracy
        avg_loss = total_loss / len(loader)
        avg_accuracy = total_accuracy / len(loader)
        logging.info(f"Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}")
        return avg_loss, avg_accuracy

    def plot_metrics(self, epoch, train_losses, val_losses, val_accs, test_losses, test_accs, output_dir):
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

def main():
    parser = argparse.ArgumentParser(description="Train a GPT-2 model for sequence classification.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Root directory containing dataset subdirectories.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to store outputs and checkpoints.")
    parser.add_argument('--epochs', type=int, default=3, help="Number of epochs to train the model.")
    args = parser.parse_args()

    setup_logging(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Set the padding token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = GPT2SequenceClassifier(model_name='gpt2', num_labels=3)
    classifier.model.to(device)

    train_dataset, val_dataset, test_dataset = classifier.load_data(args.dataset_path)
    classifier.train(train_dataset, val_dataset, test_dataset, device=device, epochs=args.epochs, batch_size=8, learning_rate=1e-5, output_dir=args.output_dir)

if __name__ == '__main__':
    main()