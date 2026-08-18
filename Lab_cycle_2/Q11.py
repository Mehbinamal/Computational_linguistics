import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.optim as optim
import numpy as np

# ==========================================
# 1. Configuration & Confusion Sets
# ==========================================
CONFUSION_SETS = {
    0: ["write", "right", "rite"],
    1: ["peace", "piece"],
    2: ["their", "there", "they're"]
}

# Reverse lookup map: word -> set_id
WORD_TO_SET = {}
for set_id, words in CONFUSION_SETS.items():
    for word in words:
        WORD_TO_SET[word.lower()] = set_id

MODEL_NAME = 'bert-base-uncased'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. Neural Classifier Architecture
# ==========================================
class NeuralHomophoneClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super(NeuralHomophoneClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.2)
        
        # Classification heads for each specific confusion set
        self.head_set0 = nn.Linear(self.bert.config.hidden_size, len(CONFUSION_SETS[0]))
        self.head_set1 = nn.Linear(self.bert.config.hidden_size, len(CONFUSION_SETS[1]))
        self.head_set2 = nn.Linear(self.bert.config.hidden_size, len(CONFUSION_SETS[2]))

    def forward(self, input_ids, attention_mask, target_indices, set_ids):
        # Extract contextual representations from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_dim)

        # Gather target token vectors based on target_indices
        batch_size = input_ids.size(0)
        target_reprs = sequence_output[torch.arange(batch_size), target_indices]
        target_reprs = self.dropout(target_reprs)

        # Compute logits dynamically per set
        logits_list = []
        for i in range(batch_size):
            sid = set_ids[i].item()
            repr_i = target_reprs[i].unsqueeze(0)
            if sid == 0:
                logits = self.head_set0(repr_i)
            elif sid == 1:
                logits = self.head_set1(repr_i)
            elif sid == 2:
                logits = self.head_set2(repr_i)
            logits_list.append(logits)

        return logits_list

# ==========================================
# 3. Dataset & Data Processor
# ==========================================
class HomophoneDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, target_idx, correct_word = self.data[idx]
        tokens = sentence.split()
        
        set_id = WORD_TO_SET[correct_word.lower()]
        candidates = CONFUSION_SETS[set_id]
        label = candidates.index(correct_word.lower())

        # Tokenize full sentence with BERT
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        # Map word-level index to wordpiece token-level index
        word_ids = encoding.word_ids(batch_index=0)
        token_target_idx = 0
        for t_idx, w_id in enumerate(word_ids):
            if w_id == target_idx:
                token_target_idx = t_idx
                break

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'target_idx': torch.tensor(token_target_idx, dtype=torch.long),
            'set_id': torch.tensor(set_id, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ==========================================
# 4. Inference & Correction Pipeline
# ==========================================
class NeuralHomophoneCorrector:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def correct_text(self, text):
        tokens = text.split()
        corrected_tokens = list(tokens)

        for i, word in enumerate(tokens):
            clean_word = word.lower().strip(".,!?;:")
            if clean_word in WORD_TO_SET:
                set_id = WORD_TO_SET[clean_word]
                candidates = CONFUSION_SETS[set_id]

                # Prepare input
                encoding = self.tokenizer(
                    tokens,
                    is_split_into_words=True,
                    padding='max_length',
                    truncation=True,
                    max_length=64,
                    return_tensors="pt"
                )

                word_ids = encoding.word_ids(batch_index=0)
                token_target_idx = next(t_idx for t_idx, w_id in enumerate(word_ids) if w_id == i)

                input_ids = encoding['input_ids'].to(DEVICE)
                attention_mask = encoding['attention_mask'].to(DEVICE)
                target_idx_tensor = torch.tensor([token_target_idx], dtype=torch.long).to(DEVICE)
                set_id_tensor = torch.tensor([set_id], dtype=torch.long).to(DEVICE)

                # Neural Prediction
                with torch.no_grad():
                    logits_list = self.model(input_ids, attention_mask, target_idx_tensor, set_id_tensor)
                    pred_idx = torch.argmax(logits_list[0], dim=1).item()
                    predicted_word = candidates[pred_idx]

                # Preserve Capitalization
                if word[0].isupper():
                    predicted_word = predicted_word.capitalize()
                
                # Re-attach punctuation
                if word[-1] in ".,!?;:":
                    predicted_word += word[-1]

                corrected_tokens[i] = predicted_word

        return " ".join(corrected_tokens)

# ==========================================
# 5. Training & Execution Script
# ==========================================
if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Sample dataset: (Sentence, target_word_index, correct_ground_truth_word)
    training_data = [
        ("Please write down the correct response.", 1, "write"),
        ("Please right down the correct response.", 1, "write"),
        ("The books left on the shelf lost their covers.", 7, "their"),
        ("The books left on the shelf lost there covers.", 7, "their"),
        ("They are over there near the counter.", 3, "there"),
        ("I would like a piece of cake.", 4, "piece"),
        ("They want world peace for all.", 3, "peace"),
    ]

    # Initialize Dataset and DataLoader
    dataset = HomophoneDataset(training_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Model & Optimization Setup
    model = NeuralHomophoneClassifier().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    print("Training Neural Classifier...")
    model.train()
    for epoch in range(5):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            target_indices = batch['target_idx'].to(DEVICE)
            set_ids = batch['set_id'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            logits_list = model(input_ids, attention_mask, target_indices, set_ids)

            # Compute cross-entropy loss across batch elements
            loss = 0.0
            for i in range(len(logits_list)):
                loss += criterion(logits_list[i], labels[i].unsqueeze(0))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/5 | Loss: {total_loss / len(dataloader):.4f}")

    # Test Evaluation Pipeline
    corrector = NeuralHomophoneCorrector(model, tokenizer)

    test_sentences = [
        "The students left there books over their.",
        "Can you right a piece for the newspaper?",
        "I hope we find peace and a peace of cake."
    ]

    print("\n--- Neural Model Correction Results ---")
    for test_sentence in test_sentences:
        corrected = corrector.correct_text(test_sentence)
        print(f"Input    : {test_sentence}")
        print(f"Corrected: {corrected}\n")