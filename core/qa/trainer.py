import torch
from torch.utils.data import DataLoader

from transformers import AdamW

from core.qa.utils import (
    read_squad,
    add_end_idx,
    add_token_positions,
    tokenizer,
    model,
)

train_contexts, train_questions, train_answers = read_squad(
    "squad-style-answers.json"
)

train_encodings = tokenizer(
    train_contexts, train_questions, truncation=True, padding=True
)

add_end_idx(train_answers, train_contexts)
add_token_positions(train_encodings, train_answers)


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        }

    def __len__(self):
        return len(self.encodings.input_ids)


train_dataset = SquadDataset(train_encodings)

device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()
torch.save(model.state_dict(), "core/qa/saved_models/model_weights.pth")
