import os
import random
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast  # 导入混合精度模块
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


class TextPreprocessor:
    def __init__(self, min_freq=2, max_vocab_size=10000):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        self.vocab_size = 0

    def tokenize(self, text):
        text = text.lower().replace("\n", " ")
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"([.,!?;:])", r" \1 ", text)
        tokens = [t for t in text.split() if t]
        return tokens

    def build_vocab(self, tokens):
        self.word_counts = Counter(tokens)
        sorted_words = sorted(
            self.word_counts.items(), key=lambda x: x[1], reverse=True
        )
        if self.max_vocab_size:
            sorted_words = sorted_words[: self.max_vocab_size - 2]
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        for word, count in sorted_words:
            if count >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        self.vocab_size = len(self.word2idx)
        print(f"词汇表大小: {self.vocab_size}")
        return self.word2idx

    def text_to_indices(self, tokens):
        return [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]

    def indices_to_text(self, indices):
        return " ".join([self.idx2word.get(idx, "<UNK>") for idx in indices])


class TextDataset(Dataset):
    def __init__(self, sequences, seq_length, step_size=1):
        self.sequences = sequences
        self.seq_length = seq_length
        self.step_size = step_size
        self.indices = self._create_indices()

    def _create_indices(self):
        indices = []
        for seq in self.sequences:
            if len(seq) <= self.seq_length:
                continue
            for i in range(0, len(seq) - self.seq_length, self.step_size):
                indices.append(i)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        seq = self.sequences[0]
        i = self.indices[idx]
        input_seq = seq[i : i + self.seq_length]
        target_seq = seq[i + 1 : i + self.seq_length + 1]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(
            target_seq, dtype=torch.long
        )


class GRUModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        dropout=0.7,
        seq_length=200,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length  # 与 main 的 seq_length 同步

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        gru_out, hidden = self.gru(embedded, hidden)
        gru_out = self.dropout(gru_out)
        output = self.fc(gru_out)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)

    def generate_text(self, seed_text, preprocessor, max_length=100, temperature=1.0):
        self.eval()
        tokens = preprocessor.tokenize(seed_text)
        input_indices = preprocessor.text_to_indices(tokens)
        if len(input_indices) < self.seq_length:
            input_indices = input_indices + [preprocessor.word2idx["<PAD>"]] * (
                self.seq_length - len(input_indices)
            )
        else:
            input_indices = input_indices[-self.seq_length :]
        generated_indices = input_indices.copy()
        hidden = None
        with torch.no_grad():
            for _ in range(max_length):
                x = torch.tensor(
                    [generated_indices[-self.seq_length :]], dtype=torch.long
                ).to(device)
                with autocast():  # 混合精度生成
                    output, hidden = self(x, hidden)
                output = output[:, -1, :].squeeze() / temperature
                probs = torch.softmax(output, dim=0)
                next_index = torch.multinomial(probs, 1).item()
                generated_indices.append(next_index)
        return preprocessor.indices_to_text(generated_indices)


def train(model, train_loader, criterion, optimizer, epoch, clip=5.0):
    model.train()
    total_loss = 0
    scaler = GradScaler()  # 初始化 GradScaler
    for i, (inputs, targets) in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch}", mininterval=1.0)
    ):
        inputs, targets = inputs.to(device), targets.to(device)
        hidden = model.init_hidden(inputs.size(0))
        optimizer.zero_grad()
        with autocast():  # 启用混合精度
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        scaler.scale(loss).backward()  # 缩放梯度
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)  # 更新优化器
        scaler.update()  # 更新缩放因子
        total_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(
                f"Epoch {epoch}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}"
            )
            print(f"显存使用: {torch.cuda.memory_allocated(device)/1e6:.2f} MB")
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = model.init_hidden(inputs.size(0))
            with autocast():  # 混合精度验证
                outputs, hidden = model(inputs, hidden)
                loss = criterion(
                    outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1)
                )
            total_loss += loss.item()
    return total_loss / len(val_loader)


def main(file_path="bible_kjv.txt", seed=42):
    seq_length = 200
    embedding_dim = 256
    hidden_dim = 768
    num_layers = 3
    dropout = 0.7
    batch_size = 128  # 混合精度支持更大批次
    learning_rate = 0.0003
    num_epochs = 20
    step_size = 5
    patience = 5

    torch.manual_seed(seed)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")

    print(f"从 {file_path} 读取文本数据...")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("预处理文本...")
    preprocessor = TextPreprocessor(min_freq=2, max_vocab_size=10000)
    tokens = preprocessor.tokenize(text)
    preprocessor.build_vocab(tokens)
    indices = preprocessor.text_to_indices(tokens)

    train_size = int(0.8 * len(indices))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = TextDataset([train_indices], seq_length, step_size)
    val_dataset = TextDataset([val_indices], seq_length, step_size)

    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print("构建GRU模型...")
    model = GRUModel(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        seq_length=seq_length,
    ).to(device)

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=preprocessor.word2idx["<PAD>"])
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=5e-4
    )  # 增强权重衰减
    scheduler = CosineAnnealingLR(optimizer, T_max=20)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_model_path = "best_gru_model.pth"
    patience_counter = 0

    print(f"开始训练，总共 {num_epochs} 轮...")
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, epoch)
        val_loss = validate(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"模型已保存到 {best_model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
        scheduler.step()
        if patience_counter >= patience:
            print(f"早停：验证损失连续 {patience} 轮未改善")
            break

    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n生成文本示例:")
    seed_texts = [
        "In the beginning God created",
        "The Lord is my shepherd",
        "Blessed are the meek for",
        "Thou shalt not steal",
        "Fear not for I am",
        "Jesus wept and said",
        "Let there be light",
        "The wages of sin is",
        "Sing unto the Lord",
        "My God why hast thou",
        "And God said Let the waters bring forth abundantly the moving creature",
        "The Lord is my light and my salvation whom shall I fear",
        "For God so loved the world that he gave his only Son",
        "Thou shalt love thy neighbour as thyself for this is the law",
        "Behold I stand at the door and knock if any man hear",
        "The Spirit of the Lord is upon me because he hath anointed",
        "And Jesus said unto them Follow me and I will make you",
        "O give thanks unto the Lord for he is good and his mercy",
        "If my people which are called by my name shall humble themselves",
        "But seek ye first the kingdom of God and his righteousness and",
        "In the beginning was the Word and the Word was with God and the Word was God The same was in the beginning",
        "The Lord said unto Moses Come up to me into the mount and be there and I will give thee tables of stone",
        "O Lord our Lord how excellent is thy name in all the earth who hast set thy glory above the heavens Out of the mouth",
        "For I know the thoughts that I think toward you saith the Lord thoughts of peace and not of evil to give you an expected end",
        "And when the day of Pentecost was fully come they were all with one accord in one place And suddenly there came a sound",
        "Blessed is the man that walketh not in the counsel of the ungodly nor standeth in the way of sinners nor sitteth in the seat of the scornful",
        "And it came to pass that as Jesus sat at meat in the house behold many publicans and sinners came and sat down with him",
        "Hear O Israel The Lord our God is one Lord And thou shalt love the Lord thy God with all thine heart and with all",
        "Thus saith the Lord of hosts Consider your ways Go up to the mountain and bring wood and build the house and I will take pleasure",
        "And the angel said unto her Fear not Mary for thou hast found favour with God And behold thou shalt conceive in thy womb and bring forth",
    ]
    for seed in seed_texts[:5]:  # 仅展示前5个以节省空间
        generated = model.generate_text(
            seed, preprocessor, max_length=50, temperature=0.8
        )
        print(f"\n种子文本: {seed}")
        print(f"生成文本: {generated}")

    return model, preprocessor


if __name__ == "__main__":
    model, preprocessor = main()
