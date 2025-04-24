import os
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

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
        self.seq_length = seq_length

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

    def generate_text(self, seed_text, preprocessor, max_length=50, temperature=0.8):
        self.eval()
        tokens = preprocessor.tokenize(seed_text)
        input_indices = preprocessor.text_to_indices(tokens)
        seed_length = len(input_indices)
        if len(input_indices) < self.seq_length:
            input_indices = input_indices + [preprocessor.word2idx["<PAD>"]] * (
                self.seq_length - len(input_indices)
            )
        else:
            input_indices = input_indices[-self.seq_length :]
        generated_indices = input_indices[-seed_length:]  # 仅保留种子部分
        hidden = None
        with torch.no_grad():
            for _ in range(max_length):
                x = torch.tensor(
                    [input_indices[-self.seq_length :]], dtype=torch.long
                ).to(device)
                with autocast():
                    output, hidden = self(x, hidden)
                output = output[:, -1, :].squeeze() / temperature
                probs = torch.softmax(output, dim=0)
                next_index = torch.multinomial(probs, 1).item()
                generated_indices.append(next_index)
                input_indices.append(next_index)
        # 清理生成文本：移除<PAD>和<UNK>，截断到max_length
        cleaned_indices = [idx for idx in generated_indices if idx not in [0, 1]]
        # 确保种子部分完整，生成部分不超过max_length
        cleaned_indices = cleaned_indices[: seed_length + max_length]
        return preprocessor.indices_to_text(cleaned_indices)


def predict(
    seed_texts,
    model_path="best_gru_model.pth",
    file_path="bible_kjv.txt",
    output_file="cleaned_predictions.txt",
):
    # 1. 重建词表
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")

    print(f"从 {file_path} 读取文本数据以重建词表...")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    preprocessor = TextPreprocessor(min_freq=2, max_vocab_size=10000)
    tokens = preprocessor.tokenize(text)
    preprocessor.build_vocab(tokens)

    # 2. 初始化模型
    seq_length = 200
    embedding_dim = 256
    hidden_dim = 768
    num_layers = 3
    dropout = 0.7

    model = GRUModel(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        seq_length=seq_length,
    ).to(device)

    # 3. 加载模型参数
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

    print(f"加载模型参数从 {model_path}...")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # 4. 预测种子文本并保存清理结果
    print("\n生成预测文本：")
    results = []
    with open(output_file, "w", encoding="utf-8") as f:
        for seed in seed_texts:
            generated = model.generate_text(
                seed, preprocessor, max_length=50, temperature=0.8
            )
            results.append((seed, generated))
            print(f"\n种子文本: {seed}")
            print(f"生成文本: {generated}")
            print(f"显存使用: {torch.cuda.memory_allocated(device)/1e6:.2f} MB")
            # 写入文件
            f.write(f"Seed: {seed}\n")
            f.write(f"Generated: {generated}\n")
            f.write(f"{'-'*80}\n")

    print(f"\n清理后的预测结果已保存到 {output_file}")
    return results


if __name__ == "__main__":
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

    results = predict(seed_texts)
