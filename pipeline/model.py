# MiniGPT - Простая имплементация языковой модели на основе Transformer

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import os

from tokenizer import BPETokenizer, create_bpe_tokenizer


class Config:
    """Класс для хранения конфигурации модели и обучения"""

    def __init__(self):
        # Гиперпараметры модели
        self.batch_size = 4
        self.block_size = 512
        self.max_iters = 1000
        self.eval_interval = 100
        self.learning_rate = 3e-4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eval_iters = 20
        self.n_embd = 512
        self.n_head = 8
        self.n_layer = 8
        self.dropout = 0.0
        self.xsa = False

        # Пути к файлам
        self.output_dir = 'outputs'
        self.model_save_path = os.path.join(self.output_dir, 'model.pt')



def apply_rope(x):
    # RoPE
    B, T, D = x.shape

    assert D % 2 == 0

    positions = torch.arange(T, device=x.device)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2, device=x.device).float() / D))
    angles = torch.outer(positions, inv_freq)

    cos = angles.cos()[None, :, :].to(x.dtype)
    sin = angles.sin()[None, :, :].to(x.dtype)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)


class Head(nn.Module):
    """Один блок внимания (attention head)"""

    def __init__(self, config, head_size):
        super().__init__()
        self.xsa = config.xsa
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.dropout_p = config.dropout

    def forward(self, x):
        B, T, C = x.shape
        k = apply_rope(self.key(x))  # (B,T,head_size)
        q = apply_rope(self.query(x))  # (B,T,head_size)
        v = self.value(x)

        # Вычисление весов внимания
        out = F.scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
            is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0
        ).squeeze(1)

        if self.xsa:
            # Projection in FP32 avoids underflow of the normalization epsilon in FP16.
            Vn = F.normalize(v.float(), dim=-1)
            out = (out.float() - (out.float() * Vn).sum(dim=-1, keepdim=True) * Vn).to(v.dtype)
        return out


class MultiHeadAttention(nn.Module):
    """Многоголовное внимание (Multiple heads of self-attention in parallel)"""

    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        if config.n_embd % config.n_head or head_size % 2:
            raise ValueError("n_embd must be divisible by n_head and head_size must be even")
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Конкатенация выходов всех головок внимания
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Полносвязная нейронная сеть после блока внимания"""

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Трансформер блок: внимание + feed forward сеть"""

    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffw = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        # Архитектура с остаточными соединениями
        x = x + self.sa(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x


class MiniGPTModel(nn.Module):
    """Основная модель языкового трансформера"""

    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config

        # Эмбеддинги токенов и позиций
        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
        self.embedding_ln = nn.LayerNorm(config.n_embd)

        # Трансформер блоки
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)  # финальная нормализация
        self.lm_head = nn.Linear(config.n_embd, vocab_size)  # проекция на словарь

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Получение эмбеддингов токенов и позиций
        x = self.embedding_ln(self.token_embedding_table(idx))

        # Проход через трансформер блоки
        x = self.blocks(x)  # (B, T, n_embd)
        x = self.ln_f(x)  # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Вычисление функции потерь, если предоставлены целевые значения
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Генерация новых токенов на основе переданного контекста"""
        # idx начинается как (B, T) массив индексов
        for _ in tqdm(range(max_new_tokens), desc="Генерация токенов"):
            # Обрезаем контекст до последних block_size токенов
            idx_cond = idx[:, -self.config.block_size:]
            # Получаем предсказания
            logits, _ = self(idx_cond)
            # Фокусируемся только на последнем временном шаге
            logits = logits[:, -1, :]  # (B, vocab_size)
            # Применяем softmax для получения вероятностей
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            # Сэмплируем из распределения
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Добавляем сэмплированный индекс к текущей последовательности
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class MiniGPT:
    """Основной класс для работы с моделью - обучение, генерация, сохранение"""

    def __init__(self, config=None, tokenizer=None):
        self.config = config if config else Config()
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.vocab) if tokenizer else None
        self.model = None
        self.train_data = None
        self.val_data = None

        # Создаем директорию для выходных данных, если она не существует
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)

    def load_data(self, dataset):
        """Загрузка и подготовка данных"""
        print("Токенизация train...")

        train_texts = [
            text + "\n" if text.strip() else ""
            for text in dataset["train"]["text"]
        ]

        train_ids = self.tokenizer.encode_parallel(
            train_texts,
            num_workers=4,
            chunksize=200,
        )

        print("Tokenizing validation...")

        val_texts = [
            text + "\n" if text.strip() else ""
            for text in dataset["validation"]["text"]
        ]

        val_ids = self.tokenizer.encode_parallel(
            val_texts,
            num_workers=4,
            chunksize=200,
        )

        self.train_data = torch.tensor(train_ids, dtype=torch.long)
        self.val_data = torch.tensor(val_ids, dtype=torch.long)

        print(f"Train: {len(self.train_data):,} токенов")
        print(f"Validation: {len(self.val_data):,} токенов")
        print(f"Размер словаря: {self.vocab_size:,}")

    def encode(self, s):
        """Преобразование строки в последовательность индексов"""
        return self.tokenizer.encode(s)

    def decode(self, l):
        """Преобразование последовательности индексов в строку"""
        return self.tokenizer.decode(l)

    def get_batch(self, split):
        """Генерация мини-батча данных"""
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.config.block_size, (self.config.batch_size,))
        x = torch.stack([data[i:i + self.config.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.config.block_size + 1] for i in ix])
        x, y = x.to(self.config.device), y.to(self.config.device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self):
        """Оценка функции потерь на тренировочной и валидационной выборках"""
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                X, Y = self.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def build_model(self):
        """Создание модели"""
        print("Создание модели")
        self.model = MiniGPTModel(self.config, self.vocab_size)
        self.model.to(self.config.device)

        # Вывод информации о модели
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Модель создана: {total_params / 1e6:.2f} миллионов параметров")
        return self.model

    def train(self):
        """Обучение модели"""
        print("Начало обучения...")

        # Создаем оптимизатор
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)

        # Основной цикл обучения
        progress_bar = tqdm(range(self.config.max_iters), desc="Обучение")

        for iter in progress_bar:
            # Периодическая оценка на тренировочных и валидационных данных
            if iter % self.config.eval_interval == 0 or iter == self.config.max_iters - 1:
                losses = self.estimate_loss()
                print(f"Шаг {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                progress_bar.set_postfix(train_loss=f"{losses['train']:.4f}", val_loss=f"{losses['val']:.4f}")

            # Выборка мини-батча
            xb, yb = self.get_batch('train')

            # Вычисление потерь
            logits, loss = self.model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        print("Обучение завершено")

        # Сохранение модели
        self.save_model()

    def save_model(self):
        """Сохранение модели"""
        print(f"Сохранение модели в {self.config.model_save_path}")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": vars(self.config),
                "tokenizer_vocab": self.tokenizer.vocab,
                "tokenizer_merges": self.tokenizer.merges,
                "special_tokens": self.tokenizer.special_tokens,
            },
            self.config.model_save_path,
        )

    def load_model(self, path=None):
        """Загрузка модели из файла"""
        path = path or self.config.model_save_path
        print(f"Загрузка модели из {path}")

        checkpoint = torch.load(
            path,
            map_location=self.config.device,
            weights_only=False,
        )

        # Обновление конфигурации
        for key, value in checkpoint['config'].items():
            setattr(self.config, key, value)

        # Загрузка словаря
        self.tokenizer = BPETokenizer(
            vocab=checkpoint["tokenizer_vocab"],
            merges=checkpoint["tokenizer_merges"],
            special_tokens=checkpoint["special_tokens"],
        )
        self.vocab_size = len(self.tokenizer.vocab)

        # Создание и загрузка модели
        self.model = MiniGPTModel(self.config, self.vocab_size)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.config.device)
        self.model.eval()

        print("Модель успешно загружена")

    def generate_text(self, prompt=" ", max_new_tokens=500):
        """Генерация текста на основе промпта"""
        if self.model is None:
            raise ValueError("Модель не инициализирована. Запустите build_model() или load_model() сначала.")

        print(f"Генерация текста с промптом: '{prompt}'")
        context = torch.tensor(self.encode(prompt), dtype=torch.long, device=self.config.device).unsqueeze(0)

        # Генерация
        generated_indices = self.model.generate(context, max_new_tokens)[0].tolist()
        generated_text = self.decode(generated_indices)

        return generated_text
