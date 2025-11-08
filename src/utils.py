import torch
import numpy as np
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
from collections import defaultdict
from datasets import Dataset, DatasetDict  # 仅使用数据集格式工具，不联网
from config import Config

config = Config()

# ===================== 离线自定义Tokenizer（支持批量编码，无网络依赖）=====================
class OfflineTokenizer:
    def __init__(self, vocab_size=32000):
        self.special_tokens = {
            "<pad>": 0,  # padding token
            "<s>": 1,   # 起始token
            "</s>": 2,  # 结束token
            "<unk>": 3  # 未知token
        }
        self.vocab_size = vocab_size
        self.word2id = self.special_tokens.copy()
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.vocab_built = False

    def build_vocab(self, texts):
        word_count = defaultdict(int)
        for text in texts:
            for word in text.lower().split():
                word_count[word] += 1
        sorted_words = sorted(word_count.keys(), key=lambda x: word_count[x], reverse=True)
        selected_words = sorted_words[:self.vocab_size - len(self.special_tokens)]
        for word in selected_words:
            self.word2id[word] = len(self.word2id)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.vocab_built = True
        print(f"离线词汇表构建完成，共{len(self.word2id)}个token")

    def encode(self, texts, padding="max_length", truncation=True, max_length=50):
        if not self.vocab_built:
            raise ValueError("请先调用build_vocab构建词汇表！")
        
        input_ids_list = []
        for text in texts:
            words = text.lower().split()
            if truncation and len(words) > max_length:
                words = words[:max_length]
            input_ids = [self.word2id.get(word, self.word2id["<unk>"]) for word in words]
            if padding == "max_length":
                pad_len = max_length - len(input_ids)
                input_ids += [self.word2id["<pad>"]] * pad_len
            input_ids_list.append(input_ids)
        
        return {"input_ids": torch.tensor(input_ids_list, dtype=torch.long)}

    def decode(self, input_ids, skip_special_tokens=True):
        if input_ids.ndim == 2:
            return [self.decode(single_ids, skip_special_tokens) for single_ids in input_ids]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.cpu().numpy().tolist()
        words = [self.id2word.get(id, "<unk>") for id in input_ids]
        if skip_special_tokens:
            words = [w for w in words if w not in self.special_tokens]
        return " ".join(words)

    @property
    def pad_token_id(self):
        return self.special_tokens["<pad>"]

# 初始化离线tokenizer（此处不调用set_seed，移到后面）
tokenizer = OfflineTokenizer(vocab_size=32000)

# ===================== 工具函数（先定义set_seed，再调用）=====================
def set_seed():
    """固定随机种子（确保可重现性）"""
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True

def create_padding_mask(x):
    """创建Padding掩码：padding位置（0）设为False，其他设为True"""
    padding_mask = (x != tokenizer.pad_token_id).unsqueeze(-2)  # [batch, 1, seq_len]
    return padding_mask

def create_future_mask(x):
    """创建Future掩码：掩盖未来token，下三角为True（基于输入x的实际长度）"""
    seq_len = x.shape[1]  # 动态获取输入序列长度（如49）
    future_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).bool()  # [49, 49]
    padding_mask = create_padding_mask(x)  # [batch, 1, 49]（基于x的padding）
    future_mask = future_mask & padding_mask  # [batch, 49, 49]（与padding掩码结合）
    return future_mask

def preprocess_data():
    """完全离线数据预处理：生成随机模拟数据（英德翻译对）"""
    # 关键修复：在生成随机数据前调用set_seed（此时set_seed已定义）
    set_seed()
    
    # 1. 生成随机模拟数据（替代IWSLT2017，无网络依赖）
    import random
    # random.seed(config.seed)  # 无需重复，set_seed已固定全局种子
    
    # 生成英文句子（模拟源语言）
    def generate_english(num):
        nouns = ["cat", "dog", "book", "computer", "tree", "house", "car", "student"]
        verbs = ["eats", "runs", "reads", "writes", "grows", "builds", "drives", "studies"]
        adjectives = ["red", "fast", "smart", "big", "small", "happy", "old", "new"]
        sentences = []
        for i in range(num):
            sent = f"{random.choice(adjectives)} {random.choice(nouns)} {random.choice(verbs)} ."
            sentences.append(sent)
        return sentences
    
    # 生成德文句子（模拟目标语言）
    def generate_german(num):
        nouns = ["katze", "hund", "buch", "computer", "baum", "haus", "auto", "student"]
        verbs = ["isst", "läuft", "liest", "schreibt", "wächst", "baut", "fährt", "studiert"]
        adjectives = ["rot", "schnell", "klug", "groß", "klein", "glücklich", "alt", "neu"]
        sentences = []
        for i in range(num):
            sent = f"{random.choice(adjectives)} {random.choice(nouns)} {random.choice(verbs)} ."
            sentences.append(sent)
        return sentences
    
    # 构建数据集（训练集2万，验证集1千，测试集1千）
    train_en = generate_english(20000)
    train_de = generate_german(20000)
    val_en = generate_english(1000)
    val_de = generate_german(1000)
    test_en = generate_english(1000)
    test_de = generate_german(1000)
    
    # 转换为datasets格式（与原代码兼容）
    dataset = DatasetDict({
        "train": Dataset.from_dict({"en": train_en, "de": train_de}),
        "validation": Dataset.from_dict({"en": val_en, "de": val_de}),
        "test": Dataset.from_dict({"en": test_en, "de": test_de})
    })
    print("已生成离线模拟数据（英德翻译对）：")
    print(f"  训练集：{len(train_en)}句对 | 验证集：{len(val_en)}句对 | 测试集：{len(test_en)}句对")

    # 2. 构建离线词汇表（基于生成的模拟数据）
    print("正在构建离线词汇表...")
    all_texts = []
    for split in ["train", "validation", "test"]:
        all_texts.extend(dataset[split]["en"])
        all_texts.extend(dataset[split]["de"])
    tokenizer.build_vocab(all_texts)

    # 3. 数据编码函数
    def encode_function(examples):
        src_encodings = tokenizer.encode(
            examples["en"],
            padding="max_length",
            truncation=True,
            max_length=config.max_seq_len
        )
        tgt_texts = [f"<s> {text} </s>" for text in examples["de"]]
        tgt_encodings = tokenizer.encode(
            tgt_texts,
            padding="max_length",
            truncation=True,
            max_length=config.max_seq_len
        )
        return {
            "src_input_ids": src_encodings["input_ids"],
            "tgt_input_ids": tgt_encodings["input_ids"],
            "labels": tgt_encodings["input_ids"][:, 1:]
        }

    # 4. 批量编码数据
    encoded_dataset = dataset.map(
        encode_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # 5. 转换为PyTorch格式
    encoded_dataset.set_format("torch", columns=["src_input_ids", "tgt_input_ids", "labels"])

    # 6. 保存数据集信息
    with open("results/dataset_info.txt", "w", encoding="utf-8") as f:
        f.write("数据集：离线模拟英德翻译对（无网络依赖）\n")
        f.write(f"训练集规模：20000句对\n")
        f.write(f"验证集规模：1000句对\n")
        f.write(f"测试集规模：1000句对\n")

    return encoded_dataset

def compute_bleu(preds, refs):
    """计算BLEU评分（适配离线数据）"""
    preds = [tokenizer.decode(pred, skip_special_tokens=True).split() for pred in preds]
    refs = [[tokenizer.decode(ref, skip_special_tokens=True).split()] for ref in refs]
    bleu_scores = [sentence_bleu(ref, pred, weights=(0.25, 0.25, 0.25, 0.25)) for pred, ref in zip(preds, refs)]
    return np.mean(bleu_scores) * 100

def plot_ppl_curve(train_ppls, val_ppls):
    """绘制训练/验证PPL曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_ppls)+1), train_ppls, label="训练集PPL", linewidth=2)
    plt.plot(range(1, len(val_ppls)+1), val_ppls, label="验证集PPL", linewidth=2, linestyle="--")
    plt.xlabel("训练轮次（Epoch）", fontsize=12)
    plt.ylabel("困惑度（PPL）", fontsize=12)
    plt.title("Transformer训练/验证PPL变化曲线", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig(config.ppl_curve_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"PPL曲线已保存至：{config.ppl_curve_path}")