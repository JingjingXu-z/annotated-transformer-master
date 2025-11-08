import os
import sys
# 将 src 目录的绝对路径加入系统路径（关键！）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from model import Transformer
# -------------------------- 强制正确导入（必须包含 tokenizer）--------------------------
from utils import (
    set_seed, 
    preprocess_data, 
    compute_bleu, 
    plot_ppl_curve, 
    tokenizer  # 这里必须加！漏了就报NameError
)
from config import Config
# -------------------------- 新增：验证导入是否成功（调试用，可后续删除）--------------------------
try:
    print(f"✅ tokenizer导入成功！pad_token_id={tokenizer.pad_token_id}")
except Exception as e:
    print(f"❌ tokenizer导入失败：{e}")
    exit(1)

config = Config()
set_seed()  # 固定随机种子（作业可重现性要求）


def get_lr_scheduler(optimizer):
    """学习率调度：线性warm-up + 余弦衰减（作业1.19进阶要求）"""
    def lr_lambda(step):
        # 前warmup_steps步：线性上升
        if step < config.warmup_steps:
            return step / config.warmup_steps
        # 之后：余弦衰减
        else:
            return 0.5 * (1 + np.cos(np.pi * (step - config.warmup_steps) / (config.total_steps - config.warmup_steps)))
    
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """单轮训练"""
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        # 数据移至设备（CPU/GPU）
        src = batch["src_input_ids"].to(device)
        tgt = batch["tgt_input_ids"].to(device)
        labels = batch["labels"].to(device)

        # 前向传播
        outputs = model(src, tgt)  # [batch, seq_len-1, vocab_size]
        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1))  # 展平计算交叉熵

        # 反向传播 + 梯度裁剪（作业1.19进阶要求）
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)  # 梯度裁剪
        optimizer.step()
        scheduler.step()  # 学习率更新

        total_loss += loss.item() * src.shape[0]  # 累计批次损失

    # 计算平均损失与PPL
    avg_loss = total_loss / len(dataloader.dataset)
    avg_ppl = np.exp(avg_loss)
    return avg_loss, avg_ppl


def eval_epoch(model, dataloader, criterion, device):
    """单轮评估（计算损失、PPL、BLEU）"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_refs = []

    with torch.no_grad():  # 禁用梯度计算，加速评估
        for batch in dataloader:
            src = batch["src_input_ids"].to(device)
            tgt = batch["tgt_input_ids"].to(device)
            labels = batch["labels"].to(device)

            # 前向传播
            outputs = model(src, tgt)
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1))
            total_loss += loss.item() * src.shape[0]

            # 预测token（取概率最大的token）
            preds = torch.argmax(outputs, dim=-1)  # [batch, seq_len-1]
            all_preds.extend(preds.cpu().numpy())
            all_refs.extend(labels.cpu().numpy())  # 标签作为参考

    # 计算评估指标
    avg_loss = total_loss / len(dataloader.dataset)
    avg_ppl = np.exp(avg_loss)
    avg_bleu = compute_bleu(all_preds, all_refs)
    return avg_loss, avg_ppl, avg_bleu


def generate_translation_samples(model, dataloader, device, num_samples=5):
    """生成翻译样本（作业6.73要求：展示样本预测结果）"""
    model.eval()
    samples = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            src = batch["src_input_ids"].to(device)[:1]  # 取每个批次的第一个样本
            tgt = batch["tgt_input_ids"].to(device)[:1]

            # 生成预测（贪心解码）
            preds = torch.argmax(model(src, tgt), dim=-1)  # [1, seq_len-1]

            # 解码为文本
            src_text = tokenizer.decode(src[0], skip_special_tokens=True)
            tgt_text = tokenizer.decode(tgt[0], skip_special_tokens=True)
            pred_text = tokenizer.decode(preds[0], skip_special_tokens=True)

            samples.append({
                "英文原文": src_text,
                "德文参考译文": tgt_text,
                "模型预测译文": pred_text
            })

    # 保存样本到文件
    with open(config.sample_save_path, "w", encoding="utf-8") as f:
        f.write("Transformer翻译样本（英→德）\n")
        f.write("="*50 + "\n")
        for idx, sample in enumerate(samples, 1):
            f.write(f"样本{idx}：\n")
            f.write(f"英文原文：{sample['英文原文']}\n")
            f.write(f"德文参考译文：{sample['德文参考译文']}\n")
            f.write(f"模型预测译文：{sample['模型预测译文']}\n")
            f.write("-"*50 + "\n")


def main():
    # 1. 设备配置（自动适配CPU/GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # 2. 加载预处理数据
    encoded_dataset = preprocess_data()
    # 创建DataLoader
    train_loader = DataLoader(encoded_dataset["train"], batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(encoded_dataset["validation"], batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(encoded_dataset["test"], batch_size=config.batch_size, shuffle=False)

    # 3. 初始化模型、损失函数、优化器
    model = Transformer().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # 忽略padding位置的损失
    optimizer = optim.AdamW(  # 进阶优化器（作业1.19要求）
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # 4. 计算总步数（用于学习率调度）
    config.total_steps = config.epochs * len(train_loader)
    scheduler = get_lr_scheduler(optimizer)

    # 5. 训练循环（含早停：验证集PPL连续3轮不下降则停止）
    best_val_ppl = float("inf")
    train_ppls = []
    val_ppls = []
    early_stop_count = 0

    print("开始训练...")
    for epoch in range(1, config.epochs + 1):
        # 训练
        train_loss, train_ppl = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        # 验证
        val_loss, val_ppl, val_bleu = eval_epoch(model, val_loader, criterion, device)

        # 记录PPL
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)

        # 打印日志
        print(f"Epoch [{epoch}/{config.epochs}]")
        print(f"  训练集：Loss={train_loss:.4f}, PPL={train_ppl:.2f}")
        print(f"  验证集：Loss={val_loss:.4f}, PPL={val_ppl:.2f}, BLEU={val_bleu:.2f}")

        # 保存最优模型（作业1.19要求：模型保存/加载）
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(model.state_dict(), config.model_save_path)
            print(f"  保存最优模型至：{config.model_save_path}")
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= 3:
                print("  验证集PPL连续3轮不下降，触发早停")
                break

    # 6. 训练后处理
    # 绘制PPL曲线（作业1.15要求）
    plot_ppl_curve(train_ppls, val_ppls)
    print(f"PPL曲线已保存至：{config.ppl_curve_path}")

    # 生成翻译样本（作业6.73要求）
    generate_translation_samples(model, test_loader, device)
    print(f"翻译样本已保存至：{config.sample_save_path}")

    # 测试集最终评估
    model.load_state_dict(torch.load(config.model_save_path))  # 加载最优模型
    test_loss, test_ppl, test_bleu = eval_epoch(model, test_loader, criterion, device)
    print("="*50)
    print("测试集最终结果：")
    print(f"Loss={test_loss:.4f}, PPL={test_ppl:.2f}, BLEU={test_bleu:.2f}")
    print("="*50)


if __name__ == "__main__":
    main()
