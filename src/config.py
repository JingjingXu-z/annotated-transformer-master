# 超参数配置：严格匹配作业表3，补充进阶优化参数
class Config:
    # 模型基础参数（作业表3要求）
    d_model = 128        # 嵌入维度
    n_heads = 4          # 注意力头数
    d_ff = 512           # 前馈网络维度
    n_encoder_layers = 2 # Encoder层数
    n_decoder_layers = 2 # Decoder层数（作业要求加Decoder得80-90分）
    
    # 训练参数（作业可重现性要求）
    seed = 42            # 固定随机种子（作业要求 exact 重现）
    batch_size = 32      # 批大小（作业表3要求）
    epochs = 20          # 训练轮次
    lr = 3e-4            # 初始学习率（作业表3要求）
    weight_decay = 1e-4  # AdamW权重衰减（进阶优化）
    clip_norm = 1.0      # 梯度裁剪阈值（进阶优化）
    warmup_steps = 1000  # 学习率warm-up步数（进阶优化）
    
    # 数据参数（作业数据集建议）
    dataset_name = "iwslt2017" # 数据集（Seq2Seq任务，作业表2推荐）
    src_lang = "en"      # 源语言（英文）
    tgt_lang = "de"      # 目标语言（德文）
    max_seq_len = 50     # 句子最大长度（避免冗余）
    train_ratio = 0.8    # 训练集占比
    val_ratio = 0.1      # 验证集占比
    
    # 路径参数
    model_save_path = "results/best_model.pth" # 最优模型保存路径
    ppl_curve_path = "results/train_val_ppl.png" # PPL曲线保存路径
    sample_save_path = "results/translation_samples.txt" # 翻译样本路径