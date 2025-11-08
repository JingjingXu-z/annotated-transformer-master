#!/bin/bash
# 作业要求：提供重现实验的exact命令行（含随机种子）

# 1. 检查Python环境（确保Python 3.10+）
if ! command -v python3.10 &> /dev/null; then
    echo "错误：未找到Python 3.10，请先安装"
    exit 1
fi

# 2. 安装依赖（作业1.14要求：requirements.txt）
echo "正在安装依赖库..."
pip install -r requirements.txt

# 3. 创建results目录（若不存在）
mkdir -p results

# 4. 训练模型（固定随机种子42，作业可重现性要求）
echo "开始训练（随机种子：${SEED:-42}）..."
python src/train.py \
    --seed ${SEED:-42} \
    --batch_size 32 \
    --epochs 20 \
    --lr 3e-4 \
    --weight_decay 1e-4 \
    --clip_norm 1.0

# 5. 评估模型（加载最优模型）
echo "开始评估..."
python src/train.py \
    --eval \
    --checkpoint results/best_model.pth

echo "所有任务完成！结果已保存至 results/ 目录"