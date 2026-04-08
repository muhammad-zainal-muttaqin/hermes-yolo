#!/bin/bash
# Safety Monitor for Training - Prevent OOM
# Run this to check system health before/during training

echo "🛡️ Safety Monitor - Checking System Health..."
echo "============================================"

# Check GPU Memory
GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_PERCENT=$((GPU_MEM_USED * 100 / GPU_MEM_TOTAL))

echo "📊 GPU Memory: ${GPU_MEM_USED}/${GPU_MEM_TOTAL} MiB (${GPU_PERCENT}%)"

if [ $GPU_PERCENT -gt 90 ]; then
    echo "   ❌ CRITICAL: GPU Memory > 90%! Stop training immediately!"
    exit 1
elif [ $GPU_PERCENT -gt 80 ]; then
    echo "   ⚠️ WARNING: GPU Memory > 80%. Monitor closely."
else
    echo "   ✅ GPU Memory OK"
fi

# Check GPU Temperature
GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
echo "🌡️  GPU Temperature: ${GPU_TEMP}°C"

if [ $GPU_TEMP -gt 85 ]; then
    echo "   ❌ CRITICAL: GPU Temp > 85°C!"
    exit 1
elif [ $GPU_TEMP -gt 75 ]; then
    echo "   ⚠️ WARNING: GPU Temp > 75°C"
else
    echo "   ✅ Temperature OK"
fi

# Check System RAM
RAM_PERCENT=$(free | grep Mem | awk '{printf("%.0f", $3/$2 * 100.0)}')
echo "💾 RAM Usage: ${RAM_PERCENT}%"

if [ $RAM_PERCENT -gt 80 ]; then
    echo "   ⚠️ WARNING: RAM > 80%"
else
    echo "   ✅ RAM OK"
fi

# Check number of training processes
TRAIN_PROCS=$(ps aux | grep -E "python.*train|yolo" | grep -v grep | wc -l)
echo "🔧 Training Processes: ${TRAIN_PROCS}"

if [ $TRAIN_PROCS -gt 10 ]; then
    echo "   ⚠️ Many Python processes running"
else
    echo "   ✅ Process count OK"
fi

echo "============================================"
echo "✅ System Health Check Complete"
