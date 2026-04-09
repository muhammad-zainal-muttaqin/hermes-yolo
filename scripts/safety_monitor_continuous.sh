#!/bin/bash
# Continuous Safety Monitor for Training
# Run this in background to monitor and prevent OOM

LOG_FILE="/workspace/Hermes-YOLO/safety_monitor.log"
ALERT_FILE="/tmp/training_alert"

# Thresholds
GPU_MEM_THRESHOLD=85  # Alert if GPU memory > 85%
GPU_TEMP_THRESHOLD=80  # Alert if GPU temp > 80°C
RAM_THRESHOLD=85      # Alert if RAM > 85%

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "🛡️ Continuous Safety Monitor Started"
log "   GPU Memory Threshold: ${GPU_MEM_THRESHOLD}%"
log "   GPU Temp Threshold: ${GPU_TEMP_THRESHOLD}°C"
log "   RAM Threshold: ${RAM_THRESHOLD}%"

while true; do
    # Check GPU Memory
    GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    GPU_PERCENT=$((GPU_MEM_USED * 100 / GPU_MEM_TOTAL))
    
    # Check GPU Temperature
    GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    
    # Check RAM
    RAM_USED=$(free | grep Mem | awk '{print $3}')
    RAM_TOTAL=$(free | grep Mem | awk '{print $2}')
    RAM_PERCENT=$((RAM_USED * 100 / RAM_TOTAL))
    
    # Alert conditions
    ALERT=0
    ALERT_MSG=""
    
    if [ $GPU_PERCENT -gt $GPU_MEM_THRESHOLD ]; then
        ALERT=1
        ALERT_MSG="GPU Memory CRITICAL: ${GPU_PERCENT}%"
    fi
    
    if [ $GPU_TEMP -gt $GPU_TEMP_THRESHOLD ]; then
        ALERT=1
        ALERT_MSG="${ALERT_MSG}, GPU Temp HIGH: ${GPU_TEMP}°C"
    fi
    
    if [ $RAM_PERCENT -gt $RAM_THRESHOLD ]; then
        ALERT=1
        ALERT_MSG="${ALERT_MSG}, RAM HIGH: ${RAM_PERCENT}%"
    fi
    
    if [ $ALERT -eq 1 ]; then
        log "⚠️ ALERT: $ALERT_MSG"
        echo "$ALERT_MSG" > "$ALERT_FILE"
        
        # Optional: Kill training if critical
        if [ $GPU_PERCENT -gt 95 ] || [ $GPU_TEMP -gt 85 ]; then
            log "🚨 CRITICAL: Consider stopping training!"
        fi
    else
        # Normal status (log every 5 minutes only)
        if [ $(($(date +%s) % 300)) -lt 10 ]; then
            log "✅ OK: GPU ${GPU_PERCENT}%, Temp ${GPU_TEMP}°C, RAM ${RAM_PERCENT}%"
        fi
    fi
    
    # Check every 10 seconds
    sleep 10
done
