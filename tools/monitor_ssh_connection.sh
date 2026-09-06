#!/bin/bash
# SSH connection monitor - checks if remote server is responsive

SERVER="root@183.147.142.130"
PORT="9000"
MAX_ATTEMPTS=60
INTERVAL=30

attempt=1
while [ $attempt -le $MAX_ATTEMPTS ]; do
    echo "[$(date +%H:%M:%S)] Attempt $attempt/$MAX_ATTEMPTS: Testing SSH connection..."

    if timeout 5 ssh -p $PORT $SERVER "echo 'OK' && cd /root/F5-TTS && git log --oneline -1" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] ✅ SSH connection restored!"
        echo "[$(date +%H:%M:%S)] Remote HEAD: $(ssh -p $PORT $SERVER 'cd /root/F5-TTS && git log --oneline -1' 2>/dev/null)"
        exit 0
    else
        echo "[$(date +%H:%M:%S)] ❌ Connection failed, waiting ${INTERVAL}s..."
        sleep $INTERVAL
    fi

    attempt=$((attempt + 1))
done

echo "[$(date +%H:%M:%S)] ⏱️ Timeout after $MAX_ATTEMPTS attempts ($(($MAX_ATTEMPTS * $INTERVAL / 60)) minutes)"
exit 1
