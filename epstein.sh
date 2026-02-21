#!/bin/bash
# Management script for Epstein Files Search

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_FILE="$SCRIPT_DIR/data/epstein.log"
PID_FILE="$SCRIPT_DIR/data/epstein.pid"

cd "$SCRIPT_DIR" || exit 1
mkdir -p "$SCRIPT_DIR/data"

case "$1" in
    start)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "Already running (PID: $PID)"
                exit 1
            fi
            rm -f "$PID_FILE"
        fi

        echo "Starting Epstein Files Search..."
        nohup python run.py >> "$LOG_FILE" 2>&1 &
        PID=$!
        echo $PID > "$PID_FILE"

        sleep 1
        if ps -p "$PID" > /dev/null 2>&1; then
            PORT=${PORT:-5555}
            echo "Running (PID: $PID) — http://localhost:$PORT"
            echo "Log: $LOG_FILE"
        else
            echo "Failed to start — check $LOG_FILE"
            rm -f "$PID_FILE"
            exit 1
        fi
        ;;

    stop)
        STOPPED=0

        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                kill "$PID"
                sleep 1
                if ps -p "$PID" > /dev/null 2>&1; then
                    kill -9 "$PID" 2>/dev/null
                fi
                echo "Stopped (PID: $PID)"
                STOPPED=1
            fi
            rm -f "$PID_FILE"
        fi

        # Catch any orphaned run.py processes
        PIDS=$(ps auxw | grep "run.py" | grep epstein | grep -v grep | awk '{print $2}' 2>/dev/null || true)
        for P in $PIDS; do
            if ps -p "$P" > /dev/null 2>&1; then
                kill "$P" 2>/dev/null
                sleep 1
                kill -9 "$P" 2>/dev/null 2>&1
                echo "Stopped orphan (PID: $P)"
                STOPPED=1
            fi
        done

        [ $STOPPED -eq 0 ] && echo "Not running"
        ;;

    restart)
        "$0" stop
        sleep 2
        "$0" start
        ;;

    status)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                ELAPSED=$(ps -p "$PID" -o etime= 2>/dev/null | xargs)
                echo "Running (PID: $PID, uptime: $ELAPSED)"
                exit 0
            else
                rm -f "$PID_FILE"
            fi
        fi
        echo "Not running"
        exit 1
        ;;

    log)
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            echo "No log file yet: $LOG_FILE"
        fi
        ;;

    *)
        echo "Usage: $0 {start|stop|restart|status|log}"
        echo ""
        echo "  start    Start as background daemon"
        echo "  stop     Stop the daemon"
        echo "  restart  Stop then start"
        echo "  status   Check if running"
        echo "  log      Tail the log (Ctrl-C to stop)"
        exit 1
        ;;
esac

exit 0
