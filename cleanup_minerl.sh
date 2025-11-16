#!/bin/bash
# Cleanup script for MineRL processes and PID files

echo "Cleaning up MineRL processes..."

# Kill stuck processes
pkill -9 -f 'minerl.utils.process_watcher' 2>/dev/null
pkill -9 -f 'launchClient.sh' 2>/dev/null
pkill -9 -f 'MCP-Reborn' 2>/dev/null
pkill -9 -f 'java.*malmo' 2>/dev/null
pkill -9 -f 'java.*MCP' 2>/dev/null

# Wait a bit
sleep 2

# Remove PID files
rm -f /tmp/minerl_watcher_*.pid
rm -f /tmp/*malmo*.pid
rm -f /tmp/process_watcher_*.pid
rm -f ./minerl_watcher_*.pid

echo "Cleanup complete!"

