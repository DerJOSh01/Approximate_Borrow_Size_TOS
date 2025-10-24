#!/bin/sh

# bashrc setup
if [ ! -f /home/app/.bashrc ]; then
    # echo "python -m venv /tmp/venv" >> /home/app/.bashrc
    echo "source /home/app/venv/bin/activate" >> /home/app/.bashrc
    echo "cd /codespace" >> /home/app/.bashrc
    # echo "pip install -r requirements.txt" >> /home/app/.bashrc
    echo "python watchlist_launcher_gui.py" >> /home/app/.bashrc
fi

exec /usr/bin/xterm -e /bin/bash