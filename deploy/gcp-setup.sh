#!/usr/bin/env bash
# Run on a fresh Ubuntu GCP VM (after cloning the repo).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LINUX_USER="${SUDO_USER:-$USER}"

echo "==> Installing system packages..."
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip nginx libsndfile1

echo "==> Creating virtualenv and installing Python deps..."
cd "$REPO_DIR"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "==> Installing nginx site..."
sudo cp "$REPO_DIR/deploy/nginx-dopplersim.conf" /etc/nginx/sites-available/dopplersim
sudo ln -sf /etc/nginx/sites-available/dopplersim /etc/nginx/sites-enabled/dopplersim
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl enable nginx
sudo systemctl restart nginx

echo "==> Installing systemd service..."
SECRET_KEY="$(openssl rand -hex 32)"
SERVICE_FILE="/tmp/dopplersim.service"
sed -e "s|YOUR_LINUX_USER|$LINUX_USER|g" \
    -e "s|CHANGE_ME_TO_A_LONG_RANDOM_STRING|$SECRET_KEY|g" \
    "$REPO_DIR/deploy/dopplersim.service" > "$SERVICE_FILE"
sudo cp "$SERVICE_FILE" /etc/systemd/system/dopplersim.service
rm -f "$SERVICE_FILE"

sudo systemctl daemon-reload
sudo systemctl enable dopplersim
sudo systemctl restart dopplersim

echo ""
echo "Done. Open http://YOUR_VM_EXTERNAL_IP/ in a browser."
echo "Check status: sudo systemctl status dopplersim"
echo "View logs:    sudo journalctl -u dopplersim -f"
