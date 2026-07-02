# VPS Deployment Guide

Use this guide to deploy the app on a VPS so it can be opened from mobile or desktop browser.

## 1. Copy Project To VPS

Copy the `main_project` folder to your VPS, for example:

```bash
/opt/prescription-dashboard/main_project
```

## 2. Install System Packages

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip nginx
```

## 3. Create Python Environment

```bash
cd /opt/prescription-dashboard/main_project
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Configure Environment

Create or update `.env`:

```bash
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL_NAME=gemma-4-31b-it
```

Never share this `.env` file publicly.

## 5. Test Run

```bash
cd /opt/prescription-dashboard/main_project
source venv/bin/activate
streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

Open:

```text
http://YOUR_VPS_IP:8501
```

If it works, stop it with `Ctrl+C`.

## 6. Run As A Service

Make the run script executable and give the service user access:

```bash
sudo chmod +x /opt/prescription-dashboard/main_project/deploy/run_streamlit.sh
sudo chown -R www-data:www-data /opt/prescription-dashboard/main_project
```

Copy the service file:

```bash
sudo cp deploy/prescription-dashboard.service /etc/systemd/system/prescription-dashboard.service
sudo systemctl daemon-reload
sudo systemctl enable prescription-dashboard
sudo systemctl start prescription-dashboard
sudo systemctl status prescription-dashboard
```

Logs:

```bash
journalctl -u prescription-dashboard -f
```

## 7. Nginx Reverse Proxy

Copy the Nginx config:

```bash
sudo cp deploy/nginx-prescription-dashboard.conf /etc/nginx/sites-available/prescription-dashboard
sudo ln -s /etc/nginx/sites-available/prescription-dashboard /etc/nginx/sites-enabled/prescription-dashboard
sudo nginx -t
sudo systemctl reload nginx
```

Then open:

```text
http://YOUR_VPS_IP
```

## 8. Domain And HTTPS

If you have a domain, point it to the VPS IP first. Then install HTTPS:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

After this, users can open from mobile:

```text
https://your-domain.com
```

## 9. Important VPS Notes

- Keep `users.db`, `output/`, and `dataset/` writable by the service user.
- Keep `.env` private.
- Use HTTPS before giving access to real users.
- Do not expose port `8501` publicly if Nginx is already proxying port `80/443`.
- Backup `users.db` and `output/csv_collections/` regularly.
