# World Cup AI Agent Deployment

## Local Streamlit

Use the same Python interpreter as the project environment:

```powershell
cd C:\Users\Administrator\Desktop\worldcup-ai-agent
C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open:

```text
http://127.0.0.1:8501
```

## Local Website Server

The static website server serves `website/public` and exposes the protected refresh endpoint.

```powershell
cd C:\Users\Administrator\Desktop\worldcup-ai-agent
C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe website\server.py
```

Open:

```text
http://127.0.0.1:8765
```

Refresh endpoint:

```text
POST http://127.0.0.1:8765/api/worldcup/refresh-intel
```

## Environment Variables

Copy `.env.template` to `.env` for local development or configure these variables in the host provider:

```text
NO_PAID_API_MODE=True
ODDS_PROVIDER=
ODDS_API_KEY=
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.pixvyn.com
OPENAI_INTEL_MODEL=gpt-5.5
REFRESH_ADMIN_USERNAME=
REFRESH_ADMIN_PASSWORD=
```

`REFRESH_ADMIN_USERNAME` and `REFRESH_ADMIN_PASSWORD` are optional for local development. If both are missing, refresh is allowed locally.

For deployment, always set:

```text
REFRESH_ADMIN_USERNAME
REFRESH_ADMIN_PASSWORD
```

Do not commit `.env`.

## Refresh Protection

Public visitors can view the dashboard without logging in.

Manual refresh actions require admin credentials:

- Streamlit sidebar refresh form asks for username and password.
- `backend/api.py` validates credentials before calling refresh workflows.
- `website/server.py` validates the refresh endpoint with either JSON body credentials or HTTP Basic Auth.

JSON example:

```json
{
  "username": "admin",
  "password": "secret"
}
```

Basic Auth is also accepted.

If credentials are wrong, the endpoint returns `403 Forbidden` and does not run refresh.

Passwords are not logged.

## Render Deployment

This project includes `render.yaml` for Render Blueprint deployment:

```yaml
services:
  - type: web
    name: worldcup-ai-agent
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "streamlit run app.py --server.port $PORT --server.address 0.0.0.0"
```

### Streamlit Dashboard

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

Set environment variables in Render dashboard:

```text
OPENAI_API_KEY
OPENAI_BASE_URL=https://api.pixvyn.com
OPENAI_INTEL_MODEL=gpt-5.5
REFRESH_ADMIN_USERNAME
REFRESH_ADMIN_PASSWORD
```

After deployment, test:

- Page opens for normal visitors.
- Normal visitors can view the dashboard without login.
- Refresh fails with invalid admin credentials.
- Refresh runs only with the correct admin username/password.

### Static website server

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
python website/server.py
```

For Render, update the server binding to use `0.0.0.0` and the provider `PORT` if this server is deployed publicly.

## VPS Deployment

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Optional website server:

```bash
python website/server.py
```

Recommended process managers:

- Windows Task Scheduler
- NSSM on Windows
- systemd on Linux VPS
- supervisor or pm2 if already used

## Refresh Jobs

Manual:

```bash
python scripts/run_daily_refresh.py
python scripts/run_hourly_intel_refresh.py
```

Scheduler:

```bash
python scripts/scheduler.py
```

The scheduler only runs while its Python process is alive. For production, prefer OS-level scheduling.

## Data And Logs

Generated data:

```text
data/processed/
reports/
website/public/data/
```

Ignored local/runtime files:

```text
.env
data/cache/
reports/*.log
__pycache__/
*.pyc
```
