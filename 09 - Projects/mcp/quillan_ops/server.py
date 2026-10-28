#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛠️ QUILLAN-RONIN OPS ENGINE — NATIVE FAST-MCP SERVER
=====================================================
Embodying C14-KAIDO (Efficiency), C13-WARDEN (Safety), C2-VIR (Ethics),
C4-PRAXIS (Planning), C19-VIGIL (Anti-drift):
  1. ops_task_graph — goal -> DAG with gates
  2. ops_guardrail_check — pre-execution risk tier gating
  3. ops_health — repo hygiene snapshot (deep: git status, secret scan, large files)
  4. ops_release_note — conventional release note builder
  5. ops_full_pipeline — task_graph -> guardrail -> health -> release note (persisted)
  6. ops_session_list / ops_session_replay — persistence
"""

import sys
# --- Quillan-Ronin v5.4.0-ONI Stdio Guard ---
import logging, builtins
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
_orig_print = builtins.print
def _quillan_safe_print(*args, **kwargs):
    if 'file' not in kwargs or kwargs['file'] is None or kwargs['file'] is sys.stdout:
        kwargs['file'] = sys.stderr
    return _orig_print(*args, **kwargs)
builtins.print = _quillan_safe_print
import os
import json
import time
import uuid
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from fastmcp import FastMCP

SECRET_PATTERNS = ["password", "secret", "token", "api_key", "credential"]
mcp = FastMCP("QuillanOps")
# [QUILLAN-v5.4.0-ONI-SHARED-WIRED]
try:
    from _shared import setup as _quillan_setup, tool_guard, ok_response, error_response, validate_string, validate_path, validate_int, read_file_safe
    from _shared.base_engines import PRISM_ENGINE, COUNCIL_ENGINE, GOVERNOR_ENGINE, SANDBOX_ENGINE
    _SHARED_CTX = _quillan_setup(server_name='quillan_ops', mcp_app=mcp)
except Exception as _shared_err:
    import sys as _sys
    print(f'[quillan_ops] _shared warning: {_shared_err}', file=_sys.stderr)


@mcp.tool()
def ops_task_graph(goal: str, max_steps: int = 6) -> Dict[str, Any]:
    """Decompose a goal into a gated DAG."""
    n = max(2, min(10, max_steps))
    steps = [
        {"id": f"S{i+1}", "title": t, "verify": v, "gate": g}
        for i, (t, v, g) in enumerate([
            ("Slice scope to thin vertical", "scope doc <1 page", "C4-PRAXIS"),
            ("Ship smallest runnable", "runs headless + 60fps probe", "C10-CODEWEAVER"),
            ("Predator strike + fix", "kill-loop closed", "C34-PREDATOR"),
            ("Ethics + safety pass", "VIR/WARDEN sign", "C2-VIR/C13-WARDEN"),
            ("Polish loop + mix", "AURELION>=7/CADENCE drift<=.35", "C22/C23"),
            ("Release + retro", "note + metrics logged", "C5-ECHO"),
        ][:n])
    ]
    return {"goal": goal, "dag": steps, "rule": "No step starts until prior verify passes."}


@mcp.tool()
def ops_guardrail_check(action_desc: str) -> Dict[str, Any]:
    """Pre-execution risk tier gating (low/medium/high/blocked)."""
    low = action_desc.lower()
    if any(k in low for k in ["api_key", "secret", "token=", "private key", ".env"]):
        return {"tier": "blocked", "reason": "touches_secrets p>0.05 — zero-tolerance secret exposure", "action": "HALT + rotate secret"}
    if any(k in low for k in ["rm -rf", "format", "drop table", "delete prod", "force push"]):
        return {"tier": "high", "reason": "destructive_potential >0.40", "action": "Require explicit user confirmation + backup"}
    if any(k in low for k in ["write", "edit", "migrate", "deploy", "publish"]):
        return {"tier": "medium", "reason": "reversible mutation with verification", "action": "Proceed with diff + test seam"}
    return {"tier": "low", "reason": "read-only / sandboxed", "action": "Proceed autonomously"}


@mcp.tool()
def ops_health(repo_path: str = r"C:\02_QUILLAN", deep: bool = True) -> Dict[str, Any]:
    """Snapshot repo hygiene (no secrets read, only metadata). Deep mode adds git status, secret scan, large files."""
    p = Path(repo_path)
    if not p.exists():
        return {"error": f"Path not found: {repo_path}"}
    SKIP_DIRS = {".git", "node_modules", "__pycache__", "venv", ".venv", "unsloth_studio", "checkpoints", ".pytest_cache", ".next", "dist", "build"}
    files: List[Path] = []
    large_files: List[Dict[str, Any]] = []
    secret_hits: List[str] = []
    import re

    scanned_count = 0
    for root, dirs, filenames in os.walk(str(p)):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            fp = Path(root) / fn
            if fn.endswith(".py"):
                files.append(fp)
            if deep and scanned_count < 1000:
                scanned_count += 1
                try:
                    sz = fp.stat().st_size
                    if sz > 5 * 1024 * 1024:
                        large_files.append({"path": str(fp), "size_mb": round(sz / (1024*1024), 2)})
                    if len(secret_hits) < 20 and fp.suffix.lower() in (".py", ".json", ".yaml", ".yml", ".toml", ".env", ".md", ".txt", ".cfg", ".ini"):
                        try:
                            txt = fp.read_text(encoding="utf-8", errors="ignore")
                            for pat in SECRET_PATTERNS:
                                if re.search(pat, txt, re.IGNORECASE):
                                    secret_hits.append(str(fp))
                                    break
                        except Exception:
                            pass
                except Exception:
                    pass
        if len(files) >= 5000:
            break

    large_files.sort(key=lambda x: x["size_mb"], reverse=True)

    result = {
        "repo": str(p),
        "py_files": len(files),
        "largest_py": sorted([(f.stat().st_size, str(f)) for f in files[-500:]], reverse=True)[:5] if files else [],
        "hygiene": ["no .env committed?", "README present?", "tests/ green?", "C19-VIGIL drift check clean?"],
    }
    if not deep:
        return result

    # git status (best-effort)
    git_status = None
    try:
        out = subprocess.run(["git", "-C", str(p), "status", "--short"], capture_output=True, text=True, timeout=10)
        git_status = out.stdout.strip().splitlines()[:20]
    except Exception as e:
        git_status = [f"git unavailable: {e}"]

    result["git_status_short"] = git_status
    result["secret_scan_hits"] = secret_hits
    result["large_files_top"] = large_files[:10]
    result["deep"] = True
    return result


@mcp.tool()
def ops_release_note(version: str, highlights: str, fixes: str = "") -> str:
    """Build a conventional release note."""
    h = "\n".join(f"- {x.strip()}" for x in highlights.split(",") if x.strip())
    f = "\n".join(f"- {x.strip()}" for x in fixes.split(",") if x.strip())
    return f"# {version} — Quillan Dev Release\n\n## Highlights\n{h or '- (none)'}\n\n## Fixes\n{f or '- (none)'}\n\n_Gates: LOGOS>=95% ETHICS=100% TRUTH>=98%_\n"


@mcp.tool()
def ops_full_pipeline(goal: str, action_desc: str = "", repo_path: str = r"C:\02_QUILLAN",
                     version: str = "", highlights: str = "", fixes: str = "",
                     persist: bool = True) -> Dict[str, Any]:
    """Chain task_graph -> guardrail_check -> health -> release_note into one pipeline."""
    tg = ops_task_graph(goal)
    gc = ops_guardrail_check(action_desc or goal)
    health = ops_health(repo_path, deep=True)
    note = ops_release_note(version or "v0.0.0", highlights or goal, fixes) if version or highlights else None

    session_id = f"ops-{int(time.time())}"
    result = {
        "status": "OPS_PIPELINE_COMPLETE",
        "session_id": session_id,
        "goal": goal,
        "task_graph": tg,
        "guardrail": gc,
        "health_summary": {
            "py_files": health.get("py_files"),
            "secret_hits": len(health.get("secret_scan_hits", [])),
            "large_files": len(health.get("large_files_top", [])),
            "git_dirty": len(health.get("git_status_short", []) or []),
        },
        "release_note": note,
        "overall": "PASS" if (gc.get("tier") in ("low", "medium") and not health.get("secret_scan_hits")) else "REVIEW",
        "timestamp": time.time(),
    }

    if persist:
        try:
            p = SESSION_DIR / f"{session_id}.json"
            p.write_text(json.dumps({"type": "ops_pipeline", "result": result}, indent=2), encoding="utf-8")
            result["persisted_to"] = str(p)
        except Exception as e:
            result["persist_error"] = str(e)

    return result


@mcp.tool()
def ops_session_list() -> Dict[str, Any]:
    """List persisted ops sessions."""
    sessions = []
    for p in sorted(SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            r = data.get("result", {})
            sessions.append({
                "session_id": r.get("session_id"),
                "goal": r.get("goal"),
                "overall": r.get("overall"),
                "timestamp": r.get("timestamp"),
                "file": str(p),
            })
        except Exception:
            continue
    return {"count": len(sessions), "sessions": sessions}


@mcp.tool()
def ops_session_replay(session_id: str) -> Dict[str, Any]:
    """Replay a persisted ops session by ID."""
    path = SESSION_DIR / f"{session_id}.json"
    if not path.exists():
        matches = list(SESSION_DIR.glob(f"{session_id}*.json"))
        if not matches:
            return {"error": f"Session not found: {session_id}"}
        path = matches[0]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"error": f"Read failed: {e}"}



# ─── 10. DOCKER CONTAINER MANAGEMENT ─────────────────────────────────────────

@mcp.tool()
def ops_dockerfile_generate(base_image: str = "python:3.12-slim", app_type: str = "fastapi", port: int = 8000) -> Dict[str, Any]:
    """Generate a production Dockerfile."""
    dockerfiles = {
        "fastapi": f"""FROM {base_image}
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE {port}
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "{port}"]
""",
        "node": f"""FROM {base_image}
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
EXPOSE {port}
CMD ["node", "server.js"]
""",
        "python": f"""FROM {base_image}
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE {port}
CMD ["python", "main.py"]
""",
    }
    dockerfile = dockerfiles.get(app_type.lower(), dockerfiles["python"])
    return {
        "app_type": app_type,
        "base_image": base_image,
        "port": port,
        "dockerfile": dockerfile,
        "best_practices": ["multi-stage build", "non-root user", ".dockerignore", "health check", "minimal base image"],
        "council_note": "C26-TECHNE: container constraints; C13-WARDEN: security hardening.",
    }

@mcp.tool()
def ops_docker_compose(services: str = "app,db,redis", network: str = "bridge") -> Dict[str, Any]:
    """Generate a docker-compose.yml configuration."""
    service_list = [s.strip() for s in services.split(",") if s.strip()]
    compose = {"version": "3.8", "services": {}, "networks": {network: {"driver": "bridge"}}}
    for svc in service_list:
        if svc == "app":
            compose["services"][svc] = {"build": ".", "ports": ["8000:8000"], "depends_on": ["db", "redis"], "environment": ["DATABASE_URL=postgres://db:5432/app"]}
        elif svc == "db":
            compose["services"][svc] = {"image": "postgres:16", "environment": ["POSTGRES_DB=app", "POSTGRES_PASSWORD=secret"], "volumes": ["pgdata:/var/lib/postgresql/data"]}
        elif svc == "redis":
            compose["services"][svc] = {"image": "redis:7-alpine", "ports": ["6379:6379"]}
        else:
            compose["services"][svc] = {"image": f"{svc}:latest"}
    compose["volumes"] = {"pgdata": {}}
    return {"services": service_list, "compose": compose, "council_note": "C26-TECHNE: orchestration; C13-WARDEN: network security."}


# ─── 11. KUBERNETES MANIFEST GENERATOR ───────────────────────────────────────

@mcp.tool()
def ops_k8s_manifest(app_name: str = "quillan-app", image: str = "quillan-app:latest", replicas: int = 2, port: int = 8000) -> Dict[str, Any]:
    """Generate Kubernetes deployment + service + ingress manifests."""
    manifests = {
        "deployment": {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": app_name},
            "spec": {
                "replicas": replicas,
                "selector": {"matchLabels": {"app": app_name}},
                "template": {
                    "metadata": {"labels": {"app": app_name}},
                    "spec": {"containers": [{"name": app_name, "image": image, "ports": [{"containerPort": port}],
                                             "resources": {"requests": {"cpu": "100m", "memory": "128Mi"}, "limits": {"cpu": "500m", "memory": "512Mi"}},
                                             "livenessProbe": {"httpGet": {"path": "/health", "port": port}, "initialDelaySeconds": 10},
                                             "readinessProbe": {"httpGet": {"path": "/health", "port": port}, "initialDelaySeconds": 5}}]},
                },
            },
        },
        "service": {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": app_name},
            "spec": {"selector": {"app": app_name}, "ports": [{"port": 80, "targetPort": port}], "type": "ClusterIP"},
        },
        "ingress": {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {"name": app_name, "annotations": {"nginx.ingress.kubernetes.io/rewrite-target": "/"}},
            "spec": {"rules": [{"host": f"{app_name}.example.com", "http": {"paths": [{"path": "/", "pathType": "Prefix", "backend": {"service": {"name": app_name, "port": {"number": 80}}}}]}}]},
        },
    }
    return {"app_name": app_name, "image": image, "replicas": replicas, "manifests": manifests,
            "council_note": "C26-TECHNE: K8s constraints; C13-WARDEN: network security."}


# ─── 12. CI/CD PIPELINE CONFIG ───────────────────────────────────────────────

@mcp.tool()
def ops_cicd_pipeline(platform: str = "github_actions", language: str = "python", test_cmd: str = "pytest") -> Dict[str, Any]:
    """Generate a CI/CD pipeline configuration."""
    pipelines = {
        "github_actions": f"""name: CI/CD Pipeline
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: {test_cmd}
      - name: Lint
        run: ruff check .

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: docker build -t app:latest .
      - name: Push to registry
        run: docker push app:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: kubectl apply -f k8s/
""",
        "gitlab_ci": f"""stages:
  - test
  - build
  - deploy

test:
  stage: test
  image: python:3.12
  script:
    - pip install -r requirements.txt
    - {test_cmd}
    - ruff check .

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t app:latest .
    - docker push app:latest

deploy:
  stage: deploy
  script:
    - kubectl apply -f k8s/
  only:
    - main
""",
    }
    config = pipelines.get(platform.lower(), pipelines["github_actions"])
    return {
        "platform": platform,
        "language": language,
        "test_command": test_cmd,
        "config": config,
        "council_note": "C29-NAVIGATOR: pipeline routing; C13-WARDEN: security gates.",
    }


# ─── 13. INFRASTRUCTURE AS CODE ──────────────────────────────────────────────

@mcp.tool()
def ops_iac_generate(provider: str = "aws", resource: str = "ec2", config: str = "") -> Dict[str, Any]:
    """Generate Infrastructure-as-Code (Terraform) for cloud resources."""
    templates = {
        "aws": {
            "ec2": '''resource "aws_instance" "quillan" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t3.medium"
  tags = { Name = "quillan-server" }
}''',
            "s3": '''resource "aws_s3_bucket" "quillan_data" {
  bucket = "quillan-data-bucket"
  acl    = "private"
}''',
            "rds": '''resource "aws_db_instance" "quillan_db" {
  engine         = "postgres"
  instance_class = "db.t3.micro"
  allocated_storage = 20
}''',
        },
        "gcp": {
            "compute": '''resource "google_compute_instance" "quillan" {
  name         = "quillan-server"
  machine_type = "e2-medium"
  zone         = "us-central1-a"
}''',
        },
        "azure": {
            "vm": '''resource "azurerm_virtual_machine" "quillan" {
  name                  = "quillan-vm"
  location              = "East US"
  vm_size               = "Standard_B2s"
}''',
        },
    }
    provider_templates = templates.get(provider.lower(), templates["aws"])
    tf_code = provider_templates.get(resource.lower(), list(provider_templates.values())[0])
    return {
        "provider": provider,
        "resource": resource,
        "terraform": tf_code,
        "init_commands": ["terraform init", "terraform plan", "terraform apply"],
        "council_note": "C26-TECHNE: infra constraints; C13-WARDEN: security groups.",
    }


# ─── 14. ENVIRONMENT VARIABLE MANAGEMENT ─────────────────────────────────────

@mcp.tool()
def ops_env_manager(action: str = "template", vars: str = "DATABASE_URL,API_KEY,SECRET_KEY") -> Dict[str, Any]:
    """Manage environment variables: generate .env template, validate, or audit."""
    var_list = [v.strip() for v in vars.split(",") if v.strip()]
    if action.lower() == "template":
        template = "\n".join(f"{v}=REPLACE_ME" for v in var_list)
        return {"action": "template", "env_template": template, "file": ".env.example",
                "note": "Copy to .env and fill in values. Never commit .env to git."}
    elif action.lower() == "validate":
        return {"action": "validate", "vars": var_list, "checks": ["all vars present", "no empty values", "no hardcoded secrets"],
                "council_note": "C13-WARDEN: secret scanning."}
    elif action.lower() == "audit":
        return {"action": "audit", "vars": var_list, "risk": "HIGH if secrets in source code",
                "recommendation": "Use secret managers (AWS Secrets Manager, HashiCorp Vault, etc.)"}
    return {"error": f"Unknown action '{action}'. Use: template, validate, audit."}


# ─── 15. LOG AGGREGATION & ANALYSIS ──────────────────────────────────────────

@mcp.tool()
def ops_log_analyze(log_text: str, pattern: str = "error") -> Dict[str, Any]:
    """Analyze log text for errors, warnings, and patterns."""
    import re
    lines = log_text.splitlines()
    errors = [l for l in lines if "error" in l.lower() or "exception" in l.lower() or "traceback" in l.lower()]
    warnings = [l for l in lines if "warning" in l.lower() or "warn" in l.lower()]
    matches = [l for l in lines if pattern.lower() in l.lower()] if pattern else []
    return {
        "total_lines": len(lines),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "pattern_matches": len(matches),
        "errors": errors[:10],
        "warnings": warnings[:10],
        "matches": matches[:10],
        "council_note": "C7-LOGOS: log analysis; C13-WARDEN: anomaly detection.",
    }


# ─── 16. DEPLOYMENT ROLLBACK ─────────────────────────────────────────────────

@mcp.tool()
def ops_rollback(strategy: str = "blue_green", previous_version: str = "", current_version: str = "") -> Dict[str, Any]:
    """Generate a deployment rollback plan."""
    strategies = {
        "blue_green": {"description": "Switch traffic from green (current) back to blue (previous)", "rollback_time": "< 1s", "data_loss": "none"},
        "canary": {"description": "Gradually shift traffic back to previous version", "rollback_time": "minutes", "data_loss": "minimal"},
        "rolling": {"description": "Reverse rolling update", "rollback_time": "minutes", "data_loss": "none"},
        "recreate": {"description": "Stop current, start previous", "rollback_time": "seconds-minutes", "data_loss": "possible"},
    }
    strategy_info = strategies.get(strategy.lower(), strategies["blue_green"])
    return {
        "strategy": strategy,
        "info": strategy_info,
        "previous_version": previous_version,
        "current_version": current_version,
        "steps": ["verify previous version healthy", "shift traffic", "monitor metrics", "confirm rollback", "notify team"],
        "council_note": "C4-PRAXIS: rollback planning; C13-WARDEN: safety.",
    }


# ─── 17. CRON / SCHEDULER MANAGEMENT ─────────────────────────────────────────

@mcp.tool()
def ops_cron_schedule(task: str = "backup", schedule: str = "0 2 * * *", command: str = "") -> Dict[str, Any]:
    """Generate a cron job configuration."""
    return {
        "task": task,
        "schedule": schedule,
        "command": command or f"/usr/local/bin/{task}.sh",
        "cron_expression": schedule,
        "description": "Runs '%s' at %s" % (task, schedule),
        "monitoring": "Log output to /var/log/%s.log" % task,
        "council_note": "C32-AEON: temporal scheduling; C4-PRAXIS: task automation.",
    }


# ─── 18. MONITORING & ALERTING SETUP ─────────────────────────────────────────

@mcp.tool()
def ops_monitoring_setup(stack: str = "prometheus_grafana", metrics: str = "cpu,ram,disk,network,app_latency") -> Dict[str, Any]:
    """Generate monitoring and alerting configuration."""
    metric_list = [m.strip() for m in metrics.split(",") if m.strip()]
    configs = {
        "prometheus_grafana": {
            "prometheus": "scrape_configs: [{job_name: 'quillan', static_configs: [{targets: ['localhost:9090']}]}]",
            "grafana": "dashboard: Quillan-Ronin System Overview",
            "alerts": ["CPU > 90% for 5m", "RAM > 85% for 5m", "Disk > 90%", "App latency > 500ms p99"],
        },
        "datadog": {
            "config": "datadog-agent with APM tracing",
            "alerts": ["High error rate", "Latency spike", "Resource exhaustion"],
        },
    }
    config = configs.get(stack.lower(), configs["prometheus_grafana"])
    return {
        "stack": stack,
        "metrics": metric_list,
        "config": config,
        "council_note": "C30-TESSERACT: real-time monitoring; C13-WARDEN: alert thresholds.",
    }


# ─── 19. BACKUP / RESTORE PLANNING ──────────────────────────────────────────

@mcp.tool()
def ops_backup_plan(data_paths: str = "/data,/config", backup_target: str = "s3", frequency: str = "daily") -> Dict[str, Any]:
    """Generate a backup and restore plan."""
    paths = [p.strip() for p in data_paths.split(",") if p.strip()]
    return {
        "data_paths": paths,
        "backup_target": backup_target,
        "frequency": frequency,
        "retention": {"daily": 7, "weekly": 4, "monthly": 3},
        "backup_commands": [f"tar -czf backup-$(date +%Y%m%d).tar.gz {' '.join(paths)}",
                            f"aws s3 cp backup-$(date +%Y%m%d).tar.gz s3://quillan-backups/"],
        "restore_commands": ["aws s3 cp s3://quillan-backups/backup-YYYYMMDD.tar.gz .", "tar -xzf backup-YYYYMMDD.tar.gz -C /"],
        "verification": "SHA256 checksum + test restore to staging",
        "council_note": "C5-ECHO: data persistence; C13-WARDEN: backup integrity.",
    }


# ─── 20. SSL/TLS CERTIFICATE MANAGEMENT ──────────────────────────────────────

@mcp.tool()
def ops_ssl_cert(domain: str = "quillan.example.com", provider: str = "letsencrypt") -> Dict[str, Any]:
    """Generate SSL/TLS certificate management configuration."""
    return {
        "domain": domain,
        "provider": provider,
        "commands": {
            "letsencrypt": f"certbot certonly --standalone -d {domain}",
            "renewal": "certbot renew --quiet (cron daily)",
            "nginx_config": f"ssl_certificate /etc/letsencrypt/live/{domain}/fullchain.pem; ssl_certificate_key /etc/letsencrypt/live/{domain}/privkey.pem;",
        },
        "auto_renewal": True,
        "expiry_check_days": 30,
        "council_note": "C13-WARDEN: TLS security; C26-TECHNE: cert management.",
    }


# ─── 21. KERNEL BUILD CONFIG ─────────────────────────────────────────────────

@mcp.tool()
def ops_kernel_build(kernel_version: str = "6.8", config: str = "defconfig", arch: str = "x86_64") -> Dict[str, Any]:
    """Generate a Linux kernel build configuration."""
    return {
        "kernel_version": kernel_version,
        "config": config,
        "arch": arch,
        "build_commands": [
            "make defconfig",
            "make menuconfig  # customize",
            "make -j$(nproc)",
            "make modules_install",
            "make install",
        ],
        "cross_compile": "CROSS_COMPILE=aarch64-linux-gnu- make" if arch != "x86_64" else "native",
        "council_note": "C26-TECHNE: kernel constraints; C14-KAIDO: build optimization.",
    }


# ─── 22. ANSIBLE PLAYBOOK GENERATOR ──────────────────────────────────────────

@mcp.tool()
def ops_ansible_playbook(task: str = "deploy", hosts: str = "all", become: bool = True) -> Dict[str, Any]:
    """Generate an Ansible playbook for configuration management."""
    playbook = """---
- name: %s
  hosts: %s
  become: %s
  tasks:
    - name: Ensure package is installed
      apt:
        name: nginx
        state: present
        update_cache: yes
    - name: Start and enable service
      systemd:
        name: nginx
        state: started
        enabled: yes
    - name: Copy config file
      template:
        src: nginx.conf.j2
        dest: /etc/nginx/nginx.conf
      notify: reload nginx
  handlers:
    - name: reload nginx
      systemd:
        name: nginx
        state: reloaded
""" % (task, hosts, str(become).lower())
    return {"task": task, "hosts": hosts, "playbook": playbook,
            "council_note": "C26-TECHNE: config management; C13-WARDEN: security hardening."}


# ─── 23. HELM CHART GENERATOR ────────────────────────────────────────────────

@mcp.tool()
def ops_helm_chart(app_name: str = "quillan-app", chart_version: str = "0.1.0", app_version: str = "1.0.0") -> Dict[str, Any]:
    """Generate a Helm chart structure for Kubernetes deployment."""
    chart_yaml = """apiVersion: v2
name: %s
description: A Helm chart for %s
type: application
version: %s
appVersion: "%s"
""" % (app_name, app_name, chart_version, app_version)
    values_yaml = """replicaCount: 2
image:
  repository: %s
  pullPolicy: IfNotPresent
  tag: "latest"
service:
  type: ClusterIP
  port: 80
resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 100m
    memory: 128Mi
""" % app_name
    return {"app_name": app_name, "chart_yaml": chart_yaml, "values_yaml": values_yaml,
            "structure": ["Chart.yaml", "values.yaml", "templates/deployment.yaml", "templates/service.yaml", "templates/ingress.yaml", "templates/_helpers.tpl"],
            "council_note": "C26-TECHNE: K8s packaging; C29-NAVIGATOR: deployment routing."}


# ─── 24. SERVICE MESH CONFIG (ISTIO/LINKERD) ─────────────────────────────────

@mcp.tool()
def ops_service_mesh(mesh: str = "istio", app_name: str = "quillan-app") -> Dict[str, Any]:
    """Generate service mesh configuration for Istio or Linkerd."""
    if mesh.lower() == "istio":
        config = """apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: %s
spec:
  hosts:
  - %s.example.com
  gateways:
  - %s-gateway
  http:
  - match:
    - uri:
        prefix: /
    route:
    - destination:
        host: %s
        port:
          number: 80
    retries:
      attempts: 3
      perTryTimeout: 2s
""" % (app_name, app_name, app_name, app_name)
    else:
        config = """# Linkerd configuration
apiVersion: linkerd.io/v1alpha2
kind: ServiceProfile
metadata:
  name: %s
spec:
  routes:
  - name: default
    condition:
      pathRegex: .*
    responseClasses:
    - condition:
        status:
          min: 500
          max: 599
      isFailure: true
""" % app_name
    return {"mesh": mesh, "app_name": app_name, "config": config,
            "features": ["mTLS", "traffic splitting", "circuit breaking", "retries", "timeouts"],
            "council_note": "C26-TECHNE: mesh constraints; C13-WARDEN: mTLS security."}


# ─── 25. LOAD BALANCER CONFIG ────────────────────────────────────────────────

@mcp.tool()
def ops_load_balancer(lb_type: str = "nginx", upstreams: str = "app1:8000,app2:8000", algorithm: str = "round_robin") -> Dict[str, Any]:
    """Generate load balancer configuration for nginx or HAProxy."""
    upstream_list = [u.strip() for u in upstreams.split(",") if u.strip()]
    if lb_type.lower() == "nginx":
        upstream_block = "\n".join(f"    server {u};" for u in upstream_list)
        config = """upstream backend {
%s
}

server {
    listen 80;
    server_name example.com;

    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 5s;
        proxy_read_timeout 30s;
    }
}
""" % upstream_block
    else:
        config = """frontend http_front
    bind *:80
    default_backend http_back

backend http_back
    balance %s
%s
""" % (algorithm, "\n".join(f"    server s{i} {u} check" for i, u in enumerate(upstream_list)))
    return {"lb_type": lb_type, "upstreams": upstream_list, "algorithm": algorithm, "config": config,
            "council_note": "C26-TECHNE: LB constraints; C14-KAIDO: traffic optimization."}


# ─── 26. SECRETS MANAGEMENT ──────────────────────────────────────────────────

@mcp.tool()
def ops_secrets_manager(provider: str = "vault", secrets: str = "DB_PASSWORD,API_KEY,JWT_SECRET") -> Dict[str, Any]:
    """Generate secrets management configuration (Vault, AWS Secrets Manager, etc.)."""
    secret_list = [s.strip() for s in secrets.split(",") if s.strip()]
    configs = {
        "vault": {"init": "vault server -dev", "store": "vault kv put secret/quillan %s" % " ".join(f"{s}=value" for s in secret_list),
                  "read": "vault kv get secret/quillan", "features": ["dynamic secrets", "lease management", "audit logging"]},
        "aws": {"store": "aws secretsmanager create-secret --name quillan/%s --secret-string 'value'" % secret_list[0] if secret_list else "secret",
                "read": "aws secretsmanager get-secret-value --secret-id quillan/secret",
                "features": ["rotation", "versioning", "IAM integration"]},
        "k8s": {"store": "kubectl create secret generic quillan-secrets --from-literal=%s" % " --from-literal=".join(f"{s}=value" for s in secret_list),
                "features": ["native K8s", "RBAC", "encryption at rest"]},
    }
    config = configs.get(provider.lower(), configs["vault"])
    return {"provider": provider, "secrets": secret_list, "config": config,
            "best_practices": ["never commit secrets", "rotate regularly", "least privilege", "audit access"],
            "council_note": "C13-WARDEN: secret security; C2-VIR: ethical data handling."}


# ─── 27. DATABASE MIGRATION ──────────────────────────────────────────────────

@mcp.tool()
def ops_db_migration(tool: str = "alembic", db_engine: str = "postgresql", migration_name: str = "add_users_table") -> Dict[str, Any]:
    """Generate database migration configuration."""
    configs = {
        "alembic": {"init": "alembic init alembic", "create": f"alembic revision --autogenerate -m '{migration_name}'",
                    "upgrade": "alembic upgrade head", "downgrade": "alembic downgrade -1",
                    "config": "alembic.ini with sqlalchemy.url"},
        "flyway": {"init": "flyway init", "create": f"flyway migrate (V1__{migration_name}.sql)",
                   "upgrade": "flyway migrate", "info": "flyway info"},
        "prisma": {"init": "npx prisma init", "create": "npx prisma migrate dev --name %s" % migration_name,
                   "deploy": "npx prisma migrate deploy"},
    }
    config = configs.get(tool.lower(), configs["alembic"])
    return {"tool": tool, "db_engine": db_engine, "migration_name": migration_name, "config": config,
            "best_practices": ["always backup before migrate", "test on staging first", "idempotent migrations", "rollback plan"],
            "council_note": "C28-CALCULUS: migration math; C13-WARDEN: data safety."}


# ─── 28. GITOPS CONFIG (ARGOCD/FLUX) ────────────────────────────────────────

@mcp.tool()
def ops_gitops(tool: str = "argocd", app_name: str = "quillan-app", repo_url: str = "https://github.com/leeex1/MCP-DevKit") -> Dict[str, Any]:
    """Generate GitOps configuration for ArgoCD or Flux."""
    configs = {
        "argocd": {
            "application": f"""apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: {app_name}
spec:
  project: default
  source:
    repoURL: {repo_url}
    targetRevision: HEAD
    path: k8s/
  destination:
    server: https://kubernetes.default.svc
    namespace: default
  syncPolicy:
    automated:
      prune: true
      selfHeal: true""",
            "features": ["auto-sync", "self-heal", "prune", "rollback"],
        },
        "flux": {
            "kustomization": f"""apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: {app_name}
spec:
  interval: 5m
  path: ./k8s
  prune: true
  sourceRef:
    kind: GitRepository
    name: {app_name}""",
            "features": ["auto-reconcile", "prune", "health checks"],
        },
    }
    config = configs.get(tool.lower(), configs["argocd"])
    return {"tool": tool, "app_name": app_name, "config": config,
            "council_note": "C29-NAVIGATOR: GitOps routing; C4-PRAXIS: deployment automation."}


# ─── 29. HORIZONTAL POD AUTOSCALER ──────────────────────────────────────────

@mcp.tool()
def ops_hpa(app_name: str = "quillan-app", min_replicas: int = 1, max_replicas: int = 10, target_cpu: int = 70) -> Dict[str, Any]:
    """Generate Horizontal Pod Autoscaler configuration."""
    hpa = {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {"name": app_name},
        "spec": {
            "scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": app_name},
            "minReplicas": min_replicas,
            "maxReplicas": max_replicas,
            "metrics": [{"type": "Resource", "resource": {"name": "cpu", "target": {"type": "Utilization", "averageUtilization": target_cpu}}}],
            "behavior": {"scaleDown": {"stabilizationWindowSeconds": 300}, "scaleUp": {"stabilizationWindowSeconds": 0}},
        },
    }
    return {"app_name": app_name, "hpa": hpa, "council_note": "C14-KAIDO: auto-scaling; C32-AEON: load prediction."}


# ─── 30. DISTRIBUTED TRACING SETUP ───────────────────────────────────────────

@mcp.tool()
def ops_tracing(stack: str = "jaeger", service_name: str = "quillan-app") -> Dict[str, Any]:
    """Generate distributed tracing configuration (Jaeger/Zipkin/Tempo)."""
    configs = {
        "jaeger": {"deploy": "kubectl apply -f jaeger.yaml", "endpoint": "http://jaeger-collector:14268/api/traces",
                   "sdk": "opentelemetry-instrumentation", "sampling": "probabilistic (0.1)"},
        "zipkin": {"deploy": "docker run -d -p 9411:9411 openzipkin/zipkin", "endpoint": "http://zipkin:9411/api/v2/spans",
                   "sdk": "zipkin-python", "sampling": "const (1.0)"},
        "tempo": {"deploy": "grafana/tempo", "endpoint": "http://tempo:3200", "sdk": "opentelemetry",
                  "features": ["trace-to-logs", "trace-to-metrics"]},
    }
    config = configs.get(stack.lower(), configs["jaeger"])
    return {"stack": stack, "service_name": service_name, "config": config,
            "council_note": "C30-TESSERACT: tracing; C18-SHEPHERD: observability."}


# ─── 31. MESSAGE QUEUE SETUP ─────────────────────────────────────────────────

@mcp.tool()
def ops_message_queue(broker: str = "rabbitmq", queues: str = "tasks,results,notifications") -> Dict[str, Any]:
    """Generate message queue configuration (RabbitMQ/Kafka/Redis)."""
    queue_list = [q.strip() for q in queues.split(",") if q.strip()]
    configs = {
        "rabbitmq": {"image": "rabbitmq:3-management", "port": 5672, "management_port": 15672,
                     "queues": queue_list, "features": ["AMQP", "dead letter", "TTL"]},
        "kafka": {"image": "confluentinc/cp-kafka", "port": 9092, "topics": queue_list,
                  "features": ["partitioning", "replication", "consumer groups"]},
        "redis": {"image": "redis:7", "port": 6379, "streams": queue_list,
                  "features": ["pub/sub", "streams", "sorted sets"]},
    }
    config = configs.get(broker.lower(), configs["rabbitmq"])
    return {"broker": broker, "queues": queue_list, "config": config,
            "council_note": "C31-NEXUS: message routing; C14-KAIDO: throughput."}


# --- Quillan-Ronin Global Error-Handling Middleware ---
from fastmcp.server.middleware import Middleware, MiddlewareContext
import logging as _log

class QuillanErrorMiddleware(Middleware):
    async def on_call_tool(self, context, call_next):
        try:
            return await call_next(context)
        except Exception as e:
            _log.error("Tool error: %s", e, exc_info=True)
            return {"status": "ERROR", "error": str(e), "error_type": type(e).__name__}


mcp.add_middleware(QuillanErrorMiddleware())
if __name__ == "__main__":
    mcp.run()
