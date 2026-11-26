#!/usr/bin/env python3
"""Rent one GPU through the AutoDL Pro API using only the Python standard library."""

import argparse
from datetime import datetime, timezone
from http.client import HTTPException, HTTPSConnection
import json
import os
from pathlib import Path
import ssl
import sys
import time
from uuid import uuid4


API_HOST = "api.autodl.com"
CREATE_PATH = "/api/v1/dev/instance/pro/create"
STOCK_MESSAGES = {
    "当前算力规格暂无库存",
    "当前算力规格暂无库存，请稍后再试",
}


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def tls_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    if ssl.get_default_verify_paths().cafile or os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
        return context

    # Some Python builds look for /etc/ssl/cert.pem on Linux while NixOS
    # installs its system trust bundle under /etc/ssl/certs/.
    candidates = (
        os.environ.get("NIX_SSL_CERT_FILE"),
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/ssl/certs/ca-bundle.crt",
    )
    for path in candidates:
        if path and Path(path).is_file():
            context.load_verify_locations(cafile=path)
            break
    return context


def create_instance(token: str, payload: dict) -> tuple[int, dict]:
    connection = HTTPSConnection(API_HOST, context=tls_context(), timeout=20)
    try:
        connection.request(
            "POST", CREATE_PATH, json.dumps(payload).encode("utf-8"),
            {"Authorization": token, "Content-Type": "application/json"},
        )
        response = connection.getresponse()
        try:
            body = json.loads(response.read())
        except (UnicodeError, ValueError):
            body = None
        return response.status, body if isinstance(body, dict) else {}
    finally:
        connection.close()


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def normalize_message(message: str) -> str:
    return message.strip().rstrip("。.!！")


def rent(token: str, payload: dict, attempts: int, interval: int, stock_messages: set[str]) -> int:
    name = payload["instance_name"]
    log(f"准备按量租用 1 张 {payload['gpu_spec_uuid']}，实例备注名 {name}")
    for attempt in range(1, attempts + 1):
        try:
            status, body = create_instance(token, payload)
        except ssl.SSLCertVerificationError as exc:
            log(f"TLS 证书验证失败，租用请求未发送。{exc.verify_message}")
            return 3
        except (HTTPException, OSError, ValueError) as exc:
            log(f"请求结果未知（{type(exc).__name__}）。重新运行前，请在控制台查找 {name}。")
            return 3

        code = body.get("code")
        instance_id = body.get("data")
        if 200 <= status < 300 and code == "Success" and isinstance(instance_id, str) and instance_id:
            log("创建请求成功。请在 AutoDL 控制台查看实例状态。")
            print(instance_id, flush=True)
            return 0

        message = normalize_message(str(body.get("msg") or ""))
        clear_rejection = isinstance(code, str) and code not in ("", "Success") and instance_id in (None, "")
        if status in (200, 400, 409, 422) and clear_rejection and message in stock_messages:
            if attempt == attempts:
                log(f"{attempts} 次尝试均遇到无库存，等待结束。")
                return 1
            log(f"[{attempt}/{attempts}] 暂无库存，{interval} 秒后重试。")
            time.sleep(interval)
            continue

        detail = f"HTTP {status}, code={code!r}, msg={message!r}, request_id={body.get('request_id')!r}"
        log(detail.replace(token, "[已隐藏]"))
        log(f"已停止。重新运行前，请在控制台查找 {name}。")
        return 3


def main() -> int:
    parser = argparse.ArgumentParser(
        description="通过 AutoDL 容器实例 Pro API 等待并租用一张 GPU。默认只预览请求。",
        epilog="官方规格 ID 与鉴权说明：https://www.autodl.com/docs/instance_pro_api/",
    )
    parser.add_argument("--apply", action="store_true", help="发送按量计费的创建请求。从 AUTODL_TOKEN 读取 Token。")
    parser.add_argument("--gpu-spec", default="4090D", help="Pro 算力规格 ID，默认 4090D。")
    parser.add_argument("--image", required=True, help="AutoDL Pro 公共或私有镜像 UUID。")
    parser.add_argument("--cuda-min", type=positive_int, default=124, help="宿主机驱动的 CUDA 下限，124 表示 12.4。")
    parser.add_argument("--region", action="append", help="地区 ID，可重复指定。省略时由 AutoDL 调度。")
    parser.add_argument("--name", help="实例备注名。默认生成唯一名称，便于在控制台查找。")
    parser.add_argument("--attempts", type=positive_int, default=30, help="最大请求次数，默认 30。")
    parser.add_argument("--interval", type=positive_int, default=60, help="确认无库存后的重试间隔，默认 60 秒。")
    parser.add_argument("--stock-message", action="append", default=[], help="额外的确切无库存文案，可重复指定。")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    payload = {
        "req_gpu_amount": 1,
        "expand_system_disk_by_gb": 0,
        "gpu_spec_uuid": args.gpu_spec,
        "image_uuid": args.image,
        "cuda_v_from": args.cuda_min,
        "instance_name": args.name or f"fusion-{timestamp}-{uuid4().hex[:6]}",
    }
    if args.region:
        payload["data_center_list"] = args.region

    if not args.apply:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    token = os.environ.get("AUTODL_TOKEN", "").strip()
    if not token:
        parser.error("--apply 需要环境变量 AUTODL_TOKEN")

    stock_messages = STOCK_MESSAGES | set(filter(None, map(normalize_message, args.stock_message)))
    try:
        return rent(token, payload, args.attempts, args.interval, stock_messages)
    except KeyboardInterrupt:
        log(f"已中断。重新运行前，请在控制台查找 {payload['instance_name']}。")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
