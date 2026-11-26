import json
import ssl
import sys
from unittest.mock import Mock

import pytest

from scripts import autodl_rent


PAYLOAD = {"instance_name": "fusion-test", "gpu_spec_uuid": "4090D"}
OUT_OF_STOCK = (200, {"code": "InternalError", "msg": "当前算力规格暂无库存。"})
CREATED = (200, {"code": "Success", "data": "pro-mock-created"})


def test_create_sends_official_request(monkeypatch):
    response = Mock(status=200)
    response.read.return_value = json.dumps(CREATED[1]).encode()
    connection = Mock()
    connection.getresponse.return_value = response
    https = Mock(return_value=connection)
    context = Mock()
    monkeypatch.setattr(autodl_rent, "HTTPSConnection", https)
    monkeypatch.setattr(autodl_rent, "tls_context", lambda: context)

    assert autodl_rent.create_instance("test-token", PAYLOAD) == CREATED
    https.assert_called_once_with("api.autodl.com", context=context, timeout=20)
    method, path, body, headers = connection.request.call_args.args
    assert (method, path) == ("POST", "/api/v1/dev/instance/pro/create")
    assert json.loads(body) == PAYLOAD
    assert headers["Authorization"] == "test-token"
    connection.close.assert_called_once()


def test_tls_loads_system_bundle_when_openssl_default_is_missing(monkeypatch, tmp_path):
    bundle = tmp_path / "ca-bundle.crt"
    bundle.touch()
    context = Mock()
    monkeypatch.setattr(ssl, "create_default_context", lambda: context)
    monkeypatch.setattr(ssl, "get_default_verify_paths", lambda: Mock(cafile=None))
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    monkeypatch.setenv("NIX_SSL_CERT_FILE", str(bundle))

    assert autodl_rent.tls_context() is context
    context.load_verify_locations.assert_called_once_with(cafile=str(bundle))


def test_reports_certificate_failure_before_http_request(monkeypatch, capsys):
    error = ssl.SSLCertVerificationError()
    error.verify_message = "certificate verify failed"
    create = Mock(side_effect=error)
    monkeypatch.setattr(autodl_rent, "create_instance", create)

    assert autodl_rent.rent("test-token", PAYLOAD, 2, 60, autodl_rent.STOCK_MESSAGES) == 3
    create.assert_called_once_with("test-token", PAYLOAD)
    assert "租用请求未发送" in capsys.readouterr().err


def test_preview_has_one_gpu_and_sends_no_request(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["autodl_rent.py", "--image", "base-image-preview"])
    monkeypatch.delenv("AUTODL_TOKEN", raising=False)
    create = Mock()
    monkeypatch.setattr(autodl_rent, "create_instance", create)

    assert autodl_rent.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["req_gpu_amount"] == 1
    assert payload["gpu_spec_uuid"] == "4090D"
    assert payload["cuda_v_from"] == 124
    create.assert_not_called()


@pytest.mark.parametrize(
    ("responses", "expected_exit", "expected_waits"),
    [([OUT_OF_STOCK, CREATED], 0, 1), ([OUT_OF_STOCK] * 2, 1, 1)],
)
def test_retries_only_until_success_or_limit(monkeypatch, capsys, responses, expected_exit, expected_waits):
    create = Mock(side_effect=responses)
    sleep = Mock()
    monkeypatch.setattr(autodl_rent, "create_instance", create)
    monkeypatch.setattr(autodl_rent.time, "sleep", sleep)

    assert autodl_rent.rent("test-token", PAYLOAD, 2, 60, autodl_rent.STOCK_MESSAGES) == expected_exit
    assert create.call_count == len(responses)
    assert sleep.call_count == expected_waits
    assert capsys.readouterr().out.strip() == ("pro-mock-created" if expected_exit == 0 else "")


@pytest.mark.parametrize(
    "response",
    [
        TimeoutError(),
        (200, {"code": "Success"}),
        (200, {"code": "InternalError", "msg": "unknown error"}),
        (200, {"code": "BadRequest", "msg": "余额不足 test-token"}),
        (429, {}),
        (503, OUT_OF_STOCK[1]),
    ],
)
def test_stops_when_creation_is_uncertain_or_rejected_for_other_reason(monkeypatch, capsys, response):
    create = Mock(side_effect=[response])
    sleep = Mock()
    monkeypatch.setattr(autodl_rent, "create_instance", create)
    monkeypatch.setattr(autodl_rent.time, "sleep", sleep)

    assert autodl_rent.rent("test-token", PAYLOAD, 2, 60, autodl_rent.STOCK_MESSAGES) == 3
    create.assert_called_once_with("test-token", PAYLOAD)
    sleep.assert_not_called()
    captured = capsys.readouterr()
    assert "fusion-test" in captured.err
    assert "test-token" not in captured.err
    assert captured.out == ""
