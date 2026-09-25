"""Tests for the vLLM backend, parallel runs and the no-debate baselines.

Run with pytest, or directly:  python tests/test_backend.py
A tiny local HTTP server stands in for vLLM, so no GPU or model is needed.
"""
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def _in_repo(fn):
    def wrapper():
        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            fn()
        finally:
            os.chdir(cwd)
    wrapper.__name__ = fn.__name__
    return wrapper


class _FakeVLLM(BaseHTTPRequestHandler):
    requests = []

    def do_GET(self):
        data = json.dumps({"data": [{"id": "org/model-AWQ"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeVLLM.requests.append((self.path, body))
        if "too long" in body["messages"][0]["content"]:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":"maximum context length exceeded"}')
            return
        out = {"choices": [{"message": {"content": '{"action":"18"}'}}],
               "usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def _serve():
    srv = HTTPServer(("127.0.0.1", 0), _FakeVLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_vllm_provider_sends_openai_request_with_seed():
    from src.agents import base_agent
    srv = _serve()
    try:
        with tempfile.TemporaryDirectory() as d:
            cfg = os.path.join(d, "m.yaml")
            with open(cfg, "w") as f:
                f.write(f"model:\n  provider: vllm\n  name: org/model-AWQ\n"
                        f"  base_url: http://127.0.0.1:{srv.server_port}/v1\n"
                        f"  temperature: 0.3\n  max_tokens: 64\nagents:\n  solver: {{}}\n")
            agent = base_agent.BaseAgent("solver", cfg)
            agent.seed = 7
            # _call_vllm directly: other test files may have replaced call_llm with a fake
            text, tokens = agent._call_vllm("hello")
            assert json.loads(text)["action"] == "18" and tokens == 42
            path, body = _FakeVLLM.requests[-1]
            assert path == "/v1/chat/completions" and body["model"] == "org/model-AWQ"
            assert body["seed"] == 7 and body["max_tokens"] == 64
            assert body["messages"] == [{"role": "user", "content": "hello"}]
            try:
                agent._call_vllm("this prompt is too long")
            except ValueError:
                pass
            else:
                raise AssertionError("a 400 must raise at once, not retry")
    finally:
        srv.shutdown()


def test_check_vllm_server_catches_dead_or_wrong_server():
    import socket
    from src.agents.base_agent import check_vllm_server
    srv = _serve()
    with socket.socket() as s:  # a port nobody listens on
        s.bind(("127.0.0.1", 0))
        dead_port = s.getsockname()[1]
    try:
        with tempfile.TemporaryDirectory() as d:
            def cfg(name, port):
                path = os.path.join(d, f"{port}_{name.replace('/', '_')}.yaml")
                with open(path, "w") as f:
                    f.write(f"model:\n  provider: vllm\n  name: {name}\n"
                            f"  base_url: http://127.0.0.1:{port}/v1\n")
                return path
            check_vllm_server(cfg("org/model-AWQ", srv.server_port))
            for bad in (cfg("org/other-model", srv.server_port), cfg("org/model-AWQ", dead_port)):
                try:
                    check_vllm_server(bad)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError(f"{bad} must fail the pre-flight check")
    finally:
        srv.shutdown()


@_in_repo
def test_vllm_configs_parse():
    from src.agents.base_agent import BaseAgent
    for f in sorted(os.listdir("config/vllm")):
        a = BaseAgent("verifier" if "verifier" in f else "solver", os.path.join("config/vllm", f))
        assert a.provider == "vllm" and a.base_url.startswith("http://localhost:800")
    assert BaseAgent("verifier", "config/vllm/verifier_phi4.yaml").temperature == 0.0


def _fake_answers():
    from src.agents.base_agent import BaseAgent
    seen = []
    lock = threading.Lock()

    def call_llm(self, prompt):
        with lock:
            seen.append((self.role, self.seed))
        if self.role == "critic" and "SOLVER" in prompt:
            return json.dumps({"reasoning": ["ok"], "action": "agree", "confidence": 0.9, "content": "ok"}), 5
        return json.dumps({"reasoning": ["16-3-4=9", "9x2=18"], "action": "18", "confidence": 0.9,
                           "content": "18"}), 5
    BaseAgent.call_llm = call_llm
    return seen


@_in_repo
def test_parallel_run_task_writes_every_question_once():
    import MAD
    _fake_answers()
    with tempfile.TemporaryDirectory() as d:
        MAD.run_task("gsm8k", n=6, max_rounds=2, seed=3, resume=False, stop_policy="none",
                     critic_mode="live", log_root=d, workers=4)
        files = [os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs if f.endswith(".jsonl")]
        recs = [json.loads(l) for f in files for l in open(f, encoding="utf-8") if l.strip()]
    assert len(recs) == 6 and len({r["sample_id"] for r in recs}) == 6
    assert all(r["seed"] == 3 and r["rounds_used"] == 2 for r in recs)


@_in_repo
def test_no_debate_votes_get_distinct_seeds_without_touching_env():
    from baselines.no_debate import run_one
    from tasks import get_benchmark
    os.environ["MAD_SEED"] = "5"
    seen = _fake_answers()
    cfg = "config/model_config_solver_qwen.yaml"
    rec = run_one(get_benchmark("gsm8k"), 1, "q", [("agent_a", cfg, "qwen")], "18", n_votes=3,
                  mode="self_consistency")
    assert [s for _, s in seen] == [5, 5001, 5002]
    assert os.environ["MAD_SEED"] == "5" and rec["final_answer"] == "18" and rec["final_correct"]
    single = run_one(get_benchmark("gsm8k"), 1, "q", [("agent_a", cfg, "qwen")], "18", n_votes=1,
                     mode="single_agent")
    assert single["mode"] == "single_agent" and single["n_votes_total"] == 1


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                failed += 1
                import traceback
                traceback.print_exc()
                print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if failed else 0)
