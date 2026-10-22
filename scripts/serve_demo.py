"""Serve the local learned-model simulation; binds loopback only, never uploads."""
from pathlib import Path
import argparse
from dataclasses import fields, asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from delivery_risk.demo_model import load_frozen_demo_model
from delivery_risk.sidewalk_simulation import SidewalkConfig, simulate_sidewalk
from delivery_risk.sidewalk_worlds import SIDEWALK_CATALOG
from delivery_risk.assistance import ASSISTANCE_VERSION


def information(model):
    return {"model": "HistGradientBoosting · full · seed42", "training_rows": model.fit_audit["training_rows"],
        "feature_count": len(model.columns), "dev_threshold": model.dev_policy["cutoff"],
        "eval_ap": .7302762063131997, "baseline_ap": .44926474314513035,
        "synthetic": True, "domain": "robot", "scenario_catalog": SIDEWALK_CATALOG,
        "assistance_module_version": ASSISTANCE_VERSION,
        "primary_strategy": "learned_with_fallback_and_explicit_simulated_operator_commands",
        "notes": ["配送机器人与行人仅在人行道/斑马线通行；汽车仅机动车道", "实际冻结HistGB评分；原训练域为道路车辆，机器人属于域外迁移示意", "模型分数未经机器人域校准", "运动、通行域与协助机制是显式规则", "垃圾、建筑与围挡不是原模型已学习类别"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=ROOT / "artifacts/prepared/samples.csv")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--export-example", action="store_true")
    args = parser.parse_args()
    if not args.samples.is_file():
        parser.error("Original local prepared samples required for model inference; see README. Offline animation still works from demo/index.html.")
    model = load_frozen_demo_model(args.samples)
    info = information(model)
    cache, lock = {}, threading.Lock()
    def run(config):
        cfg = SidewalkConfig(**config)
        cfg.validate()
        key = tuple(sorted(asdict(cfg).items()))
        with lock:
            if key not in cache:
                if len(cache) >= 30:
                    cache.pop(next(iter(cache)))
                cache[key] = simulate_sidewalk(model, cfg)
            return cache[key]
    gallery_path = ROOT / "demo/sidewalk_gallery.js"
    if gallery_path.is_file():
        gallery = json.loads(gallery_path.read_text(encoding="utf-8").split("=", 1)[1].strip().removesuffix(";"))
        for replay in gallery.get("runs", {}).values():
            if replay.get("metadata", {}).get("assistance_policy", {}).get("module_version") != ASSISTANCE_VERSION:
                continue
            cfg = SidewalkConfig(**replay["config"])
            cfg.validate()
            cache[tuple(sorted(asdict(cfg).items()))] = replay
    if args.export_example:
        default = run({"scenario": "blind_corner", "domain": "robot", "fleet": 6, "threshold": info["dev_threshold"]})
        path = ROOT / "demo/sidewalk_example.js"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("window.DELIVERY_SIDEWALK_EXAMPLE=" + json.dumps({"info": info, "run": default}, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + ";\n", encoding="utf-8")
        print(json.dumps({"exported_synthetic_animation": str(path), "summary": default["summary"]}, ensure_ascii=False), flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, message, *values):
            print(message % values, flush=True)
        def send(self, status, payload, content_type="application/json; charset=utf-8"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status); self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff"); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            if self.path == "/api/info":
                self.send(200, info)
            elif self.path in ("/", "/index.html"):
                self.send(200, (ROOT / "demo/index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/example_run.js" and (ROOT / "demo/example_run.js").is_file():
                self.send(200, (ROOT / "demo/example_run.js").read_bytes(), "text/javascript; charset=utf-8")
            elif self.path == "/sidewalk_gallery.js" and gallery_path.is_file():
                self.send(200, gallery_path.read_bytes(), "text/javascript; charset=utf-8")
            elif self.path == "/favicon.ico":
                self.send(204, b"", "image/x-icon")
            else:
                self.send(404, {"error": "not_found"})
        def do_POST(self):
            if self.path != "/api/simulate":
                self.send(404, {"error": "not_found"}); return
            origin = self.headers.get("Origin")
            if origin and origin not in (f"http://127.0.0.1:{args.port}", f"http://localhost:{args.port}"):
                self.send(403, {"error": "local_origin_required"}); return
            try:
                count = int(self.headers.get("Content-Length", "0"))
                if not 0 < count <= 4096:
                    raise ValueError("request_size_invalid")
                config = json.loads(self.rfile.read(count))
                cls = SidewalkConfig
                allowed = {field.name for field in fields(cls)} - {"dt", "duration_s", "service_s"}
                if not isinstance(config, dict) or not set(config).issubset(allowed):
                    raise ValueError("unsupported_request_fields")
                cls(**config).validate()
                self.send(200, run(config))
            except (ValueError, TypeError) as exc:
                self.send(400, {"error": str(exc)})
            except Exception as exc:
                print(f"Simulation failed: {type(exc).__name__}: {exc}", flush=True)
                self.send(500, {"error": "simulation_failed_see_local_terminal"})
    print(f"Local simulation ready: http://127.0.0.1:{args.port}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
