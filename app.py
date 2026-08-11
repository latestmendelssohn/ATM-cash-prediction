r"""
ATM cash forecasting -- FastAPI streaming service + command-line interface.
==========================================================================

Serve:
    uvicorn app:app --reload          # or:  python app.py serve

CLI:
    python app.py generate-data
    python app.py forecast  --atm ATM001 [--model holt_winters] [--horizon 14]
    python app.py backtest  --atm ATM001
    python app.py cash-plan --atm ATM001 [--service-level 0.95] [--balance 1.5e7]
    python app.py pipeline                        # forecast every ATM
    python app.py index --atm ATM001              # embed reports (needs GOOGLE_API_KEY)
    python app.py chat "How much cash for ATM001 next week?"

The forecast / backtest / cash-plan / pipeline commands run on the pure-Python
core with no third-party packages.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta

import data
import models

DATA_PATH = "data/atm_transactions.csv"


# ---------------------------------------------------------------------------
# Business logic (shared by API + CLI)
# ---------------------------------------------------------------------------

def forecast_atm(atm_id, model="holt_winters", horizon=14, level=0.95, path=DATA_PATH):
    dates, y = data.load_series(path, atm_id)
    m = models.build_model(model)
    m.fit(y)
    point, lower, upper = m.predict_interval(horizon, level)
    sigma = getattr(m, "_sigma", (upper[0] - lower[0]) / (2 * models.z_for_level(level)))
    start = dates[-1] + timedelta(days=1)
    return {
        "atm_id": atm_id, "model": model, "horizon": horizon,
        "forecast_start": start.isoformat(),
        "dates": [(start + timedelta(days=i)).isoformat() for i in range(horizon)],
        "point": point, "lower": lower, "upper": upper, "sigma": sigma,
        "params": getattr(m, "params", {}),
    }


def cash_plan_atm(atm_id, service_level=0.95, horizon=14, balance=None, path=DATA_PATH):
    fc = forecast_atm(atm_id, "holt_winters", horizon, path=path)
    plan = models.recommend_cash_load(fc["point"], fc["sigma"], service_level, balance)
    return {"atm_id": atm_id, **plan}


def build_reports(atm_id, horizon=14, service_level=0.95, path=DATA_PATH):
    import analyst
    loc = data.location_map(path).get(atm_id, "")
    fc = forecast_atm(atm_id, "holt_winters", horizon, path=path)
    board = models.leaderboard(data.load_series(path, atm_id)[1], horizon=horizon)
    plan = models.recommend_cash_load(fc["point"], fc["sigma"], service_level)
    start = fc["forecast_start"]
    return [
        analyst.forecast_report(atm_id, "holt_winters", start, fc["point"],
                                fc["lower"], fc["upper"], loc),
        analyst.backtest_report(atm_id, board),
        analyst.cash_plan_report(atm_id, {"atm_id": atm_id, **plan}),
    ]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_app():
    import uuid

    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    app = FastAPI(title="ATM Cash Forecasting + RAG Analyst", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    sessions: dict = {}          # session_id -> list[(role, text)]
    holder: dict = {}            # lazily-built Analyst

    def analyst_obj():
        if "a" not in holder:
            import analyst
            holder["a"] = analyst.Analyst()
        return holder["a"]

    class ForecastReq(BaseModel):
        atm_id: str
        model: str = "holt_winters"
        horizon: int = 14
        level: float = 0.95

    class CashReq(BaseModel):
        atm_id: str
        horizon: int = 14
        service_level: float = 0.95
        balance: float | None = None

    class ChatReq(BaseModel):
        question: str
        atm_id: str | None = None
        session_id: str | None = None

    @app.get("/health")
    def health():
        import os
        return {"status": "ok", "has_api_key": bool(os.getenv("GOOGLE_API_KEY"))}

    @app.post("/forecast")
    def forecast(req: ForecastReq):
        try:
            return forecast_atm(req.atm_id, req.model, req.horizon, req.level)
        except (ValueError, KeyError) as e:
            raise HTTPException(400, str(e))
        except ImportError as e:
            raise HTTPException(501, f"model backend unavailable: {e}")

    @app.post("/cash-plan")
    def cash_plan(req: CashReq):
        try:
            return cash_plan_atm(req.atm_id, req.service_level, req.horizon, req.balance)
        except (ValueError, KeyError) as e:
            raise HTTPException(400, str(e))

    @app.post("/index/{atm_id}")
    def index(atm_id: str):
        try:
            n = analyst_obj().add_reports(build_reports(atm_id))
            return {"atm_id": atm_id, "documents_added": n, "total": analyst_obj().count()}
        except (ValueError, KeyError) as e:
            raise HTTPException(400, str(e))
        except RuntimeError as e:
            raise HTTPException(503, str(e))

    @app.post("/ingest/pdf")
    async def ingest_pdf(file: UploadFile = File(...)):
        import os
        tmp = f"/tmp/{file.filename}"
        try:
            with open(tmp, "wb") as fh:
                fh.write(await file.read())
            n = analyst_obj().add_pdf(tmp)
        except Exception as e:
            raise HTTPException(400, f"failed to ingest PDF: {e}")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return {"documents_added": n, "total": analyst_obj().count()}

    @app.post("/chat")
    def chat(req: ChatReq):
        sid = req.session_id or uuid.uuid4().hex
        try:
            res = analyst_obj().ask(req.question, req.atm_id)
        except RuntimeError as e:
            raise HTTPException(503, str(e))
        sessions.setdefault(sid, []).append(("user", req.question))
        sessions[sid].append(("assistant", res["answer"]))
        return {"session_id": sid, **res}

    @app.post("/chat/stream")
    def chat_stream(req: ChatReq):
        sid = req.session_id or uuid.uuid4().hex

        def gen():
            yield f"event: session\ndata: {sid}\n\n"
            collected = []
            try:
                for tok in analyst_obj().stream(req.question, req.atm_id):
                    collected.append(tok)
                    yield f"data: {json.dumps({'token': tok})}\n\n"
            except RuntimeError as e:
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                return
            sessions.setdefault(sid, []).append(("assistant", "".join(collected)))
            yield "event: done\ndata: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


# ASGI entrypoint for `uvicorn app:app`
try:
    app = create_app()
except ImportError:
    app = None  # FastAPI not installed -> CLI-only mode still works


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv=None):
    p = argparse.ArgumentParser(prog="app.py", description="ATM cash forecasting + RAG analyst")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate-data"); g.add_argument("--out", default=DATA_PATH)
    g.add_argument("--atms", type=int, default=5); g.add_argument("--days", type=int, default=1095)

    fc = sub.add_parser("forecast"); fc.add_argument("--atm", required=True)
    fc.add_argument("--horizon", type=int, default=14)
    fc.add_argument("--model", default="holt_winters"); fc.add_argument("--level", type=float, default=0.95)

    cp = sub.add_parser("cash-plan"); cp.add_argument("--atm", required=True)
    cp.add_argument("--horizon", type=int, default=14)
    cp.add_argument("--service-level", dest="sl", type=float, default=0.95)
    cp.add_argument("--balance", type=float, default=None)

    ix = sub.add_parser("index"); ix.add_argument("--atm", required=True)
    ix.add_argument("--horizon", type=int, default=14)

    sub.add_parser("backtest").add_argument("--atm", required=True)
    sub.add_parser("pipeline")
    c = sub.add_parser("chat"); c.add_argument("question"); c.add_argument("--atm", default=None)
    c.add_argument("--stream", action="store_true")
    sv = sub.add_parser("serve"); sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=8000)

    a = p.parse_args(argv)

    if a.cmd == "generate-data":
        from pathlib import Path
        s = data.generate(Path(a.out), a.atms, date(2021, 1, 1), a.days)
        print(json.dumps(s, indent=2))
    elif a.cmd == "forecast":
        print(json.dumps(forecast_atm(a.atm, a.model, a.horizon, a.level), indent=2, default=str))
    elif a.cmd == "backtest":
        _, y = data.load_series(DATA_PATH, a.atm)
        print(f"Rolling-origin leaderboard for {a.atm}:\n")
        print(models.format_table(models.leaderboard(y)))
    elif a.cmd == "cash-plan":
        print(json.dumps(cash_plan_atm(a.atm, a.sl, a.horizon, a.balance), indent=2, default=str))
    elif a.cmd == "pipeline":
        print(f"{'ATM':<8}{'best':<15}{'MASE':>7}{'total_14d':>16}{'cycle_load':>16}")
        for atm in data.list_atms(DATA_PATH):
            _, y = data.load_series(DATA_PATH, atm)
            board = models.leaderboard(y)
            fc = forecast_atm(atm, board[0]["model"], 14)
            plan = models.recommend_cash_load(fc["point"], fc["sigma"], 0.95)
            print(f"{atm:<8}{board[0]['model']:<15}{board[0]['MASE']:>7}"
                  f"{sum(fc['point']):>16,.0f}{plan['cycle_load']:>16,.0f}")
    elif a.cmd == "index":
        import analyst
        obj = analyst.Analyst()
        print(f"Indexed {obj.add_reports(build_reports(a.atm))} docs; total {obj.count()}.")
    elif a.cmd == "chat":
        import analyst
        obj = analyst.Analyst()
        if a.stream:
            for tok in obj.stream(a.question, a.atm):
                sys.stdout.write(tok); sys.stdout.flush()
            print()
        else:
            print(obj.ask(a.question, a.atm)["answer"])
    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("app:app", host=a.host, port=a.port)


if __name__ == "__main__":
    _cli()
