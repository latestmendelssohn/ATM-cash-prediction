"""
Command-line interface  (stdlib argparse -- no heavy deps for core commands).

    atmf generate-data [--atms N] [--days D]
    atmf forecast --atm ATM001 [--model holt_winters] [--horizon 14]
    atmf backtest --atm ATM001 [--horizon 14]
    atmf cash-plan --atm ATM001 [--service-level 0.95] [--balance X]
    atmf index --atm ATM001            (requires GOOGLE_API_KEY + deps)
    atmf chat "question" [--atm ATM001] (requires GOOGLE_API_KEY + deps)
    atmf serve [--host 0.0.0.0] [--port 8000]

The forecasting / backtest / cash-plan commands run on the pure-Python core and
need no third-party packages.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

DEFAULT_DATA = "data/raw/atm_transactions.csv"


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _cmd_generate_data(args) -> None:
    from datetime import date

    from .data.generate_synthetic import generate

    y, m, d = (int(x) for x in args.start.split("-"))
    summary = generate(out_path=__import__("pathlib").Path(args.out), n_atms=args.atms,
                       start=date(y, m, d), n_days=args.days, seed=args.seed)
    _print_json(summary)


def _cmd_forecast(args) -> None:
    from . import service

    fc = service.forecast_atm(args.data, args.atm, model=args.model,
                              horizon=args.horizon, level=args.level)
    # keep the print compact
    fc_out = {k: fc[k] for k in ("atm_id", "model", "horizon", "forecast_start",
                                 "dates", "point", "lower", "upper", "residual_std")}
    _print_json(fc_out)


def _cmd_backtest(args) -> None:
    from . import service
    from .evaluation.backtest import format_leaderboard

    board = service.backtest_atm(args.data, args.atm, horizon=args.horizon)
    print(f"Rolling-origin backtest leaderboard for {args.atm} (H={args.horizon}):\n")
    print(format_leaderboard(board))


def _cmd_cash_plan(args) -> None:
    from . import service

    fc = service.forecast_atm(args.data, args.atm, model=args.model, horizon=args.horizon)
    plan = service.cash_plan_atm(fc, service_level=args.service_level, current_balance=args.balance)
    _print_json(plan)


def _cmd_index(args) -> None:
    from . import service
    from .rag.agent import RAGAnalyst

    docs = service.build_reports_for_atm(args.data, args.atm, model=args.model, horizon=args.horizon)
    analyst = RAGAnalyst()
    added = analyst.store.add_reports(docs)
    print(f"Indexed {added} documents for {args.atm}. Vector store now holds {analyst.store.count()}.")


def _cmd_chat(args) -> None:
    from .rag.agent import RAGAnalyst

    analyst = RAGAnalyst()
    if args.stream:
        for tok in analyst.stream_answer(args.question, atm_id=args.atm):
            sys.stdout.write(tok)
            sys.stdout.flush()
        print()
    else:
        res = analyst.answer(args.question, atm_id=args.atm)
        print(res["answer"])


def _cmd_serve(args) -> None:
    import uvicorn

    uvicorn.run("atmforecast.api.app:app", host=args.host, port=args.port, reload=args.reload)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="atmf", description="ATM cash forecasting & RAG analyst.")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate-data", help="generate the synthetic dataset")
    g.add_argument("--out", default=DEFAULT_DATA)
    g.add_argument("--atms", type=int, default=5)
    g.add_argument("--days", type=int, default=3 * 365)
    g.add_argument("--start", default="2021-01-01")
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=_cmd_generate_data)

    f = sub.add_parser("forecast", help="forecast one ATM")
    f.add_argument("--atm", required=True)
    f.add_argument("--data", default=DEFAULT_DATA)
    f.add_argument("--model", default="holt_winters")
    f.add_argument("--horizon", type=int, default=14)
    f.add_argument("--level", type=float, default=0.95)
    f.set_defaults(func=_cmd_forecast)

    b = sub.add_parser("backtest", help="rolling-origin leaderboard for one ATM")
    b.add_argument("--atm", required=True)
    b.add_argument("--data", default=DEFAULT_DATA)
    b.add_argument("--horizon", type=int, default=14)
    b.set_defaults(func=_cmd_backtest)

    c = sub.add_parser("cash-plan", help="recommend a cash-replenishment plan")
    c.add_argument("--atm", required=True)
    c.add_argument("--data", default=DEFAULT_DATA)
    c.add_argument("--model", default="holt_winters")
    c.add_argument("--horizon", type=int, default=14)
    c.add_argument("--service-level", dest="service_level", type=float, default=0.95)
    c.add_argument("--balance", type=float, default=None)
    c.set_defaults(func=_cmd_cash_plan)

    i = sub.add_parser("index", help="build & embed RAG reports for one ATM")
    i.add_argument("--atm", required=True)
    i.add_argument("--data", default=DEFAULT_DATA)
    i.add_argument("--model", default="holt_winters")
    i.add_argument("--horizon", type=int, default=14)
    i.set_defaults(func=_cmd_index)

    ch = sub.add_parser("chat", help="ask the RAG analyst a question")
    ch.add_argument("question")
    ch.add_argument("--atm", default=None)
    ch.add_argument("--stream", action="store_true")
    ch.set_defaults(func=_cmd_chat)

    s = sub.add_parser("serve", help="run the FastAPI server")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=_cmd_serve)

    return p


def app(argv: Optional[List[str]] = None) -> None:
    """Console-script entrypoint (referenced by pyproject [project.scripts])."""
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    app()
