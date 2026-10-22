"""LangGraph agent: guardrails -> router -> (policy RAG | account SQL | canned reply) -> finalise.

        START
          |
     input_guard --(blocked)--> canned ----------------+
          |                                            |
       classify                                        |
      /    |     \\                                    |
 policy  account  other                                |
   |       |        |                                  |
 rewrite   |       oos                                 |
   |       |        |                                  |
  rag   account     |                                  |
   \\       |       /                                   |
    +---> finalize <-----------------------------------+
              |
             END

The LLM is injected, so the whole graph runs offline in tests with a fake model, and with
no API key at all the app degrades to retrieval-only and keyword-driven account tools.
"""
from __future__ import annotations

import re
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app import db as bankdb
from app.audit import log_event
from app.config import Settings
from app.guardrails import CANNED, check_input
from app.pii import mask_pii
from app.retrieval import HybridRetriever


class State(TypedDict, total=False):
    question: str
    safe_question: str
    pii_found: list[str]
    customer_id: str
    history: list[dict[str, str]]
    guard_action: str
    route: str
    standalone: str
    answer: str
    citations: list[dict[str, Any]]
    sql: str
    table: dict[str, Any]
    trace: list[str]
    started: float
    latency_ms: int


CLASSIFY_SYSTEM = (
    "You route messages for a retail bank's customer assistant. Reply with exactly one word:\n"
    "policy  - questions about the bank's products, fees, charges, terms, processes or regulations\n"
    "account - questions about the customer's OWN balances, transactions or spending\n"
    "other   - anything else (small talk, unrelated topics)"
)

REWRITE_SYSTEM = (
    "Rewrite the user's last message as a standalone question, using the conversation for "
    "context. Do not answer it. Output only the question."
)

RAG_SYSTEM = (
    "You are a customer assistant for a retail bank (a demo project built on public documents).\n"
    "Answer ONLY from the numbered context passages. Rules:\n"
    "- Cite the passage number like [1] after each claim you take from it.\n"
    "- If the passages do not contain the answer, say you could not find it. Never guess.\n"
    "- Quote fees, rates, limits and dates exactly as written, with their conditions.\n"
    "- If passages disagree or look out of date, say so.\n"
    "- Do not give personal financial advice or recommend a product.\n"
    "- Do not state any phone number or web address that is not in the passages.\n"
    "- Be concise. UK English. End with one short line telling the user to check the bank's "
    "current terms before making a decision that depends on this answer."
)

SQL_SYSTEM = (
    "Write ONE SQLite SELECT query that answers the customer's question.\n"
    "{schema}\n"
    "Rules: SELECT only. The data is already limited to this customer, so do not filter by "
    "customer. The latest date in the data is {as_of}; for relative periods use "
    "date('{as_of}', '-30 days') style expressions, never date('now'). "
    "Spending is a negative amount; show spending as positive numbers by negating it. "
    "Round money to 2 decimals. Add LIMIT 50 unless aggregating.\n"
    "If the question cannot be answered from these tables, return exactly: "
    "SELECT 'unsupported' AS result\n"
    "Return only the SQL."
)

SUMMARY_SYSTEM = (
    "Summarise this query result for a bank customer in one or two plain sentences. "
    "Use only numbers that appear in the table. Do not give advice."
)


def _text(resp: Any) -> str:
    return str(getattr(resp, "content", resp)).strip()


def get_llm(settings: Settings) -> Any | None:
    if not settings.groq_api_key:
        return None
    from langchain_groq import ChatGroq

    return ChatGroq(groq_api_key=settings.groq_api_key, model_name=settings.groq_model, temperature=0.0, max_tokens=700)


def heuristic_route(q: str) -> str:
    low = q.lower()
    account_cues = [
        r"\bmy (current |savings |credit card )?(account|balance|transactions?|spending|statement)\b",
        r"\bhow much (did|have) i (spend|spent|pay|paid)\b", r"\bi (spent|paid)\b",
        r"\brecent transactions\b", r"\blast (\d+ )?transactions\b", r"\bmy balance\b",
        r"\bspending (on|by|in|last|this)\b",
    ]
    if any(re.search(p, low) for p in account_cues):
        return "account"
    return "policy"


def to_markdown(cols: list[str], rows: list[tuple[Any, ...]]) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:,.2f}"
        return str(v)

    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def build_graph(retriever: HybridRetriever, llm: Any | None, settings: Settings):
    def step(state: State, note: str) -> list[str]:
        return [*state.get("trace", []), note]

    def input_guard(state: State) -> dict[str, Any]:
        safe, found = mask_pii(state["question"])
        res = check_input(safe)
        return {
            "safe_question": safe, "pii_found": found, "guard_action": res.action,
            "started": time.time(), "trace": [f"guard:{res.action}"] + ([f"masked:{','.join(found)}"] if found else []),
        }

    def canned(state: State) -> dict[str, Any]:
        action = state["guard_action"]
        return {"answer": CANNED[action], "route": action, "citations": [], "trace": step(state, f"canned:{action}")}

    def classify(state: State) -> dict[str, Any]:
        q = state["safe_question"]
        route = None
        if llm is not None:
            try:
                out = _text(llm.invoke([("system", CLASSIFY_SYSTEM), ("user", q)])).lower()
                m = re.search(r"\b(policy|account|other)\b", out)
                route = m.group(1) if m else None
            except Exception as e:  # model/API failure must not take the chat down
                return {"route": heuristic_route(q), "trace": step(state, f"classify:llm_error:{type(e).__name__}->heuristic")}
        route = route or heuristic_route(q)
        return {"route": route, "trace": step(state, f"classify:{route}")}

    def rewrite(state: State) -> dict[str, Any]:
        q = state["safe_question"]
        history = state.get("history") or []
        if llm is None or not history:
            return {"standalone": q, "trace": step(state, "rewrite:skipped")}
        convo = "\n".join(f'{m["role"]}: {m["content"]}' for m in history[-6:])
        try:
            out = _text(llm.invoke([("system", REWRITE_SYSTEM), ("user", f"{convo}\nuser: {q}")]))
            return {"standalone": out or q, "trace": step(state, "rewrite:done")}
        except Exception:
            return {"standalone": q, "trace": step(state, "rewrite:error")}

    def rag(state: State) -> dict[str, Any]:
        q = state.get("standalone") or state["safe_question"]
        hits = retriever.search(q, top_k=settings.top_k, fetch_k=settings.fetch_k)
        best = retriever.best_dense_score(q)
        trace = step(state, f"retrieve:{len(hits)} hits, best_dense={best:.2f}")
        if not hits or best < settings.min_score:
            return {"answer": CANNED["not_found"], "citations": [], "trace": trace + ["abstain:low_score"]}

        cites = [{
            "n": i, "source": h.source, "page": h.page, "doc_type": h.doc_type,
            "dense": h.dense_score, "bm25": h.bm25_score, "rerank": h.rerank_score,
            "excerpt": h.text[:400], "cited": False,
        } for i, h in enumerate(hits, start=1)]

        if llm is None:
            lines = [f"[{c['n']}] {c['source']} p.{c['page']}: {c['excerpt']}" for c in cites[:3]]
            msg = "The language model isn't configured (no GROQ_API_KEY), so here are the most relevant passages:\n\n" + "\n\n".join(lines)
            for c in cites[:3]:
                c["cited"] = True
            return {"answer": msg, "citations": cites, "trace": trace + ["generate:skipped_no_llm"]}

        context = "\n\n".join(f"[{c['n']}] ({c['source']}, p.{c['page']}, {c['doc_type']})\n{h.text}" for c, h in zip(cites, hits))
        try:
            ans = _text(llm.invoke([("system", RAG_SYSTEM), ("user", f"Context:\n{context}\n\nQuestion: {q}")]))
        except Exception as e:
            return {"answer": "Sorry, I couldn't generate an answer just now. Please try again.", "citations": cites,
                    "trace": trace + [f"generate:error:{type(e).__name__}"]}
        used = {int(n) for n in re.findall(r"\[(\d+)\]", ans)}
        for c in cites:
            c["cited"] = c["n"] in used
        return {"answer": ans, "citations": cites, "trace": trace + ["generate:done"]}

    def account(state: State) -> dict[str, Any]:
        cid = state.get("customer_id")
        if not cid:
            return {"answer": "Please select a demo customer in the sidebar first.", "trace": step(state, "account:no_customer")}
        q = state["safe_question"]
        path = settings.db_path

        if llm is None:
            low = q.lower()
            if "transaction" in low or "recent" in low:
                rows = bankdb.get_recent_transactions(path, cid, 10)
                cols = ["date", "account", "description", "category", "amount"]
            elif "spen" in low or "category" in low:
                rows = bankdb.spending_by_category(path, cid, 30)
                cols = ["category", "spent"]
            else:
                rows = bankdb.get_accounts(path, cid)
                cols = ["account_id", "type", "balance"]
            table = to_markdown(cols, [tuple(r[c] for c in cols) for r in rows])
            return {"answer": table, "table": {"cols": cols, "rows": len(rows)}, "trace": step(state, "account:tool_no_llm")}

        as_of = bankdb.get_as_of(path, cid)
        try:
            raw = _text(llm.invoke([("system", SQL_SYSTEM.format(schema=bankdb.SCHEMA_FOR_LLM, as_of=as_of)), ("user", q)]))
            sql = bankdb.clean_sql(raw)
            cols, rows = bankdb.run_readonly_sql(path, cid, sql)
        except bankdb.UnsafeSQL as e:
            return {"answer": "I couldn't run that query safely, so I won't guess. Try asking in a simpler way.",
                    "trace": step(state, f"account:rejected:{e}")}
        except Exception as e:
            return {"answer": "Sorry, I couldn't look that up just now.", "trace": step(state, f"account:error:{type(e).__name__}")}

        if rows == [("unsupported",)]:
            return {"answer": CANNED["out_of_scope"], "sql": sql, "trace": step(state, "account:unsupported")}
        if not rows:
            return {"answer": "No matching transactions found.", "sql": sql, "trace": step(state, "account:empty")}

        table = to_markdown(cols, rows)
        try:
            summary = _text(llm.invoke([("system", SUMMARY_SYSTEM), ("user", f"Question: {q}\n\n{table}")]))
        except Exception:
            summary = ""
        return {"answer": (summary + "\n\n" if summary else "") + table, "sql": sql,
                "table": {"cols": cols, "rows": len(rows)}, "trace": step(state, "account:sql_ok")}

    def oos(state: State) -> dict[str, Any]:
        return {"answer": CANNED["out_of_scope"], "citations": [], "trace": step(state, "out_of_scope")}

    def finalize(state: State) -> dict[str, Any]:
        answer, found = mask_pii(state.get("answer", ""))
        latency = int((time.time() - state.get("started", time.time())) * 1000)
        log_event(settings.audit_log, {
            "route": state.get("route"), "guard": state.get("guard_action"),
            "question": state.get("safe_question"), "pii_masked": state.get("pii_found", []),
            "sql": state.get("sql"), "trace": state.get("trace"), "latency_ms": latency,
            "sources": [(c["source"], c["page"]) for c in state.get("citations", []) if c.get("cited")],
            "model": settings.groq_model if llm is not None else None,
        })
        return {"answer": answer, "latency_ms": latency, "trace": step(state, "done" + (":answer_masked" if found else ""))}

    g = StateGraph(State)
    for name, fn in [("input_guard", input_guard), ("canned", canned), ("classify", classify), ("rewrite", rewrite),
                     ("rag", rag), ("account", account), ("oos", oos), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", lambda s: "classify" if s["guard_action"] == "allow" else "canned",
                            {"classify": "classify", "canned": "canned"})
    g.add_conditional_edges("classify", lambda s: s["route"], {"policy": "rewrite", "account": "account", "other": "oos"})
    g.add_edge("rewrite", "rag")
    for n in ("canned", "rag", "account", "oos"):
        g.add_edge(n, "finalize")
    g.add_edge("finalize", END)
    return g.compile()


class Assistant:
    def __init__(self, retriever: HybridRetriever, llm: Any | None, settings: Settings) -> None:
        self.graph = build_graph(retriever, llm, settings)
        self.llm_enabled = llm is not None

    def ask(self, question: str, customer_id: str | None = None, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        masked_history = [{"role": m["role"], "content": mask_pii(m["content"])[0]} for m in (history or [])]
        return self.graph.invoke({"question": question, "customer_id": customer_id or "", "history": masked_history})
