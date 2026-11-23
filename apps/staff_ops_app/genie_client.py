"""Thin wrapper over the Genie Conversation API (databricks-sdk `w.genie`), with REST fallbacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import pandas as pd

from data import statement_to_frame

SAMPLE_QUESTIONS = [
    "Quel est le taux d'occupation des créneaux de midi la semaine dernière ?",
    "Which hours on weekdays have the lowest occupancy in October?",
    "How much revenue did rainy Saturdays bring vs dry Saturdays?",
    "Combien d'annulations tardives ce mois-ci ?",
    "Which players have a Pass Chrono expiring in the next 14 days?",
]


@dataclass
class GenieAnswer:
    conversation_id: str
    message_id: str
    status: str
    text: str = ""
    sql: str = ""
    sql_description: str = ""
    attachment_id: str = ""
    result: pd.DataFrame | None = None
    suggested_questions: list[str] = field(default_factory=list)
    error: str = ""


def _as_dict(obj: Any) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    return obj.as_dict()


def parse_message(message: Any) -> GenieAnswer:
    """Extract text, SQL and follow-up suggestions from a GenieMessage (SDK object or REST dict)."""
    m = _as_dict(message)
    answer = GenieAnswer(
        conversation_id=m.get("conversation_id", ""),
        message_id=m.get("message_id") or m.get("id", ""),
        status=str(m.get("status", "")),
    )
    texts = []
    for att in m.get("attachments") or []:
        if att.get("text"):
            texts.append(att["text"].get("content", ""))
        if att.get("query"):
            answer.sql = att["query"].get("query", "")
            answer.sql_description = att["query"].get("description", "")
            answer.attachment_id = att.get("attachment_id", "")
        if att.get("suggested_questions"):
            answer.suggested_questions = list(att["suggested_questions"].get("questions") or [])
    answer.text = "\n\n".join(t for t in texts if t)
    if m.get("error"):
        err = m["error"]
        answer.error = err.get("error", "") if isinstance(err, dict) else str(err)
    return answer


class GenieClient:
    def __init__(self, client: Any, space_title: str, space_id: str = ""):
        self.w = client
        self.space_title = space_title
        self._space_id = space_id or None

    def space_id(self) -> str:
        if self._space_id:
            return self._space_id
        token = None
        while True:
            if hasattr(self.w.genie, "list_spaces"):
                resp = self.w.genie.list_spaces(page_token=token).as_dict()
            else:
                query = {"page_token": token} if token else None
                resp = self.w.api_client.do("GET", "/api/2.0/genie/spaces", query=query)
            for s in resp.get("spaces") or []:
                if s.get("title") == self.space_title:
                    self._space_id = s["space_id"]
                    return self._space_id
            token = resp.get("next_page_token")
            if not token:
                raise LookupError(
                    f"No Genie space titled '{self.space_title}' is visible to this identity. "
                    "Run the setup_genie job task, which grants the app CAN_RUN."
                )

    def ask(self, question: str, conversation_id: str | None = None, timeout_s: int = 180) -> GenieAnswer:
        sid = self.space_id()
        timeout = timedelta(seconds=timeout_s)
        if conversation_id:
            message = self.w.genie.create_message_and_wait(sid, conversation_id, question, timeout=timeout)
        else:
            message = self.w.genie.start_conversation_and_wait(sid, question, timeout=timeout)
        answer = parse_message(message)
        if answer.sql and answer.attachment_id:
            answer.result = self._query_result(sid, answer.conversation_id, answer.message_id, answer.attachment_id)
        return answer

    def _query_result(self, sid: str, conv: str, msg: str, attachment_id: str) -> pd.DataFrame | None:
        g = self.w.genie
        for name in ("get_message_attachment_query_result", "get_message_query_result_by_attachment"):
            if hasattr(g, name):
                resp = getattr(g, name)(sid, conv, msg, attachment_id)
                break
        else:
            from databricks.sdk.service.dashboards import GenieGetMessageQueryResultResponse

            raw = self.w.api_client.do(
                "GET",
                f"/api/2.0/genie/spaces/{sid}/conversations/{conv}/messages/{msg}/attachments/{attachment_id}"
                "/query-result",
            )
            resp = GenieGetMessageQueryResultResponse.from_dict(raw)
        statement = resp.statement_response
        if statement is None:
            return None
        return statement_to_frame(statement)
