import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from schemas import SessionState
from retrieval import SimulatedRetriever
from tools import get_all_tools, ToolLogger
from agent import create_workflow, AgentState
from prompts import MEMORY_SUMMARY_PROMPT

# Project root (the folder holding main.py), so logs/ and sessions/ land in the same
# place whatever directory the assistant is launched from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BASE_URL = "https://openai.vocareum.com/v1"


class DocumentAssistant:
    """
    The assistant creates and loads sessions and
    stores state/session data within a file.
    """

    def __init__(
            self,
            openai_api_key: str,
            model_name: str = "gpt-4o",
            temperature: float = 0.1,
            session_storage_path: str = os.path.join(PROJECT_ROOT, "sessions"),
            logs_path: str = os.path.join(PROJECT_ROOT, "logs"),
            base_url: Optional[str] = None,
            llm=None,
    ):
        # Initialize LLM (an already-built chat model can be injected, e.g. for tests)
        self.llm = llm or ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            temperature=temperature,
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        )

        # Initialize components
        self.retriever = SimulatedRetriever()
        self.tool_logger = ToolLogger(logs_dir=logs_path)
        self.tools = get_all_tools(self.retriever, self.tool_logger)

        # Create workflow (compiled with checkpointer inside create_workflow)
        self.workflow = create_workflow(self.llm, self.tools)

        # Session management
        self.session_storage_path = session_storage_path
        os.makedirs(session_storage_path, exist_ok=True)

        # Current session
        self.current_session: Optional[SessionState] = None

    def start_session(self, user_id: str, session_id: Optional[str] = None) -> str:
        """Start a new session or resume an existing one."""
        if session_id and self._session_exists(session_id):
            # Load existing session
            self.current_session = self._load_session(session_id)
            self._restore_graph_state()
            print(f"Resumed session {session_id}")
        else:
            # Create new session
            session_id = session_id or str(uuid.uuid4())
            self.current_session = SessionState(
                session_id=session_id,
                user_id=user_id,
                conversation_history=[],
                document_context=[]
            )
            print(f"Started new session {session_id}")
        # One tool-call log per session: logs/session_<id>.json
        self.tool_logger.set_session(session_id)
        return session_id

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Saved sessions, newest first, for the /sessions command."""
        sessions = []
        for name in os.listdir(self.session_storage_path):
            if not name.endswith(".json"):
                continue
            try:
                s = self._load_session(name[:-5])
            except Exception:
                continue
            sessions.append({"session_id": s.session_id, "user_id": s.user_id,
                             "turns": len(s.conversation_history), "last_updated": s.last_updated})
        return sorted(sessions, key=lambda s: s["last_updated"], reverse=True)

    def _config(self) -> Dict[str, Any]:
        return {
            "configurable": {
                "thread_id": self.current_session.session_id,
                "llm": self.llm,
                "tools": self.tools,
            }
        }

    def _restore_graph_state(self) -> None:
        """
        InMemorySaver lives only as long as the process. When a session is resumed from
        disk, seed the checkpointer with the saved turns so the conversation carries on
        with its history, summary and active documents intact.
        """
        history = self.current_session.conversation_history
        if not history:
            return
        messages: List[BaseMessage] = []
        for turn in history:
            messages.append(HumanMessage(content=turn.get("user_input", "")))
            if turn.get("response"):
                messages.append(AIMessage(content=turn["response"]))
        self.workflow.update_state(
            self._config(),
            {
                "messages": messages,
                "conversation_summary": history[-1].get("summary", ""),
                "active_documents": self.current_session.document_context,
                "session_id": self.current_session.session_id,
                "user_id": self.current_session.user_id,
            },
            as_node="update_memory",
        )

    def _session_exists(self, session_id: str) -> bool:
        filepath = os.path.join(self.session_storage_path, f"{session_id}.json")
        return os.path.exists(filepath)

    def _load_session(self, session_id: str) -> SessionState:
        filepath = os.path.join(self.session_storage_path, f"{session_id}.json")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return SessionState(**data)

    def _save_session(self) -> None:
        if self.current_session:
            filepath = os.path.join(
                self.session_storage_path,
                f"{self.current_session.session_id}.json"
            )
            # mode="json" turns datetimes into ISO strings; conversation_history holds
            # only JSON-safe turn records (see _turn_record), never raw graph state.
            session_dict = self.current_session.model_dump(mode="json")

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_dict, f, indent=2, ensure_ascii=False)

    def _get_conversation_summary(self, config) -> str:
        if not self.current_session or not self.current_session.conversation_history:
            return "No previous conversation."

        current_state = self.workflow.get_state(config).values

        summary = current_state.get("conversation_summary", "")
        return summary

    def _get_conversation_history(self, config) -> List[BaseMessage]:
        if not self.current_session or not self.current_session.conversation_history:
            return []

        current_state = self.workflow.get_state(config).values

        history = current_state.get("messages", [])
        return history

    @staticmethod
    def _turn_record(user_input: str, final_state: Dict[str, Any], actions: List[str],
                     response: Optional[str]) -> Dict[str, Any]:
        """A JSON-safe summary of one turn for the session file."""
        intent = final_state.get("intent")
        return {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "intent": intent.model_dump(mode="json") if intent is not None else None,
            "actions_taken": actions,
            "tools_used": final_state.get("tools_used", []),
            "response": response,
            "structured_response": final_state.get("current_response"),
            "active_documents": final_state.get("active_documents", []),
            "summary": final_state.get("conversation_summary", ""),
        }

    def process_message(self, user_input: str) -> Dict[str, Any]:
        """Process a user message using the LangGraph workflow."""

        if not self.current_session:
            raise ValueError("No active session. Call start_session() first.")

        # thread_id ties every turn of this session to the same checkpoint thread;
        # llm and tools are read by the nodes from config["configurable"].
        config = {
            "configurable": {
                "thread_id": self.current_session.session_id,
                "llm": self.llm,
                "tools": self.tools,
            }
        }

        # actions_taken accumulates across turns (operator.add + checkpointer), so note
        # where this turn starts in order to report this turn's nodes separately.
        previous_actions = self.workflow.get_state(config).values.get("actions_taken", [])

        initial_state: AgentState = {
            "messages": [],
            "user_input": user_input,
            "intent": None,
            "next_step": "classify_intent",
            "conversation_summary": self._get_conversation_summary(config),
            "active_documents": self.current_session.document_context,
            "current_response": None,
            "tools_used": [],
            "session_id": self.current_session.session_id,
            "user_id": self.current_session.user_id,
            # Nothing to add yet: the operator.add reducer appends each node's entry
            "actions_taken": []
        }
        try:
            # Invoke the workflow with a thread_id equal to the session_id
            final_state = self.workflow.invoke(initial_state, config=config)
            all_actions = final_state.get("actions_taken", [])
            turn_actions = all_actions[len(previous_actions):]
            response = final_state.get("messages")[-1].content if final_state.get("messages") else None

            # Update session with new state
            if final_state.get("messages"):
                self.current_session.conversation_history.append(
                    self._turn_record(user_input, final_state, turn_actions, response)
                )
                self.current_session.last_updated = datetime.now()
                if final_state.get("active_documents"):
                    self.current_session.document_context = list(dict.fromkeys(
                        self.current_session.document_context +
                        final_state["active_documents"]
                    ))
                self._save_session()
            return {
                "success": True,
                "response": response,
                "intent": final_state.get("intent").model_dump() if final_state.get("intent") else None,
                "tools_used": final_state.get("tools_used", []),
                "sources": final_state.get("active_documents", []),
                "active_documents": final_state.get("active_documents", []),
                "structured_response": final_state.get("current_response"),
                "actions_taken": turn_actions,
                "all_actions_taken": all_actions,
                "summary": final_state.get("conversation_summary", "")
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": None
            }
