import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from print_color import print

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Windows consoles default to cp1252, which cannot print the emoji below.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from src.assistant import DocumentAssistant


def print_header():
    """Print a nice header"""
    print("\n" + "=" * 60)
    print("DocDacity Intelligent Document Assistant", color='blue')
    print("=" * 60 + "\n")


def print_help():
    """Print help information"""
    print("\nAVAILABLE COMMANDS:", color='blue')
    print("  /help     - Show this help message")
    print("  /docs     - List available documents")
    print("  /sessions - List saved sessions (resume one by entering its ID at startup)")
    print("  /quit     - Exit the assistant")
    print("\nExample queries:")
    print("  - What's the total amount in invoice INV-001?")
    print("  - Summarize all contracts")
    print("  - Calculate the sum of all invoice totals")
    print("  - Find documents with amounts over $50,000")
    print()


def list_documents(assistant: DocumentAssistant):
    """List all available documents"""
    print("\nAVAILABLE DOCUMENTS:", color='blue')
    print("-" * 40)

    for doc_id, doc in assistant.retriever.documents.items():
        print(f"ID: {doc_id}")
        print(f"Title: {doc.title}")
        print(f"Type: {doc.doc_type}")
        if 'total' in doc.metadata:
            print(f"Total: ${doc.metadata['total']:,.2f}")
        elif 'amount' in doc.metadata:
            print(f"Amount: ${doc.metadata['amount']:,.2f}")
        elif 'value' in doc.metadata:
            print(f"Value: ${doc.metadata['value']:,.2f}")
        print("-" * 40)


def list_sessions(assistant: DocumentAssistant):
    """List saved sessions"""
    print("\nSAVED SESSIONS:", color='blue')
    sessions = assistant.list_sessions()
    if not sessions:
        print("  (none yet)")
    for s in sessions:
        print(f"  {s['session_id']}  user={s['user_id']}  turns={s['turns']}  "
              f"last={s['last_updated']:%Y-%m-%d %H:%M}")


def print_structured(structured):
    """Show the key fields of the specialist's structured (Pydantic) response"""
    if not isinstance(structured, dict) or "error" in structured:
        return
    if "answer" in structured and "confidence" in structured:
        print(f"\nANSWER CONFIDENCE: {structured['confidence']:.2f}", color='green')
    if "expression" in structured:
        units = f" {structured['units']}" if structured.get("units") else ""
        print(f"\nCALCULATION: {structured['expression']} = {structured['result']:,.2f}{units}", color='green')
    if structured.get("key_points"):
        print(f"\nKEY POINTS: {len(structured['key_points'])} extracted from "
              f"{', '.join(structured.get('document_ids') or []) or 'the documents'}", color='green')


def main():
    """Main interactive loop"""
    # Load environment variables
    load_dotenv()

    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key")
        return

    # Print header
    print_header()

    # Create assistant
    print(" INITIALIZING ASSISTANT...", color='green')
    assistant = DocumentAssistant(
        openai_api_key=api_key,
        model_name=os.getenv("MODEL_NAME", "gpt-4o"),
        temperature=float(os.getenv("TEMPERATURE", "0.1")),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        **({"session_storage_path": os.getenv("SESSION_STORAGE_PATH")}
           if os.getenv("SESSION_STORAGE_PATH") else {}),
    )

    # Start session
    user_id = input("Enter your user ID (or press Enter for 'demo_user'): ").strip() or "demo_user"
    resume_id = input("Session ID to resume (or press Enter for a new session): ").strip() or None
    session_id = assistant.start_session(user_id, resume_id)
    print(f"Session started: {session_id}")

    # Show help
    print_help()

    # Main interaction loop
    while True:
        try:
            # Get user input
            user_input = input("\nEnter Message: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "/quit":
                print("\nGoodbye!", color='blue')
                break
            elif user_input.lower() == "/help":
                print_help()
                continue
            elif user_input.lower() == "/docs":
                list_documents(assistant)
                continue
            elif user_input.lower() == "/sessions":
                list_sessions(assistant)
                continue

            # Process the message
            print("\nProcessing...", color='yellow')
            result = assistant.process_message(user_input)

            if result["success"]:
                print("\n🤖 Assistant:", end=" ")

                if result.get("response"):
                    print(result["response"])
                if result.get("intent"):
                    intent = result["intent"]
                    print(f"\nINTENT: {intent['intent_type']} (confidence {intent['confidence']:.2f})", color='green')
                    print(f"REASONING: {intent['reasoning']}", color='green')
                print_structured(result.get("structured_response"))
                if result.get("actions_taken"):
                    print(f"\nGRAPH PATH: {' -> '.join(result['actions_taken'])}", color='yellow')
                if result.get("active_documents"):
                    print(f"\nSOURCES: {', '.join(result['active_documents'])}", color='blue')
                if result.get("tools_used"):
                    print(f"\nTOOLS USED: {', '.join(result['tools_used'])}", color='magenta')
                if result.get("summary"):
                    print(f"\nCONVERSATION SUMMARY: {result['summary']}", color='cyan')


            else:
                print(f"\nError: {result.get('error', 'Unknown error')}", color='red')

        except KeyboardInterrupt:
            print("\n\nGoodbye!", color='blue')
            break
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}", color='red')


if __name__ == "__main__":
    main()
