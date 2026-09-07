# agent.py
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    langchain,   # <-- this is key
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
    bey
)
from livekit.agents.inference import TurnDetector

from graph import create_workflow  # <-- our compiled LangGraph app

load_dotenv(".env.local")

class InterviewAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=(
            "You are a professional interviewer conducting a job interview. "
            "The LangGraph workflow will drive the conversation flow. "
            "Simply speak the questions and responses as they come from the graph. "
            "Be conversational, professional, and helpful throughout the interview process."
        ))

async def entrypoint(ctx: agents.JobContext):
    print(f"--- New Job Started: {ctx.job.id} ---")
    # 1) Build/compile the LangGraph app (Runnable)
    print("Building LangGraph workflow...")
    try:
        interview_workflow = create_workflow()
        print("✅ Workflow built successfully")
    except Exception as e:
        print(f"❌ Workflow build failed: {e}")
        return

    # 2) Wrap it as an LLM for LiveKit via the LangChain plugin
    lg_llm = langchain.LLMAdapter(graph=interview_workflow)
    print("✅ LLM Adapter created")

    # 3) Configure the rest of the realtime pipeline
    print("Configuring agent session...")
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=lg_llm,
        tts=cartesia.TTS(model="sonic-2", voice="f786b574-daa5-4673-aa0c-cbe3e8534c02"),
        vad=silero.VAD.load(),
        turn_detection=TurnDetector.from_default(),
    )
    print("✅ Session configured")

    avatar = bey.AvatarSession(
        avatar_id="694c83e2-8895-4a98-bd16-56332ca3f449",
    )

    # Start the avatar and wait for it to join
    print("Starting avatar...")
    await avatar.start(session, room=ctx.room)
    print("✅ Avatar joined")

    print("Starting session...")
    await session.start(
        room=ctx.room,
        agent=InterviewAgent(),
        room_input_options=RoomInputOptions(
          noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    print("✅ Session started")

    # --- TRIGGER INITIAL GREETING ---
    print("Triggering initial interview greeting...")
    try:
        initial_state = {"messages": []}
        print("Invoking graph for first message...")
        result = await interview_workflow.ainvoke(initial_state)

        if not result or "messages" not in result or not result["messages"]:
            print("❌ Graph returned no messages")
            return

        first_message = result["messages"][-1].content
        print(f"AI wants to say: {first_message}")

        await session.say(first_message)
        print("✅ Greeting sent successfully!")
    except Exception as e:
        print(f"❌ Error triggering initial greeting: {e}")



if __name__ == "__main__":
    # Pre-flight Check
    print("\n--- Agent Pre-flight Check ---")
    import os
    from dotenv import load_dotenv
    load_dotenv(".env.local")

    url = os.getenv("LIVEKIT_URL")
    key = os.getenv("LIVEKIT_API_KEY")

    if not url or not key:
        print("❌ ERROR: LIVEKIT_URL or LIVEKIT_API_KEY missing in .env.local")
    else:
        print(f"✅ Target Server: {url}")
        print(f"✅ API Key: {key[:5]}...{key[-5:]}")
        print("Starting Worker... (You should see 'Worker registered' shortly)")
    print("-----------------------------\n")

    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
