import asyncio
from decouple import config 
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,RunConfig

# 🔐 Load environment variables using python-decouple
gemini_api_key = config('GEMINI_API_KEY')

if not gemini_api_key:
    raise ValueError("🚨 GEMINI_API_KEY environment variable is not set.")

# 🌐 Initialize Gemini API client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# 🤖 Setup the model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# ⚙️ Configure run settings
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

agent = Agent(
    name="FAQ Bot",
    instructions="""
    💡 You are a helpful FAQ bot. Your purpose is to answer questions about yourself and assist users.
    
    Here are some predefined questions and answers you should know:
    - What is your name? → I'm FAQ Bot, your helpful assistant!
    - What can you do? → I can answer questions about myself and provide helpful information on various topics.
    - Who created you? → I was created by a developer using the OpenAI Agent SDK.
    - How do I exit? → You can type 'exit' or 'quit' to end our conversation.
    - What is your purpose? → My purpose is to assist users by answering their questions in a friendly and helpful manner.
    - When were you launched? → I was launched recently as part of a programming assignment to create a FAQ bot.
    - What version are you? → I'm currently version 1.0, created using OpenAI Agent SDK and Gemini API.
    
    Be friendly, engaging, and helpful in all your responses. If you're asked about something outside your knowledge, 
    politely indicate that you're designed primarily to answer FAQ questions.
    """,
    model=model
)

# 🚀 Main interaction loop
async def main():
    print("\n🤖 Welcome to the **FAQ Bot**!")
    print("💬 Ask me anything about myself.")
    print("🔚 Type 'exit' or 'quit' to end the conversation.\n")

    while True:
        user_question = input("\n👤 You: ").strip()
        if user_question.lower() in ["exit", "quit"]:
            print("👋 Goodbye! Have a great day!")
            break
        
        # Use the agent for all questions
        result = await Runner.run(agent, user_question, run_config=config)
        print("🤖 Bot:", result.final_output)


asyncio.run(main())
