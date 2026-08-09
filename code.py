import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

OUTPUT_DIR = "generated_website"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class FileWriterInput(BaseModel):
    filename: str = Field(..., description="File name, e.g. index.html, style.css, app.py")
    content: str = Field(..., description="Full text content to write into the file")


class FileWriterTool(BaseTool):
    name: str = "file_writer"
    description: str = "Saves code content to a file inside the generated_website folder"
    args_schema: type[BaseModel] = FileWriterInput

    def _run(self, filename: str, content: str) -> str:
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Saved {filename} ({len(content)} chars)"


file_writer = FileWriterTool()
 
llm = LLM(
    model="gemini/gemini-2.5-pro",
    api_key=os.getenv("GEMINI_API_KEY")
)

user_input = input("Enter the type of website you want to build: ")

analyst_agent = Agent(
    role="Requirement Analyst",
    goal="Understand what website the user wants and turn it into one detailed master prompt for the design team",
    backstory="You are a senior business analyst who interviews clients and converts vague ideas into clear technical briefs.",
    llm=llm,
    verbose=False
)

analyst_task = Task(
    description=f"""
    The user said: "{user_input}"

    Get the user input and give the master prompt to the next agent.
    Turn this into one detailed master prompt covering:
    - website purpose and target audience
    - required pages and sections
    - tone and style (professional, playful, minimal, etc.)
    - any features mentioned (forms, login, cart, booking, etc.)

    Output ONLY the master prompt as plain text.
    """,
    expected_output="A detailed master prompt describing the website requirements",
    agent=analyst_agent
)

uiux_agent = Agent(
    role="UI/UX Designer",
    goal="Get the master prompt and start building the website by producing a full design specification",
    backstory="You are a senior UI/UX designer who converts requirement briefs into concrete design specs: color palette, typography, layout and components.",
    llm=llm,
    verbose=False
)

uiux_task = Task(
    description="""
    Get the master prompt and start building the website.
    Using the master prompt from the previous agent, produce a design specification with:
    - color palette (hex codes)
    - font choices
    - section-by-section layout
    - reusable UI components needed

    Output ONLY the design specification as plain text.
    """,
    expected_output="A structured design specification",
    agent=uiux_agent,
    context=[analyst_task]
)

frontend_agent = Agent(
    role="Frontend Developer",
    goal="Choose the modern tech stack for frontend and start coding the website",
    backstory="You are a senior frontend developer who writes clean, production-ready HTML, CSS and JavaScript.",
    llm=llm,
    tools=[file_writer],
    verbose=False
)

frontend_task = Task(
    description="""
    Agent 3: website frontend.
    Choose the modern tech stack for frontend and start coding.
    Using the design specification, build a complete responsive website in plain HTML, CSS and JavaScript
    so it runs instantly with no build step.

    You MUST use the file_writer tool to save:
    - index.html
    - style.css
    - script.js

    Build it fully, not a placeholder.
    """,
    expected_output="Confirmation that index.html, style.css and script.js were saved",
    agent=frontend_agent,
    context=[uiux_task]
)

backend_agent = Agent(
    role="Backend Developer",
    goal="Build the functionality of the website using a lightweight backend",
    backstory="You are a backend developer who builds minimal Flask APIs to power frontend features like contact forms.",
    llm=llm,
    tools=[file_writer],
    verbose=False
)

backend_task = Task(
    description="""
    Agent 4: backend (optional).
    Build the functionality of the website.
    Based on the frontend code, identify any feature that genuinely needs a backend
    (contact form, newsletter signup, booking, etc.) and build a minimal Flask app for it.

    Use the file_writer tool to save app.py.
    If no backend feature is truly needed, save NOTE.txt explaining why not.
    """,
    expected_output="Confirmation that app.py or NOTE.txt was saved",
    agent=backend_agent,
    context=[frontend_task]
)

crew = Crew(
    agents=[analyst_agent, uiux_agent, frontend_agent, backend_agent],
    tasks=[analyst_task, uiux_task, frontend_task, backend_task],
    process=Process.sequential,
    verbose=False
)

result = crew.kickoff()
print(result)