import os
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

load_dotenv()


def get_api_key():
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None

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


def build_crew(user_input: str):
    llm = LLM(model="gemini/gemini-2.5-pro", api_key=get_api_key())
    file_writer = FileWriterTool()

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
        backstory="You are a senior UI/UX designer who converts requirement briefs into concrete design specs.",
        llm=llm,
        verbose=False
    )
    uiux_task = Task(
        description="""
        Get the master prompt and start building the website.
        Produce a design specification with color palette (hex codes), fonts,
        section-by-section layout, reusable UI components, and a visual style
        (e.g. glassmorphism, minimal, bold gradients).
        For each section, specify which Font Awesome icon fits it
        (e.g. fa-solid fa-code for a services section, fa-solid fa-envelope for contact).
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
        Choose the modern tech stack for frontend and start coding.
        Build a complete responsive website in plain HTML, CSS and JavaScript.

        Make it visually attractive:
        - Include Font Awesome via CDN in the HTML head:
          <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
        - Use relevant <i class="fa-solid fa-..."></i> icons in the navbar, feature sections,
          buttons, contact info and footer (not just text labels)
        - Add hover effects, smooth scroll, subtle box-shadows and rounded corners
        - Add a hero section with a clear heading, subheading and call-to-action button with an icon
        - Use CSS variables for the color palette so the theme stays consistent across the file
        - Add fade-in or slide-in animations on scroll using plain JavaScript (IntersectionObserver)
        - Ensure spacing and typography look modern, not cramped default browser styles

        You MUST use the file_writer tool to save index.html, style.css, script.js.
        Build it fully, not a placeholder.
        """,
        expected_output="Confirmation that index.html, style.css and script.js were saved, using icons and animations",
        agent=frontend_agent,
        context=[uiux_task]
    )

    backend_agent = Agent(
        role="Backend Developer",
        goal="Build the functionality of the website using a lightweight backend",
        backstory="You are a backend developer who builds minimal Flask APIs to power frontend features.",
        llm=llm,
        tools=[file_writer],
        verbose=False
    )
    backend_task = Task(
        description="""
        Identify any feature that genuinely needs a backend (contact form, signup, booking)
        and build a minimal Flask app for it using the file_writer tool to save app.py.
        If no backend feature is truly needed, save NOTE.txt explaining why not.
        """,
        expected_output="Confirmation that app.py or NOTE.txt was saved",
        agent=backend_agent,
        context=[frontend_task]
    )

    return Crew(
        agents=[analyst_agent, uiux_agent, frontend_agent, backend_agent],
        tasks=[analyst_task, uiux_task, frontend_task, backend_task],
        process=Process.sequential,
        verbose=False
    )


st.set_page_config(page_title="AI Website Builder", page_icon="🤖")
st.title("AI Website Builder Agent")
st.write("Describe the website you want. 4 agents will plan, design and build it.")

user_input = st.text_area(
    "Website description",
    placeholder="A portfolio website for a photographer, dark theme, gallery and contact form"
)

if st.button("Build Website", type="primary"):
    if not user_input.strip():
        st.warning("Type a description first.")
    else:
        with st.spinner("Agents working... this can take a minute"):
            crew = build_crew(user_input)
            result = crew.kickoff()

        st.success("Done")
        st.subheader("Final Output")
        st.write(result)

        index_path = os.path.join(OUTPUT_DIR, "index.html")
        if os.path.exists(index_path):
            st.subheader("Live Preview")
            with open(index_path, "r", encoding="utf-8") as f:
                html_code = f.read()
            st.components.v1.html(html_code, height=600, scrolling=True)

            st.subheader("Generated Files")
            for filename in os.listdir(OUTPUT_DIR):
                filepath = os.path.join(OUTPUT_DIR, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                with st.expander(filename):
                    st.code(content, language="html" if filename.endswith((".html", ".css", ".js")) else "python")