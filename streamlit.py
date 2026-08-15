import os
import io
import zipfile
import socket
import functools
import threading
import http.server
import socketserver
import requests
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


def get_netlify_token():
    token = os.getenv("NETLIFY_API_TOKEN")
    if token:
        return token
    try:
        return st.secrets["NETLIFY_API_TOKEN"]
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
    llm = LLM(model="gemini/gemini-2.5-flash", api_key=get_api_key())
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


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def start_local_server(directory):
    """
    Serves the generated_website folder over http.server on a free local port.
    Runs once per Streamlit session (reused across reruns via session_state) so
    we don't try to bind the same port twice and crash.
    """
    if "local_server_port" in st.session_state:
        return st.session_state.local_server_port

    port = get_free_port()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    httpd = socketserver.TCPServer(("", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    st.session_state.local_server_port = port
    st.session_state.local_server_httpd = httpd
    return port


def zip_static_files(output_dir):
    """Zip only the static frontend files (html/css/js) — that's all Netlify can host."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in os.listdir(output_dir):
            if filename.endswith((".html", ".css", ".js")):
                filepath = os.path.join(output_dir, filename)
                zf.write(filepath, arcname=filename)
    buffer.seek(0)
    return buffer


def deploy_to_netlify(output_dir):
    """
    Deploys the generated_website folder as a zip to Netlify's 'deploy without git' endpoint.
    Netlify creates a brand new site + live URL from the zip in one call.
    Requires a free Netlify Personal Access Token (NETLIFY_API_TOKEN).
    """
    token = get_netlify_token()
    if not token:
        return None, "Missing NETLIFY_API_TOKEN. Get a free token at app.netlify.com > User settings > Applications > Personal access tokens, then add it to .env or Streamlit secrets."

    zip_buffer = zip_static_files(output_dir)

    try:
        response = requests.post(
            "https://api.netlify.com/api/v1/sites",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/zip"
            },
            data=zip_buffer.getvalue(),
            timeout=60
        )
    except requests.RequestException as e:
        return None, f"Could not reach Netlify: {e}"

    if response.status_code not in (200, 201):
        return None, f"Netlify deploy failed ({response.status_code}): {response.text}"

    data = response.json()
    live_url = data.get("ssl_url") or data.get("url")
    return live_url, None


st.set_page_config(page_title="AI Website Builder", page_icon="🤖")
st.title("AI Website Builder Agent")
st.write("Describe the website you want. 4 agents will plan, design, build, and deploy it live.")

user_input = st.text_area(
    "Website description",
    placeholder="A portfolio website for a photographer, dark theme, gallery and contact form"
)

view_mode = st.radio(
    "How do you want to see the result?",
    ["Preview locally (localhost)", "Deploy live (Netlify link)"],
    help="Local preview only works when you run this app on your own machine with `streamlit run app.py`. "
         "It will NOT work if this app itself is hosted on Streamlit Cloud, because 'localhost' would then "
         "point to Streamlit's server, not your browser. Use Netlify for anything you need to share or is deployed remotely."
)

if st.button("Build Website", type="primary"):
    if not user_input.strip():
        st.warning("Type a description first.")
    else:
        with st.spinner("Agents working... this can take a minute"):
            crew = build_crew(user_input)
            crew.kickoff()

        index_path = os.path.join(OUTPUT_DIR, "index.html")
        if not os.path.exists(index_path):
            st.error("The agents didn't produce an index.html file. Check your crew output above.")
        elif view_mode == "Preview locally (localhost)":
            port = start_local_server(os.path.abspath(OUTPUT_DIR))
            local_url = f"http://localhost:{port}"
            st.success("Your website is running locally!")
            st.markdown(f"### 🔗 [{local_url}]({local_url})")
            st.caption("This link only works on this machine, in this session. It stops when the app process stops.")
        else:
            with st.spinner("Deploying your website to a live link..."):
                live_url, error = deploy_to_netlify(OUTPUT_DIR)

            if error:
                st.error(error)
                st.info("Deployment failed, so here's the generated code instead.")
                with open(index_path, "r", encoding="utf-8") as f:
                    st.components.v1.html(f.read(), height=600, scrolling=True)
                for filename in os.listdir(OUTPUT_DIR):
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                    with st.expander(filename):
                        st.code(content, language="html" if filename.endswith((".html", ".css", ".js")) else "python")
            else:
                st.success("Your website is live!")
                st.markdown(f"### 🔗 [{live_url}]({live_url})")
                st.caption("Note: only the frontend (HTML/CSS/JS) is deployed. If the backend agent generated app.py, that needs a separate host like Render or Railway since Netlify only serves static files.")

                with st.expander("View generated files (optional)"):
                    for filename in os.listdir(OUTPUT_DIR):
                        filepath = os.path.join(OUTPUT_DIR, filename)
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()
                        st.write(f"**{filename}**")
                        st.code(content, language="html" if filename.endswith((".html", ".css", ".js")) else "python")
                        st.download_button(
                            label=f"Download {filename}",
                            data=content,
                            file_name=filename,
                            key=f"download_{filename}"
                        )