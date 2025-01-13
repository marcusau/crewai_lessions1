from crewai import Agent, Crew, Process, Task, LLM
import os,json,yaml
from dotenv import load_dotenv
from crewai_tools import SerperDevTool
load_dotenv()

os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
OPENAI_MODEL_NAME = "gpt-4" # Use standard model name

with open('config/agents.yaml' , 'r') as file:
    agents_config = yaml.safe_load(file)

with open('config/tasks.yaml' , 'r') as file:
    tasks_config = yaml.safe_load(file)

print(tasks_config['research_task'])

llm = LLM(
    model=OPENAI_MODEL_NAME,
    temperature=0,
    max_tokens=4096,
    #api_base="https://api.openai.com/v1", # Explicitly set API base
    api_key=os.getenv("OPENAI_API_KEY")
)

researcher = Agent(
    config=agents_config['researcher'],
    verbose=True,
    llm=llm,
    allow_delegation=False,
    tools=[SerperDevTool()]
)

reporting_analyst = Agent(
    config=agents_config['reporting_analyst'],
    verbose=True,
    llm=llm,
    allow_delegation=False
)

research_task = Task(
    description=tasks_config['research_task']['description'],
    expected_output=tasks_config['research_task']['expected_output'],
    agent=researcher
)

reporting_task = Task( 
    description=tasks_config['reporting_task']['description'],
    expected_output=tasks_config['reporting_task']['expected_output'],
    agent=reporting_analyst,
    output_file='report.md'
)

crew = Crew(
    agents=[researcher, reporting_analyst],
    tasks=[research_task, reporting_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff(inputs={'topic': 'AI Agents'})
print(result)