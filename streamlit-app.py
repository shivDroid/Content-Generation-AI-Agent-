import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="CrewAI Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-color: #1E90FF;
    }
    .research-output {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .blog-output {
        background-color: #e6f3ff;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

def create_agents_and_tasks(topic):
    # Initialize tools
    llm = LLM(model="gpt-4")
    search_tool = SerperDevTool(n=5)
    
    # Create agents
    senior_research_analyst = Agent(
        role="Senior Research Analyst",
        goal="Research, analyze and synthesize",
        backstory="Expert researcher with years of experience in analyzing industry trends",
        allow_delegation=False,
        verbose=True,
        tools=[search_tool],
        llm=llm
    )
    
    content_writer = Agent(
        role="Content Writer",
        goal="Transform research findings into engaging blog post while maintaining accuracy",
        backstory="Experienced content writer specializing in technical and industry analysis",
        allow_delegation=False,
        verbose=True,
        llm=llm
    )
    
    # Create tasks
    research_task = Task(
        description=f"""1. Conduct comprehensive research on {topic} including:
            - Recent developments and news
            - Key industry trends and innovations
            - Expert opinions and analyses
            - Statistical data and market insights
            2. Evaluate source credibility and fact-check all information
            3. Organize findings into a structured research brief
            4. Include all relevant citations and sources""",
        expected_output="""A detailed research report containing:
            - Executive summary of key findings
            - Comprehensive analysis of current trends and developments
            - List of verified facts and statistics
            - All citations and links to original sources
            - Clear categorization of main themes and patterns""",
        agent=senior_research_analyst
    )
    
    writing_task = Task(
        description="""Using the research brief provided, create an engaging blog post that:
            1. Transforms technical information into accessible content
            2. Maintains all factual accuracy and citations from the research
            3. Includes:
                - Attention-grabbing introduction
                - Well-structured body sections with clear headings
                - Compelling conclusion
            4. Preserves all source citations in [Source: URL] format
            5. Includes a References section at the end""",
        expected_output="""A polished blog post in markdown format that:
            - Engages readers while maintaining accuracy
            - Contains properly structured sections
            - Includes inline citations hyperlinked to the original source URL
            - Presents information in an accessible yet informative way
            - Follows proper markdown formatting""",
        agent=content_writer
    )
    
    return senior_research_analyst, content_writer, research_task, writing_task

def run_crew(topic):
    senior_research_analyst, content_writer, research_task, writing_task = create_agents_and_tasks(topic)
    
    crew = Crew(
        agents=[senior_research_analyst, content_writer],
        tasks=[research_task, writing_task],
        verbose=True
    )
    
    return crew.kickoff(inputs={"topic": topic})

# Sidebar
with st.sidebar:
    st.title("🤖 CrewAI Dashboard")
    st.markdown("---")
    
    # API Key Input
    api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    serper_api_key = st.text_input("Enter your SerperDev API Key:", type="password")
    if serper_api_key:
        os.environ["SERPER_API_KEY"] = serper_api_key
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses CrewAI to:
    1. Research your topic
    2. Generate a detailed analysis
    3. Create a blog post
    """)

# Main content
st.title("🔍 AI Research & Content Generation")

# Topic input
topic = st.text_input("Enter your research topic:", value="Medical Industry using Generative AI")

# Process button
if st.button("Start Research & Content Generation"):
    if not api_key or not serper_api_key:
        st.error("Please enter both API keys in the sidebar first!")
    else:
        try:
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Update status
            status_text.text("🔍 Initiating research process...")
            progress_bar.progress(20)
            time.sleep(1)
            
            status_text.text("📊 Analyzing data and generating insights...")
            progress_bar.progress(40)
            
            # Run CrewAI
            result = run_crew(topic)
            
            status_text.text("✍️ Creating content...")
            progress_bar.progress(80)
            time.sleep(1)
            
            # Split research and blog content
            try:
                research_content, blog_content = result.split("BLOG POST:", 1)
            except ValueError:
                research_content = result
                blog_content = ""
            
            status_text.text("✅ Process completed!")
            progress_bar.progress(100)
            
            # Display results in tabs
            tab1, tab2 = st.tabs(["📊 Research Report", "📝 Blog Post"])
            
            with tab1:
                st.markdown("### Research Findings")
                st.markdown(f'<div class="research-output">{research_content}</div>', 
                          unsafe_allow_html=True)
                
                # Export button for research
                st.download_button(
                    label="Download Research Report",
                    data=research_content,
                    file_name="research_report.txt",
                    mime="text/plain"
                )
            
            with tab2:
                st.markdown("### Blog Post")
                st.markdown(f'<div class="blog-output">{blog_content}</div>', 
                          unsafe_allow_html=True)
                
                # Export button for blog
                st.download_button(
                    label="Download Blog Post",
                    data=blog_content,
                    file_name="blog_post.txt",
                    mime="text/plain"
                )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your API keys and try again.")

# Footer
st.markdown("---")
st.markdown("Created with CrewAI & Streamlit.  By Shiv ❤️")