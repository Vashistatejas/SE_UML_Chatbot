import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Set page config for a cleaner look
st.set_page_config(page_title="TIET Assistant Chatbot", page_icon="🤖", layout="centered")

# Custom CSS for a "beautiful" ChatGPT-like look
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Chat message container */
    .stChatMessage {
        background-color: #262730;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #41444c;
    }
    
    /* User message specific */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1a1c24;
    }
    
    /* Input container */
    .stChatInputContainer {
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    h1 {
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 2rem;
    }

    /* Student details container */
    .student-details {
        position: fixed;
        top: 3.5rem;
        right: 2rem;
        background-color: rgba(38, 39, 48, 0.8);
        padding: 10px 15px;
        border-radius: 8px;
        border: 1px solid #41444c;
        z-index: 1000;
        backdrop-filter: blur(5px);
        font-size: 0.9rem;
    }
    
    .student-details p {
        margin: 0;
        line-height: 1.4;
        color: #e0e0e0;
    }
    
    .student-details strong {
        color: #ffffff;
    }
    
    /* Add Button Styling */
    .add-details-btn {
        position: fixed;
        top: 3.5rem;
        right: 2rem;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

def display_student_details(name, roll_no, branch):
    st.markdown(f"""
    <div class="student-details">
        <p><strong>Name:</strong> {name}</p>
        <p><strong>Roll No:</strong> {roll_no}</p>
        <p><strong>Branch:</strong> {branch}</p>
    </div>
    """, unsafe_allow_html=True)

# Initialize session state for student details
if "student_details" not in st.session_state:
    st.session_state.student_details = None

# Function to handle form submission
def save_details():
    st.session_state.student_details = {
        "name": st.session_state.input_name,
        "roll_no": st.session_state.input_roll_no,
        "branch": st.session_state.input_branch
    }
    st.session_state.show_input_form = False

if "show_input_form" not in st.session_state:
    st.session_state.show_input_form = False

def toggle_form():
    st.session_state.show_input_form = not st.session_state.show_input_form

# Logic to display details or add button
if st.session_state.student_details:
    details = st.session_state.student_details
    display_student_details(details["name"], details["roll_no"], details["branch"])
    # Option to edit could be added here, but for now just display
else:
    # Display the + button
    # We use a container to position the button using Streamlit's layout or custom CSS
    # Since st.button can't be easily fixed-positioned with pure CSS without hacky ways to target it,
    # we'll put it in a placeholder and use CSS to move the wrapper if needed, 
    # OR just use a simple button at the top right if possible. 
    # Streamlit buttons are hard to style with custom CSS for position.
    # A cleaner way is to use a sidebar or just render it.
    # Let's try to put it in a column layout at the top.
    
    # Actually, to get the "top right" fixed position for a button, we might need a workaround or just place it normally.
    # But the user asked for a "plus icon which i can click".
    # I will use a standard st.button with a "+" label and use CSS to position its container if possible, 
    # but `st.button` returns a boolean.
    
    # Let's try a different approach: A floating action button using standard Streamlit components is hard.
    # I will place the button in a column at the top right.
    
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("➕ Add Details", key="add_details_btn"):
            toggle_form()

    if st.session_state.show_input_form:
        with st.form("details_form"):
            st.text_input("Name", key="input_name")
            st.text_input("Roll No", key="input_roll_no")
            st.text_input("Branch", key="input_branch")
            st.form_submit_button("Save", on_click=save_details)

st.title("TIET Assistant Chatbot")

# Initialize client
@st.cache_resource
def get_client():
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ.get("HUGGINGFACE_API_KEY"),
    )

client = get_client()

# Load system prompt
@st.cache_data
def get_system_prompt():
    try:
        with open("llm_main_prompt.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "You are a helpful assistant."

system_prompt = get_system_prompt()

# Initialize session state for history
if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    st.markdown("""
    <div style='display: flex; justify-content: center; align-items: center; height: 60vh; flex-direction: column;'>
        <h3 style='font-size: 1.5rem; color: #888;'>How can I help you?</h3>
    </div>
    """, unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare messages for API
    api_messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-6:]
    ]

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            completion = client.chat.completions.create(
                model="openai/gpt-oss-20b:groq",
                messages=api_messages,
                stream=True  # Enable streaming for better UX
            )
            
            full_response = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Add assistant response to state
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
