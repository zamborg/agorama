import streamlit as st
import asyncio
from datetime import datetime, timezone

# Import our ChatRoom and ChatMessage classes
from agorama.models import ChatMessage, ChatRoom
# Import our Agorama runner; note that we use YamlAgorama to load agent configs if provided
try:
    from agorama.agorama import YamlAgorama
except ImportError:
    YamlAgorama = None


# Set page configuration
st.set_page_config(page_title="Agorama Chat UI", layout="wide")

# Custom CSS for chat styling
st.markdown(
    """
    <style>
    .chat-message {
        padding: 10px;
        margin: 10px 0;
        border-radius: 10px;
        max-width: 80%;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #0f2499;
        align-self: flex-end;
    }
    .chat-message.agent {
        background-color: #2b0108;
        align-self: flex-start;
    }
    </style>
    """, unsafe_allow_html=True
)

# Initialize session state for chat room if not present
if 'chat_room' not in st.session_state:
    # Create a new ChatRoom instance with default room name
    st.session_state.chat_room = ChatRoom(room_name="Agorama")

# Optionally, load a YAML config to instantiate the Agorama app with agents
st.sidebar.subheader("Configuration")
config_file = st.sidebar.text_input("YAML Config File (optional):", value="")

if config_file and st.sidebar.button("Load Config"):
    if YamlAgorama is not None:
        try:
            agorama_app = YamlAgorama(config_file)
            st.session_state.chat_room = agorama_app.chat_room
            st.session_state.agorama_app = agorama_app
            st.success("Config loaded successfully")
        except Exception as e:
            st.error(f"Failed to load config: {e}")
    else:
        st.error("YamlAgorama not available.")

# Chat display section
st.title("Agorama Chat UI")

chat_container = st.container()
with chat_container:
    st.subheader("Chat")
    # Display each message with styling based on who sent it
    for message in st.session_state.chat_room.messages:
        css_class = "chat-message user" if message.created_by.lower() == 'user' else "chat-message agent"
        message_html = f"""
        <div class='{css_class}'>
            <strong>{message.created_by}</strong> <span style='font-size:0.8em; color:gray;'>({message.created_at.strftime('%Y-%m-%d %H:%M:%S')})</span><br>
            {message.message}
        </div>
        """
        st.markdown(message_html, unsafe_allow_html=True)

# Chat input section for user message
st.subheader("Send a Message")
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area("Your message", "")
    submit_button = st.form_submit_button(label="Send")

if submit_button and user_input:
    # Create a new ChatMessage from the user
    new_msg = ChatMessage(
        message=user_input,
        created_by="User",
        created_at=datetime.now(timezone.utc)
    )
    st.session_state.chat_room.add_message(new_msg)
    st.rerun()

# Optional: Button to tick agents and generate responses
st.sidebar.subheader("Agent Simulation")
if st.sidebar.button("Tick Agents"):
    if 'agorama_app' in st.session_state:
        try:
            # Run the tick method to let agents act and update the chat room
            asyncio.run(st.session_state.agorama_app.tick())
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error during tick: {e}")
    else:
        st.sidebar.error("No agent configuration loaded. Load a YAML config first.") 