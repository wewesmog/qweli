import { useState, useEffect, useRef } from 'react';
import './App.css';
import { assets } from './assets/assets.js';
import { useSelector, useDispatch } from 'react-redux';
import {
  addHistory,
  deleteHistory,
  localHistory,
  setChatIndex,
  toggleNewChat,
  updateHistory,
} from './features/promptSlice.js';

function App() {
  const [input, setInput] = useState('');
  const [conversationHistory, setConversationHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const bottomRef = useRef(null);

  const allHistory = useSelector((state) => state.prompt.historyArr);
  const newChat = useSelector((state) => state.prompt.newChat);
  const chatIndex = useSelector((state) => state.prompt.chatIndex);
  const dispatch = useDispatch();

  const handleAdd = (conv) => {
    dispatch(addHistory(conv));
  };

  const handleUpdate = (conv) => {
    dispatch(updateHistory(conv));
  };

  const handleDelete = (index) => {
    dispatch(deleteHistory(index));
  };

  const handleChat = () => {
    dispatch(toggleNewChat(false));
  };

  const handleChatIndex = (index) => {
    dispatch(setChatIndex(index));
  };

  const handleAllHistory = () => {
    let history = JSON.parse(localStorage.getItem('historyArr'));
    if (history) {
      dispatch(localHistory(history));
    }
  };

  useEffect(() => {
    handleAllHistory();
    startNewSession();
  }, []);

  const startNewSession = async () => {
    try {
      const response = await fetch('http://localhost:8000/start-session', {
        method: 'POST',
      });
      const data = await response.json();
      setSessionId(data.session_id);
      console.log('New session started with ID:', data.session_id);
    } catch (error) {
      console.error('Error starting session:', error);
    }
  };

  function jsonEscape(str) {
    str = str.split('*').join('<br>');
    return str
      .replace(new RegExp('\r?\n\n', 'g'), '<br>')
      .replace(new RegExp('\r?\n', 'g'), '<br>');
  }

  useEffect(() => {
    localStorage.setItem('historyArr', JSON.stringify(allHistory));
  }, [allHistory]);

  useEffect(() => {
    if (chatIndex >= 0) {
      setConversationHistory([...allHistory[chatIndex]]);
    }
  }, [chatIndex, allHistory]);

  useEffect(() => {
    if (newChat) {
      setConversationHistory([]);
      handleChat();
      handleChatIndex(-1);
      startNewSession();
    }
  }, [newChat]);

  useEffect(() => {
    if (conversationHistory.length === 1) {
      handleAdd(conversationHistory);
    }
    if (bottomRef.current) {
      setTimeout(() => {
        bottomRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
      }, 0);
    }
  }, [conversationHistory]);

  const handleRun = async () => {
    if (input !== '' && sessionId) {
      const userMessage = { role: 'user', parts: [{ text: input }] };
      setConversationHistory((prev) => [...prev, userMessage]);
      let prompt = input;
      setInput('');
      setLoading(true);

      try {
        const response = await fetch('http://localhost:8000/api/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            user_id: 'default_user',
            session_id: sessionId,
            user_input: prompt,
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        setLoading(false);

        const botMessage = { role: 'model', parts: [{ text: data.response }] };
        setConversationHistory((prevHistory) => [...prevHistory, botMessage]);

        if (conversationHistory.length > 1) {
          handleUpdate(conversationHistory);
        }
      } catch (error) {
        console.error('Error:', error);
        setLoading(false);
        const errorMessage = { role: 'model', parts: [{ text: 'Sorry, there was an error processing your request.' }] };
        setConversationHistory((prevHistory) => [...prevHistory, errorMessage]);
      }
    }
  };

  return (
    <div className="App">
      <div className="top">
        <span className="title">Marara</span>
        <img src={assets.user_icon} className="userIcon" alt="User Icon" />
      </div>
      <div className="mainContent">
        <div className="greet">
          <span className="colorGreet">Welcome to Marara</span>
          <span>Your Single Source of Truth.</span>
        </div>
        <div className="chatArea">
          <div className="chatContainer">
            {conversationHistory.length !== 0 ? (
              conversationHistory.map((convo, index) => (
                <div key={index} className="chat">
                  <span>{convo.parts[0].text}</span>
                </div>
              ))
            ) : (
              <div className="emptyChat">No conversation yet.</div>
            )}
          </div>
          <div className="Footer">
            <div className="Search">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    handleRun();
                  }
                }}
                placeholder="Ask me any question about SwiftCash"
                className="searchBar"
              />
              <button className="srhImg">
                <img src={assets.send_icon} alt="Send Icon" />
              </button>
            </div>
            <div className="info">
              While Marara strives for accuracy, please always countercheck information provided. Marara may not always get it right.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
