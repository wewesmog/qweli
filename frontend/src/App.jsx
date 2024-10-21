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
import ReactMarkdown from 'react-markdown';

function App() {
  const [input, setInput] = useState('');
  const [conversationHistory, setConversationHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null); // New state for session ID
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
    // Start a new session when the app loads
    startNewSession();
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
      .replace(new RegExp('\r?\n\n', 'g'), '<br>') // Replace double newlines with a break tag
      .replace(new RegExp('\r?\n', 'g'), '<br>');  // Replace single newlines with a break tag
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
      // Optionally start a new session for a new chat
      startNewSession();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
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

  useEffect(() => {
    console.log("Conversation history updated:", conversationHistory);
  }, [conversationHistory]);

  const handleRun = async () => {
    if (input !== "" && sessionId) {
      setLoading(true); // Set loading to true before API call
      const userMessage = { role: "user", parts: [{ text: input }] };
      setConversationHistory(prev => [...prev, userMessage]);
      let prompt = input;
      setInput("");

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
        console.log("Received response from backend:", data);

        if (data.qweli_response) {
          const formattedResponse = data.qweli_response.replace(/\\n/g, '\n');
          const botMessage = { role: "model", parts: [{ text: formattedResponse }] };
          setConversationHistory(prevHistory => [...prevHistory, botMessage]);
        } else {
          console.error("Unexpected response format:", data);
        }
      } catch (error) {
        console.error('Error:', error);
        const errorMessage = { role: "model", parts: [{ text: 'Sorry, there was an error processing your request.' }] };
        setConversationHistory(prevHistory => [...prevHistory, errorMessage]);
      } finally {
        setLoading(false); // Set loading to false after API call, whether it succeeded or failed
      }
    }
  };

  return (
    <div className="App">
      <div className="top">
        <span className="title">Marara</span>
        <img src={assets.user_icon} className="userIcon" alt="User Icon" />
      </div>
      <div className={conversationHistory.length === 0 ? "midContainer" : "chatContainer"}>
        {conversationHistory.length !== 0 ? (
          conversationHistory.map((convo, index) => (
            <div className="chat" key={index}>
              {convo.role === "user" ? <img src={assets.user_icon} className='chatImg' alt="" /> : <img src={assets.gemini_icon} className='chatImg gemImg' alt="" />}
              <span className={index % 2 !== 0 ? "chatText lighter" : "chatText"}>
                <ReactMarkdown>{convo.parts[0]?.text || ''}</ReactMarkdown>
              </span>
            </div>
          ))
        ) : (
          <>
            <div className="greet">
              <span className="colorGreet">Welcome to Marara</span>
              <span>Your Single Source of Truth.</span>
            </div>
            
          </>
        )}
        {loading && (
          <div className="chat">
            <img src={assets.gemini_icon} className='chatImg rotateImg' alt="" />
            <div className="loader">
              <hr />
              <hr />
              <hr />
            </div>
          </div>
        )}
      </div>
      <div className="Footer">
        <div className="sgTxt">
          <button className="sgBtn"
          onClick={() => setInput('Can you explain what VOOMA is and how it works?')}
          >
          <span>While Marara strives for accuracy, please always countercheck information provided. Marara may not always get it right.While Marara strives for accuracy, please always countercheck information provided. Marara may not always get it rightWhile Marara strives for accuracy, please always countercheck information provided. Marara may not always get it rightWhile Marara strives for accuracy, please always countercheck information provided. Marara may not always get it rightWhile Marara strives for accuracy, please always countercheck information provided. Marara may not always get it right</span>
          </button>
        </div>
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
            placeholder="Ask me any Vooma question (for Now)"
            className="searchBar"
          />
          <button className="srhImg">
            <img src={assets.gallery_icon} alt="Gallery Icon" />
          </button>
          <button className="srhImg">
            <img src={assets.mic_icon} alt="Mic Icon" />
          </button>
          {input !== '' ? (
            <button className="srhImg" onClick={handleRun}>
              <img src={assets.send_icon} alt="Send Icon" />
            </button>
          ) : (
            ''
          )}
        </div>
        <div className="info">
          While Marara strives for accuracy, please always countercheck information provided. Marara may not always get it right.
        </div>
      </div>
    </div>
  );
}

export default App;
