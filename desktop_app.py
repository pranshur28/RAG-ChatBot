from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QTextEdit, QPushButton, QProgressBar, QFileDialog, QLabel, QScrollArea,
                            QFrame, QStackedWidget, QToolButton, QMenu)
from PyQt6.QtCore import Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve, QPoint
from PyQt6.QtGui import QIcon, QColor, QPalette, QLinearGradient, QFont, QFontDatabase
import sys
import os
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import numpy as np
import time

class OpenAIEmbeddingFunction:
    def __init__(self, api_key):
        self.api_key = api_key
        self.embeddings = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        try:
            self.embeddings = OpenAIEmbeddings(api_key=self.api_key)
        except Exception as e:
            print(f"Error initializing embeddings: {str(e)}")
            raise
    
    def __call__(self, input: list[str]) -> list[list[float]]:
        try:
            if not self.embeddings:
                self._initialize_embeddings()
            
            embeddings = []
            for text in input:
                if not isinstance(text, str):
                    print(f"Warning: Expected string input, got {type(text)}")
                    text = str(text)
                embedding = self.embeddings.embed_query(text)
                if isinstance(embedding, list):
                    embeddings.append(embedding)
                else:
                    print(f"Warning: Unexpected embedding type {type(embedding)}")
                    embeddings.append([0.0] * 1536)  # Default OpenAI embedding size
            return embeddings
        except Exception as e:
            print(f"Error in embedding generation: {str(e)}")
            raise

class TradingAssistantApp(QMainWindow):
    def __init__(self):
        super().__init__()
        print("Initializing Trading Assistant...")
        self.setWindowTitle("Trading Assistant")
        self.setMinimumSize(1000, 800)
        
        # Initialize class attributes
        self.chat_layout = None
        self.message_input = None
        self.progress_bar = None
        self.dark_mode = True
        
        # Create data directory if it doesn't exist
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Path to store book info
        self.book_info_path = os.path.join(self.data_dir, "book_info.txt")
        self.rules_info_path = os.path.join(self.data_dir, "rules_info.txt")
        
        # Load environment variables
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if not os.path.exists(env_path):
            with open(env_path, "w") as f:
                f.write("OPENAI_API_KEY=your-api-key-here")
            self.show_error_message("Please add your OpenAI API key to the .env file")
            return
            
        load_dotenv(env_path)
        if not os.getenv("OPENAI_API_KEY"):
            self.show_error_message("OpenAI API key not found. Please add it to the .env file")
            return
        
        # Load fonts
        print("Loading fonts...")
        QFontDatabase.addApplicationFont("fonts/Inter-Regular.ttf")
        QFontDatabase.addApplicationFont("fonts/Inter-Medium.ttf")
        QFontDatabase.addApplicationFont("fonts/Inter-SemiBold.ttf")
        
        # Setup UI first
        print("Setting up UI...")
        self.setup_ui()
        self.setup_animations()
        self.setup_shortcuts()
        
        # Initialize backend components in a separate thread
        print("Initializing backend...")
        QTimer.singleShot(0, self.initialize_backend)
    
    def initialize_backend(self):
        try:
            # Validate OpenAI API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or api_key == "your-api-key-here":
                self.show_error_message("Please add your OpenAI API key to the .env file")
                return
                
            print("Setting up ChromaDB...")
            # Initialize ChromaDB
            self.chromadb_client = chromadb.PersistentClient(path=os.path.join(self.data_dir, "chromadb"))
            
            # Initialize OpenAI
            print("Setting up OpenAI...")
            self.setup_openai()
            
            # Get or create collections
            embedding_function = OpenAIEmbeddingFunction(api_key)
            
            # Initialize rules collection
            try:
                self.rules_collection = self.chromadb_client.get_collection(
                    name="trading_rules",
                    embedding_function=embedding_function
                )
                print(f"Found existing rules collection with {self.rules_collection.count()} documents")
                if self.rules_collection.count() > 0:
                    self.show_info_message("✅ Trading rules loaded! You can start asking questions.")
                else:
                    self.load_rules_book()
            except Exception as e:
                print("Creating new rules collection...")
                self.rules_collection = self.chromadb_client.create_collection(
                    name="trading_rules",
                    embedding_function=embedding_function
                )
                self.load_rules_book()
            
            # Initialize data collection
            try:
                self.data_collection = self.chromadb_client.get_collection(
                    name="analysis_data",
                    embedding_function=embedding_function
                )
                print(f"Found existing data collection with {self.data_collection.count()} documents")
            except Exception as e:
                print("Creating new data collection...")
                self.data_collection = self.chromadb_client.create_collection(
                    name="analysis_data",
                    embedding_function=embedding_function
                )
            
            # Get or create collection
            try:
                # Try to get existing collection
                self.collection = self.chromadb_client.get_collection(
                    name="trading_docs",
                    embedding_function=embedding_function
                )
                print(f"Found existing collection with {self.collection.count()} documents")
                if self.collection.count() > 0:
                    self.show_info_message("✅ Previous book data loaded! You can start asking questions.")
                else:
                    # Collection exists but empty
                    self.load_textbook()
            except Exception as e:
                # Collection doesn't exist, create new one
                print("Creating new collection...")
                self.collection = self.chromadb_client.create_collection(
                    name="trading_docs",
                    embedding_function=embedding_function
                )
                self.load_textbook()
            
        except Exception as e:
            print(f"Error in initialize_backend: {str(e)}")
            self.show_error_message(f"Error initializing backend: {str(e)}")

    def setup_shortcuts(self):
        # Enter key to send message
        self.message_input.installEventFilter(self)
        
    def eventFilter(self, obj, event):
        if obj == self.message_input and event.type() == event.Type.KeyPress:
            if event.key() == Qt.Key.Key_Return and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.handle_send_message()
                return True
        return super().eventFilter(obj, event)

    def handle_attachment(self):
        try:
            file_dialog = QFileDialog()
            file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
            file_dialog.setNameFilter("Documents (*.pdf *.doc *.docx *.txt);;All Files (*.*)")
            
            if file_dialog.exec():
                filenames = file_dialog.selectedFiles()
                for file in filenames:
                    # Create a message showing the attached file
                    file_name = os.path.basename(file)
                    message = f"📎 Attached: {file_name}"
                    bubble = self.create_message_bubble("You", message)
                    self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
                    
                    # Analyze the file with RAG pipeline
                    if file.endswith(".pdf"):
                        with open(file, 'rb') as f:
                            reader = PdfReader(f)
                            text = ""
                            for page in reader.pages:
                                text += page.extract_text() + "\n\n"
                            self.analyze_data(text)
                    
                    # Acknowledge the upload
                    response = f"✅ Received file: {file_name}\nI'll analyze this document and use it to provide better trading insights."
                    assistant_bubble = self.create_message_bubble("Assistant", response)
                    self.chat_layout.insertWidget(self.chat_layout.count() - 1, assistant_bubble)
        except Exception as e:
            error_msg = f"Error uploading file: {str(e)}"
            self.show_error_message(error_msg)

    def handle_emoji(self):
        # TODO: Implement emoji picker
        self.show_info_message("Emoji picker coming soon! 😊")

    def handle_voice(self):
        # TODO: Implement voice input
        self.show_info_message("Voice input coming soon! 🎤")

    def handle_theme_toggle(self):
        self.dark_mode = not self.dark_mode
        # TODO: Implement theme switching
        theme = "dark" if self.dark_mode else "light"
        self.show_info_message(f"Switched to {theme} theme! 🎨")

    def handle_font_size(self, action):
        # TODO: Implement font size adjustment
        self.show_info_message(f"Font size adjustment coming soon! 📏")

    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Top bar with settings and theme toggle
        self.setup_top_bar(layout)
        
        # Chat area
        self.setup_chat_area(layout)
        
        # Input area
        self.setup_input_area(layout)

    def show_error_message(self, message):
        error_bubble = self.create_message_bubble("Assistant", f"❌ {message}")
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, error_bubble)

    def show_info_message(self, message):
        info_bubble = self.create_message_bubble("Assistant", f"ℹ️ {message}")
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, info_bubble)

    def setup_top_bar(self, layout):
        top_bar = QFrame()
        top_bar.setStyleSheet("""
            QFrame {
                background-color: #1a1b26;
                border-bottom: 1px solid #15161e;
            }
        """)
        top_bar_layout = QHBoxLayout(top_bar)
        
        # Create menu button
        menu_button = QToolButton()
        menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu_button.setIcon(QIcon("icons/menu.png"))
        menu_button.setIconSize(QSize(24, 24))
        menu_button.setStyleSheet("""
            QToolButton {
                border: none;
                padding: 5px;
                border-radius: 5px;
            }
            QToolButton:hover {
                background-color: #24283b;
            }
        """)
        
        # Create menu
        menu = QMenu(menu_button)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1a1b26;
                border: 1px solid #15161e;
                padding: 5px;
            }
            QMenu::item {
                padding: 5px 20px;
                border-radius: 3px;
                color: #a9b1d6;
            }
            QMenu::item:selected {
                background-color: #24283b;
            }
        """)
        
        # Add actions
        reload_action = menu.addAction("Load New Book")
        reload_action.triggered.connect(self.reload_book)
        
        theme_action = menu.addAction("Toggle Theme")
        theme_action.triggered.connect(self.handle_theme_toggle)
        
        menu_button.setMenu(menu)
        
        # Add to layout
        top_bar_layout.addWidget(menu_button)
        top_bar_layout.addStretch()
        
        layout.addWidget(top_bar)
        
    def reload_book(self):
        try:
            # Delete existing collection
            self.chromadb_client.delete_collection("trading_docs")
            
            # Delete book info file
            if os.path.exists(self.book_info_path):
                os.remove(self.book_info_path)
            
            # Create new collection
            api_key = os.getenv("OPENAI_API_KEY")
            embedding_function = OpenAIEmbeddingFunction(api_key)
            
            try:
                # Try to get existing collection
                self.collection = self.chromadb_client.get_collection(
                    name="trading_docs",
                    embedding_function=embedding_function
                )
                print(f"Found existing collection with {self.collection.count()} documents")
                if self.collection.count() > 0:
                    self.show_info_message("✅ Previous book data loaded! You can start asking questions.")
                else:
                    # Collection exists but empty
                    self.load_textbook()
            except Exception as e:
                # Collection doesn't exist, create new one
                print("Creating new collection...")
                self.collection = self.chromadb_client.create_collection(
                    name="trading_docs",
                    embedding_function=embedding_function
                )
                self.load_textbook()
            
        except Exception as e:
            print(f"Error reloading book: {str(e)}")
            self.show_error_message(f"Error reloading book: {str(e)}")

    def setup_chat_area(self, layout):
        # Chat container
        chat_container = QScrollArea()
        chat_container.setObjectName("chatContainer")
        chat_container.setWidgetResizable(True)
        chat_container.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Chat content widget
        chat_content = QWidget()
        self.chat_layout = QVBoxLayout(chat_content)
        self.chat_layout.setSpacing(16)
        self.chat_layout.setContentsMargins(16, 16, 16, 16)
        self.chat_layout.addStretch()
        
        chat_container.setWidget(chat_content)
        layout.addWidget(chat_container, 1)  # Give chat area stretch factor of 1
        
        # Style the chat area
        chat_container.setStyleSheet("""
            #chatContainer {
                background-color: #24283b;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #1a1b26;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #414868;
                min-height: 30px;
                border-radius: 6px;
                margin: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #565f89;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)
        
    def setup_input_area(self, layout):
        # Input container
        input_container = QWidget()
        input_container.setObjectName("inputContainer")
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(16, 16, 16, 16)
        input_layout.setSpacing(8)
        
        # Toolbar
        toolbar = QWidget()
        toolbar.setObjectName("toolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(8)
        
        # Toolbar buttons with handlers
        toolbar_buttons = {
            "emoji": (self.handle_emoji, "Insert emoji"),
            "attachment": (self.handle_attachment, "Attach files"),
            "voice": (self.handle_voice, "Voice input")
        }
        
        for icon, (handler, tooltip) in toolbar_buttons.items():
            btn = QToolButton()
            btn.setIcon(QIcon(f"icons/{icon}.svg"))
            btn.setIconSize(QSize(20, 20))
            btn.setObjectName("toolbarButton")
            btn.setToolTip(tooltip)
            btn.clicked.connect(handler)
            toolbar_layout.addWidget(btn)
        
        toolbar_layout.addStretch()
        input_layout.addWidget(toolbar)
        
        # Main input area
        input_area = QWidget()
        input_area.setObjectName("inputArea")
        input_area_layout = QHBoxLayout(input_area)
        input_area_layout.setContentsMargins(16, 0, 16, 0)
        input_area_layout.setSpacing(12)
        
        # Text input
        self.message_input = QTextEdit()
        self.message_input.setObjectName("messageInput")
        self.message_input.setPlaceholderText("Type a message... (Press Enter to send)")
        self.message_input.setFixedHeight(40)
        self.message_input.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Send button
        self.send_button = QPushButton()
        self.send_button.setIcon(QIcon("icons/send.svg"))
        self.send_button.setIconSize(QSize(20, 20))
        self.send_button.setObjectName("sendButton")
        self.send_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_button.setToolTip("Send message (Enter)")
        self.send_button.clicked.connect(self.handle_send_message)
        
        input_area_layout.addWidget(self.message_input)
        input_area_layout.addWidget(self.send_button)
        
        input_layout.addWidget(input_area)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("progressBar")
        self.progress_bar.setFixedHeight(2)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        
        input_layout.addWidget(self.progress_bar)
        layout.addWidget(input_container)
        
        # Style the input area
        input_container.setStyleSheet("""
            #inputContainer {
                background-color: #1a1b26;
                border-top: 1px solid #2f354d;
            }
            #toolbar {
                background: transparent;
            }
            #toolbarButton {
                background: transparent;
                border: none;
                border-radius: 6px;
                padding: 6px;
            }
            #toolbarButton:hover {
                background-color: #2f354d;
            }
            #inputArea {
                background-color: #24283b;
                border: 1px solid #414868;
                border-radius: 12px;
            }
            #inputArea:focus-within {
                border-color: #7aa2f7;
                background-color: #2f354d;
            }
            #messageInput {
                background: transparent;
                border: none;
                padding: 8px 0;
                font-family: 'Inter';
                font-size: 15px;
                color: #a9b1d6;
                line-height: 1.5;
            }
            #messageInput::placeholder {
                color: #565f89;
            }
            #sendButton {
                background-color: #7aa2f7;
                border: none;
                border-radius: 8px;
                padding: 8px;
                min-width: 40px;
                min-height: 40px;
            }
            #sendButton:hover {
                background-color: #89b4fa;
            }
            #sendButton:pressed {
                background-color: #6c91e0;
            }
            #progressBar {
                background-color: transparent;
                border: none;
            }
            #progressBar::chunk {
                background-color: #7aa2f7;
                border-radius: 1px;
            }
        """)
        
    def setup_animations(self):
        # Message fade-in animation
        self.message_animation = QPropertyAnimation(self, b"opacity")
        self.message_animation.setDuration(200)
        self.message_animation.setStartValue(0.0)
        self.message_animation.setEndValue(1.0)
        self.message_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
    def create_message_bubble(self, sender: str, message: str) -> QWidget:
        bubble = QFrame()
        bubble.setObjectName("messageBubble")
        
        layout = QHBoxLayout(bubble)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        # Avatar
        avatar = QLabel()
        avatar.setObjectName("avatar")
        avatar.setFixedSize(32, 32)
        avatar.setPixmap(QIcon(f"icons/{sender.lower()}_avatar.svg").pixmap(32, 32))
        
        # Message content
        content = QWidget()
        content.setObjectName("messageContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(12, 8, 12, 8)
        
        # Sender name
        name = QLabel(sender)
        name.setObjectName("senderName")
        
        # Message text
        text = QLabel(message)
        text.setObjectName("messageText")
        text.setWordWrap(True)
        
        content_layout.addWidget(name)
        content_layout.addWidget(text)
        
        if sender == "You":
            layout.addStretch()
            layout.addWidget(content)
            layout.addWidget(avatar)
            bubble.setStyleSheet("""
                #messageBubble {
                    background: transparent;
                }
                #messageContent {
                    background-color: #2f354d;
                    border-radius: 16px;
                    border-top-right-radius: 4px;
                }
                #senderName {
                    color: #7aa2f7;
                    font-weight: 600;
                    font-size: 13px;
                }
                #messageText {
                    color: #a9b1d6;
                    font-size: 15px;
                }
            """)
        else:
            layout.addWidget(avatar)
            layout.addWidget(content)
            layout.addStretch()
            bubble.setStyleSheet("""
                #messageBubble {
                    background: transparent;
                }
                #messageContent {
                    background-color: #1a1b26;
                    border-radius: 16px;
                    border-top-left-radius: 4px;
                }
                #senderName {
                    color: #9ece6a;
                    font-weight: 600;
                    font-size: 13px;
                }
                #messageText {
                    color: #a9b1d6;
                    font-size: 15px;
                }
            """)
        
        return bubble

    def setup_openai(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        try:
            self.chat_model = ChatOpenAI(
                api_key=api_key,
                model="gpt-4-turbo-preview",
                temperature=0.7
            )
            
            # Create the RAG chain
            template = """Answer the question based only on the following context:
            {context}
            
            Question: {question}
            """
            prompt = ChatPromptTemplate.from_template(template)
            
            self.chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | prompt
                | self.chat_model
                | StrOutputParser()
            )
            
        except Exception as e:
            print(f"Error setting up OpenAI: {str(e)}")
            raise

    def load_textbook(self):
        try:
            # Check if textbook is already processed
            if self.collection.count() > 0:
                print("Textbook already processed")
                self.show_info_message("✅ Book already loaded! You can start asking questions.")
                return
            
            # Check if book info file exists
            if os.path.exists(self.book_info_path):
                with open(self.book_info_path, "r") as f:
                    file_path = f.read()
                print(f"Loading textbook from saved path: {file_path}")
            else:
                # Open file dialog to select PDF
                file_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select Trading Book (PDF)",
                    "",
                    "PDF Files (*.pdf)"
                )
                
                if not file_path:
                    return
                
                # Save book info to file
                with open(self.book_info_path, "w") as f:
                    f.write(file_path)
            
            try:
                # Read PDF in chunks to reduce memory usage
                print("Reading PDF...")
                self.show_info_message("Reading PDF file...")
                
                text_chunks = []
                with open(file_path, 'rb') as file:
                    reader = PdfReader(file)
                    total_pages = len(reader.pages)
                    
                    # Process 5 pages at a time
                    for i in range(0, total_pages, 5):
                        page_text = ""
                        end_page = min(i + 5, total_pages)
                        
                        for j in range(i, end_page):
                            try:
                                page = reader.pages[j]
                                page_text += page.extract_text() + "\n\n"
                            except Exception as e:
                                print(f"Error reading page {j}: {str(e)}")
                                continue
                        
                        # Split into smaller chunks immediately
                        if page_text.strip():
                            splitter = RecursiveCharacterTextSplitter(
                                chunk_size=300,  # Smaller chunks
                                chunk_overlap=30,
                                length_function=len,
                            )
                            chunks = splitter.split_text(page_text)
                            text_chunks.extend(chunks)
                        
                        # Update progress and keep UI responsive
                        progress = int((i + 5) / total_pages * 30)
                        self.progress_bar.setValue(progress)
                        QApplication.processEvents()
                
                # Clear some memory
                del reader
                
                # Process chunks in smaller batches
                print(f"Processing {len(text_chunks)} chunks...")
                self.show_info_message(f"Processing {len(text_chunks)} text chunks...")
                
                batch_size = 5  # Smaller batch size
                for i in range(0, len(text_chunks), batch_size):
                    batch = text_chunks[i:i+batch_size]
                    
                    # Retry logic for adding to collection
                    max_retries = 3
                    for retry in range(max_retries):
                        try:
                            self.collection.add(
                                documents=batch,
                                ids=[f"chunk_{j}" for j in range(i, i+len(batch))]
                            )
                            break
                        except Exception as e:
                            if retry == max_retries - 1:
                                raise e
                            print(f"Retry {retry + 1} for batch {i}")
                            time.sleep(1)  # Wait before retry
                    
                    # Update progress (30-100%)
                    progress = 30 + int((i + len(batch)) / len(text_chunks) * 70)
                    self.progress_bar.setValue(progress)
                    QApplication.processEvents()
                
                # Clear memory
                del text_chunks
                
                print("Textbook processed successfully")
                self.show_info_message("✅ Book loaded successfully! You can start asking questions.")
                
            except Exception as e:
                print(f"Error processing PDF: {str(e)}")
                self.show_error_message(f"Error processing PDF: {str(e)}")
                if os.path.exists(self.book_info_path):
                    os.remove(self.book_info_path)
                
            finally:
                self.progress_bar.setValue(100)
                
        except Exception as e:
            print(f"Error in load_textbook: {str(e)}")
            self.show_error_message(f"Error: {str(e)}")

    def load_rules_book(self):
        try:
            # Check if rules book is already processed
            if self.rules_collection.count() > 0:
                print("Rules book already processed")
                self.show_info_message("✅ Rules book already loaded! You can start asking questions.")
                return
            
            # Check if rules book info file exists
            if os.path.exists(self.rules_info_path):
                with open(self.rules_info_path, "r") as f:
                    file_path = f.read()
                print(f"Loading rules book from saved path: {file_path}")
            else:
                # Open file dialog to select PDF
                file_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select Trading Rules Book (PDF)",
                    "",
                    "PDF Files (*.pdf)"
                )
                
                if not file_path:
                    return
                
                # Save rules book info to file
                with open(self.rules_info_path, "w") as f:
                    f.write(file_path)
            
            try:
                # Read PDF in chunks to reduce memory usage
                print("Reading PDF...")
                self.show_info_message("Reading PDF file...")
                
                text_chunks = []
                with open(file_path, 'rb') as file:
                    reader = PdfReader(file)
                    total_pages = len(reader.pages)
                    
                    # Process 5 pages at a time
                    for i in range(0, total_pages, 5):
                        page_text = ""
                        end_page = min(i + 5, total_pages)
                        
                        for j in range(i, end_page):
                            try:
                                page = reader.pages[j]
                                page_text += page.extract_text() + "\n\n"
                            except Exception as e:
                                print(f"Error reading page {j}: {str(e)}")
                                continue
                        
                        # Split into smaller chunks immediately
                        if page_text.strip():
                            splitter = RecursiveCharacterTextSplitter(
                                chunk_size=300,  # Smaller chunks
                                chunk_overlap=30,
                                length_function=len,
                            )
                            chunks = splitter.split_text(page_text)
                            text_chunks.extend(chunks)
                        
                        # Update progress and keep UI responsive
                        progress = int((i + 5) / total_pages * 30)
                        self.progress_bar.setValue(progress)
                        QApplication.processEvents()
                
                # Clear some memory
                del reader
                
                # Process chunks in smaller batches
                print(f"Processing {len(text_chunks)} chunks...")
                self.show_info_message(f"Processing {len(text_chunks)} text chunks...")
                
                batch_size = 5  # Smaller batch size
                for i in range(0, len(text_chunks), batch_size):
                    batch = text_chunks[i:i+batch_size]
                    
                    # Retry logic for adding to collection
                    max_retries = 3
                    for retry in range(max_retries):
                        try:
                            self.rules_collection.add(
                                documents=batch,
                                ids=[f"chunk_{j}" for j in range(i, i+len(batch))]
                            )
                            break
                        except Exception as e:
                            if retry == max_retries - 1:
                                raise e
                            print(f"Retry {retry + 1} for batch {i}")
                            time.sleep(1)  # Wait before retry
                    
                    # Update progress (30-100%)
                    progress = 30 + int((i + len(batch)) / len(text_chunks) * 70)
                    self.progress_bar.setValue(progress)
                    QApplication.processEvents()
                
                # Clear memory
                del text_chunks
                
                print("Rules book processed successfully")
                self.show_info_message("✅ Rules book loaded successfully! You can start asking questions.")
                
            except Exception as e:
                print(f"Error processing PDF: {str(e)}")
                self.show_error_message(f"Error processing PDF: {str(e)}")
                if os.path.exists(self.rules_info_path):
                    os.remove(self.rules_info_path)
                
            finally:
                self.progress_bar.setValue(100)
                
        except Exception as e:
            print(f"Error in load_rules_book: {str(e)}")
            self.show_error_message(f"Error: {str(e)}")

    def analyze_data(self, data_text):
        try:
            # Split data into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=30,
                length_function=len,
            )
            chunks = splitter.split_text(data_text)
            
            # Add to data collection
            self.data_collection.add(
                documents=chunks,
                ids=[f"data_chunk_{i}" for i in range(len(chunks))]
            )
            
            return True
        except Exception as e:
            print(f"Error analyzing data: {str(e)}")
            return False
    
    def query_with_rules(self, query):
        try:
            # Get relevant rules
            rules_results = self.rules_collection.query(
                query_texts=[query],
                n_results=2
            )
            
            # Get relevant data
            data_results = self.data_collection.query(
                query_texts=[query],
                n_results=2
            )
            
            # Combine contexts
            context = ""
            
            if rules_results and rules_results['documents']:
                context += "Trading Rules:\n" + "\n\n".join(rules_results['documents'][0]) + "\n\n"
            
            if data_results and data_results['documents']:
                context += "Analysis Data:\n" + "\n\n".join(data_results['documents'][0])
            
            if not context:
                return "I don't have enough context to answer that question. Please try asking something else."
            
            # Limit context length if too long
            if len(context) > 3000:
                context = context[:3000] + "..."
            
            return context
            
        except Exception as e:
            print(f"Error querying with rules: {str(e)}")
            return f"Error retrieving context: {str(e)}"

    def query_textbook(self, query):
        try:
            # Limit query to most relevant chunks
            results = self.collection.query(
                query_texts=[query],
                n_results=3  # Limit to top 3 most relevant chunks
            )
            
            if not results or not results['documents']:
                return "I don't have enough context to answer that question. Please try asking something else."
            
            # Combine the chunks with newlines for better context separation
            context = "\n\n".join(results['documents'][0])
            
            # Limit context length if too long
            if len(context) > 2000:
                context = context[:2000] + "..."
            
            return context
            
        except Exception as e:
            print(f"Error querying textbook: {str(e)}")
            return f"Error retrieving context: {str(e)}"

    def handle_send_message(self):
        try:
            message = self.message_input.toPlainText().strip()
            if not message:
                return
            
            # Create user message bubble
            user_bubble = self.create_message_bubble("User", message)
            self.chat_layout.insertWidget(self.chat_layout.count() - 1, user_bubble)
            
            # Clear input
            self.message_input.clear()
            
            # Show progress bar
            self.progress_bar.setRange(0, 0)  # Set to indeterminate mode
            
            try:
                # Get relevant context using both rules and data
                context = self.query_with_rules(message)
                
                # Get response from OpenAI using the chain
                response = self.chain.invoke({
                    "context": context,
                    "question": message
                })
                
                # Create assistant message bubble
                assistant_bubble = self.create_message_bubble("Assistant", response)
                self.chat_layout.insertWidget(self.chat_layout.count() - 1, assistant_bubble)
                
            except Exception as e:
                print(f"Error processing message: {str(e)}")
                self.show_error_message(f"Error: {str(e)}")
            
            finally:
                self.progress_bar.setRange(0, 100)  # Reset to determinate mode
                
        except Exception as e:
            print(f"Error in handle_send_message: {str(e)}")

def main():
    try:
        print("Starting application...")
        app = QApplication(sys.argv)
        print("QApplication created")
        
        print("Initializing main window...")
        window = TradingAssistantApp()
        print("Main window initialized")
        
        print("Showing main window...")
        window.show()
        print("Main window shown")
        
        print("Starting event loop...")
        sys.exit(app.exec())
    except Exception as e:
        print(f"Application error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
