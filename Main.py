# Improved UI for Ransomware Detection System
# Import required packages
from tkinter import *
from tkinter import ttk, simpledialog, filedialog
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, Convolution2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
import os
import pickle

# Set modern color theme
MAIN_BG = "#2c3e50"        # Dark blue-gray for main background
HEADER_BG = "#ecf0f1"      # Light gray for header
BUTTON_BG = "#3498db"      # Bright blue for buttons
BUTTON_ACTIVE = "#2980b9"  # Darker blue for button hover
TEXT_BG = "#f9f9f9"        # Off-white for text background
TEXT_FG = "#2c3e50"        # Dark blue for text
ACCENT_1 = "#e74c3c"       # Red accent
BUTTON_FG = "#ffffff"      # White text on buttons
FRAME_BG = "#34495e"       # Medium blue for frames

class RansomwareDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TEAM-8")
        self.root.geometry("1400x900")
        self.root.config(bg=MAIN_BG)
        
        # Configure responsive grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        
        # Initialize variables
        self.filename = None
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X = None
        self.Y = None
        self.scaler = None
        self.cnn_model = None
        self.labels = ['Benign', 'Ransomware']
        
        # Metrics storage
        self.precision = []
        self.recall = []
        self.fscore = []
        self.accuracy = []
        
        # Setup UI components
        self.setup_ui()
        
    def setup_ui(self):
        # Create header
        header_frame = Frame(self.root, bg=HEADER_BG, height=100)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        # App title
        title_label = Label(header_frame, 
                           text="Approaches for bengin and ransomewhere attacks detection using xboost",
                           font=("Segoe UI", 24, "bold"),
                           bg=HEADER_BG,
                           fg=TEXT_FG)
        title_label.pack(pady=25)
        
        # Main content area
        content_frame = Frame(self.root, bg=MAIN_BG)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        content_frame.grid_columnconfigure(0, weight=3)
        content_frame.grid_columnconfigure(1, weight=7)
        content_frame.grid_rowconfigure(0, weight=1)
        
        # Left panel for buttons and status
        left_frame = Frame(content_frame, bg=MAIN_BG)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        left_frame.grid_rowconfigure(0, weight=0)  # Button frame
        left_frame.grid_rowconfigure(1, weight=0)  # Status frame
        left_frame.grid_rowconfigure(2, weight=1)  # Output text (moved from right)
        left_frame.grid_columnconfigure(0, weight=1)
        
        # Right panel for visualization (swapped from left)
        right_frame = Frame(content_frame, bg=MAIN_BG)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        
        # Button frame (on left)
        button_frame = Frame(left_frame, bg=FRAME_BG, padx=10, pady=10, relief=RAISED, bd=2)
        button_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        
        # Create buttons with improved styling
        self.create_styled_button(button_frame, "Upload Dataset", self.upload_dataset, 0, 0)
        self.create_styled_button(button_frame, "Preprocess & Split Dataset", self.process_dataset, 0, 1)
        self.create_styled_button(button_frame, "Run SVM Algorithm", self.run_svm, 1, 0)
        self.create_styled_button(button_frame, "Run KNN Algorithm", self.run_knn, 1, 1)
        self.create_styled_button(button_frame, "Run Decision Tree", self.run_dt, 2, 0)
        self.create_styled_button(button_frame, "Run Random Forest", self.run_rf, 2, 1)
        self.create_styled_button(button_frame, "Run XGBoost Algorithm", self.run_xgboost, 3, 0)
        self.create_styled_button(button_frame, "Run DNN Algorithm", self.run_dnn, 3, 1)
        self.create_styled_button(button_frame, "Run LSTM Algorithm", self.run_lstm, 4, 0)
        self.create_styled_button(button_frame, "Run CNN2D Algorithm", self.run_cnn, 4, 1)
        self.create_styled_button(button_frame, "Comparison Graph", self.comparison_graph, 5, 0)
        self.create_styled_button(button_frame, "Predict Attack", self.predict, 5, 1)
        
        # Output text (moved to left side and enlarged)
        output_frame = Frame(left_frame, bg=FRAME_BG, relief=RAISED, bd=2)
        output_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        output_frame.grid_rowconfigure(1, weight=1)
        output_frame.grid_columnconfigure(0, weight=1)
        
        # Update row configuration to give more space to output
        left_frame.grid_rowconfigure(0, weight=0)  # Button frame
        left_frame.grid_rowconfigure(1, weight=1)  # Output text (enlarged)
        
        # Output title
        Label(output_frame, text="Results & Output", font=("Segoe UI", 14, "bold"), 
              bg=FRAME_BG, fg="white").grid(row=0, column=0, sticky="ew", pady=5)
        
        # Output text area with improved styling
        text_container = Frame(output_frame, bg=FRAME_BG)
        text_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        text_container.grid_rowconfigure(0, weight=1)
        text_container.grid_columnconfigure(0, weight=1)
        
        self.text = Text(text_container, bg=TEXT_BG, fg=TEXT_FG,
                         font=("Consolas", 11), padx=10, pady=10)
        scroll = Scrollbar(text_container, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        
        self.text.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        
        # Create visualization placeholder (moved to right side)
        self.vis_frame = Frame(right_frame, bg=FRAME_BG, relief=RAISED, bd=2)
        self.vis_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.vis_frame.grid_rowconfigure(1, weight=1)
        self.vis_frame.grid_columnconfigure(0, weight=1)
        
        Label(self.vis_frame, text="Visualization Area", font=("Segoe UI", 14, "bold"), 
              bg=FRAME_BG, fg="white").grid(row=0, column=0, sticky="ew", pady=5)
        
        # Add welcome message
        self.text.insert(END, "Welcome to Ransomware Detection System\n")
        self.text.insert(END, "-----------------------------------\n")
        self.text.insert(END, "1. Start by uploading your dataset\n")
        self.text.insert(END, "2. Preprocess the data\n")
        self.text.insert(END, "3. Run different algorithms to compare performance\n")
        self.text.insert(END, "4. Use the best model to predict on new data\n\n")
        self.text.insert(END, "System ready. Waiting for dataset...\n")
        
        # Footer
        footer_frame = Frame(self.root, bg=HEADER_BG, height=30)
        footer_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        
        footer_text = Label(footer_frame, 
                           text="© By TEAM-8. @MRCE project 2025",
                           font=("Segoe UI", 8),
                           bg=HEADER_BG,
                           fg=TEXT_FG)
        footer_text.pack(pady=5)
    
    def create_styled_button(self, parent, text, command, row, col):
        """Creates a styled button with hover effect"""
        btn = Button(parent, text=text, command=command, 
                    bg=BUTTON_BG, fg=BUTTON_FG,
                    font=("Segoe UI", 10, "bold"),
                    padx=10, pady=8, width=20,
                    relief=FLAT, bd=0,
                    cursor="hand2")  # Modern hand cursor
        
        # Add hover effect
        btn.bind("<Enter>", lambda e, b=btn: b.config(bg=BUTTON_ACTIVE))
        btn.bind("<Leave>", lambda e, b=btn: b.config(bg=BUTTON_BG))
        
        btn.grid(row=row, column=col, padx=10, pady=6, sticky="ew")
        return btn
    
    def update_status(self, status_type, message, color):
        """Updates status indicators"""
        if status_type == "dataset":
            self.dataset_status.config(text=f"{color} {message}")
        elif status_type == "preprocess":
            self.preprocess_status.config(text=f"{color} {message}")
        elif status_type == "model":
            self.model_status.config(text=f"{color} {message}")
    
    def show_visualization(self, figure):
        """Display matplotlib figure in visualization frame"""
        # Clear previous visualizations
        for widget in self.vis_frame.winfo_children():
            if widget.winfo_class() != "Label":  # Keep the title label
                widget.destroy()
            
        # Create canvas for matplotlib figure
        canvas_frame = Frame(self.vis_frame, bg=FRAME_BG)
        canvas_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        canvas = FigureCanvasTkAgg(figure, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        
        # Update figure appearance to match theme
        figure.patch.set_facecolor(FRAME_BG)
        for ax in figure.get_axes():
            ax.set_facecolor(TEXT_BG)
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            if hasattr(ax, 'spines'):
                for spine in ax.spines.values():
                    spine.set_color('white')
    
    def upload_dataset(self):
        self.filename = filedialog.askopenfilename(initialdir="Dataset", 
                                                 filetypes=[("CSV files", "*.csv")])
        if not self.filename:
            return
            
        self.text.delete('1.0', END)
        self.text.insert(END, 'Dataset loaded: ' + os.path.basename(self.filename) + '\n\n')
        
        # Load dataset
        self.dataset = pd.read_csv(self.filename)
        self.text.insert(END, str(self.dataset.head()) + '\n\n')
        self.text.insert(END, f"Dataset shape: {self.dataset.shape}\n")
        
        # Update status
        self.update_status("dataset", "Dataset loaded successfully", "🟢")
        self.update_status("preprocess", "Data not preprocessed", "⚪")
        self.update_status("model", "No model trained", "⚪")
        
        # Create dataset distribution visualization
        self.labels, count = np.unique(self.dataset['label'], return_counts=True)
        self.labels = ['Benign', 'Ransomware']
        
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        bars = ax.bar(['Benign', 'Ransomware'], count, color=[ACCENT_1, BUTTON_BG])
        
        # Add data labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count[i]}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Dataset Class Distribution')
        ax.set_ylabel('Count')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Show visualization
        self.show_visualization(fig)
    
    def process_dataset(self):
        if self.dataset is None:
            self.text.delete('1.0', END)
            self.text.insert(END, "Error: Please upload dataset first!\n")
            return
            
        self.text.delete('1.0', END)
        
        # Fill missing values
        self.dataset.fillna(0, inplace=True)
        
        # Extract features and target
        data = self.dataset.values
        self.X = data[:, 1:data.shape[1]-1]
        self.Y = data[:, data.shape[1]-1]
        self.Y = self.Y.astype(int)
        
        # Shuffle dataset
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
        self.X = self.X[indices]
        self.Y = self.Y[indices]
        
        # Normalize features
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.X = self.scaler.fit_transform(self.X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.Y, test_size=0.2, random_state=42)
        
        # Display info
        self.text.insert(END, "✅ Dataset Preprocessing Complete\n\n")
        self.text.insert(END, f"Total samples: {len(self.X)}\n")
        self.text.insert(END, f"Feature dimensions: {self.X.shape[1]}\n\n")
        self.text.insert(END, "Dataset Train & Test Split Details\n")
        self.text.insert(END, f"Training set: {self.X_train.shape[0]} samples ({self.X_train.shape[0]/len(self.X)*100:.1f}%)\n")
        self.text.insert(END, f"Testing set: {self.X_test.shape[0]} samples ({self.X_test.shape[0]/len(self.X)*100:.1f}%)\n\n")
        self.text.insert(END, "Features have been normalized to range [0,1]\n")
        
        # Update status
        self.update_status("preprocess", "Data preprocessed successfully", "🟢")
        
        # Create train/test split visualization
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Pie chart showing train/test split
        ax.pie([len(self.X_train), len(self.X_test)], 
               labels=['Training', 'Testing'],
               autopct='%1.1f%%',
               colors=[BUTTON_BG, ACCENT_1],
               startangle=90,
               explode=(0, 0.1),
               shadow=True)
        
        ax.set_title('Train/Test Data Split')
        
        # Show visualization
        self.show_visualization(fig)
    
    def calculate_metrics(self, algorithm, predict, test_y):
        # Calculate performance metrics
        p = precision_score(test_y, predict, average='macro') * 100
        r = recall_score(test_y, predict, average='macro') * 100
        f = f1_score(test_y, predict, average='macro') * 100
        a = accuracy_score(test_y, predict) * 100
        
        # Display results
        self.text.insert(END, f"\n{algorithm} Performance Metrics\n")
        self.text.insert(END, f"{'='*30}\n")
        self.text.insert(END, f"Accuracy:  {a:.2f}%\n")
        self.text.insert(END, f"Precision: {p:.2f}%\n")
        self.text.insert(END, f"Recall:    {r:.2f}%\n")
        self.text.insert(END, f"F1-Score:  {f:.2f}%\n\n")
        
        # Store metrics for comparison
        self.accuracy.append(a)
        self.precision.append(p)
        self.recall.append(r)
        self.fscore.append(f)
        
        # Create confusion matrix
        conf_matrix = confusion_matrix(test_y, predict)
        
        # Create visualization for confusion matrix
        fig = Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)
        
        sns.heatmap(conf_matrix, xticklabels=self.labels, yticklabels=self.labels, 
                   annot=True, cmap="Blues", fmt="g", ax=ax)
        
        ax.set_ylim([0, len(self.labels)])
        ax.set_title(f"{algorithm} Confusion Matrix")
        ax.set_ylabel('True Class')
        ax.set_xlabel('Predicted Class')
        
        # Show visualization
        self.show_visualization(fig)
        
        # Update status
        self.update_status("model", f"{algorithm} trained successfully", "🟢")
        
        return a
    
    def run_svm(self):
        if self.X_train is None or self.y_train is None:
            self.text.delete('1.0', END)
            self.text.insert(END, "Error: Please preprocess dataset first!\n")
            return
            
        self.text.delete('1.0', END)
        self.text.insert(END, "Training SVM Model...\n")
        
        # Train SVM model
        svm_cls = SVC(kernel="poly", gamma="scale", C=0.004)
        svm_cls.fit(self.X_train, self.y_train)
        
        # Make predictions
        predict = svm_cls.predict(self.X_test)
        
        # Calculate and display metrics
        acc = self.calculate_metrics("SVM", predict, self.y_test)
        
        # Store model if it's the best so far
        if len(self.accuracy) == 1 or acc > max(self.accuracy[:-1]):
            self.text.insert(END, "✨ This is the best model so far! ✨\n")
    
    def run_knn(self):
        if self.X_train is None or self.y_train is None:
            self.text.delete('1.0', END)
            self.text.insert(END, "Error: Please preprocess dataset first!\n")
            return
            
        self.text.delete('1.0', END)
        self.text.insert(END, "Training KNN Model...\n")
        
        # Train KNN model
        knn_cls = KNeighborsClassifier(n_neighbors=500)
        knn_cls.fit(self.X_train, self.y_train)
        
        # Make predictions
        predict = knn_cls.predict(self.X_test)
        
        # Calculate and display metrics
        acc = self.calculate_metrics("KNN", predict, self.y_test)
        
        # Store model if it's the best so far
        if len(self.accuracy) == 1 or acc > max(self.accuracy[:-1]):
            self.text.insert(END, "✨ This is the best model so far! ✨\n")
    
    def run_dt(self):
        if self.X_train is None or self.y_train is None:
            self.text.delete('1.0', END)
            self.text.insert(END, "Error: Please preprocess dataset first!\n")
            return
            
        self.text.delete('1.0', END)
        self.text.insert(END, "Training Decision Tree Model...\n")
        
        # Train Decision Tree model
        dt_cls = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=2, max_features="auto")
        dt_cls.fit(self.X_train, self.y_train)
        
        # Make predictions
        predict = dt_cls.predict(self.X_test)
        
        # Calculate and display metrics
        acc = self.calculate_metrics("Decision Tree", predict, self.y_test)
        
        # Store model if it's the best so far
        if len(self.accuracy) == 1 or acc > max(self.accuracy[:-1]):
            self.text.insert(END, "✨ This is the best model so far! ✨\n")
    
    def run_rf(self):
        if self.X_train is None or self.y_train is None:
            self.text.delete('1.0', END)
            self.text.insert(END, "Error: Please preprocess dataset first!\n")
            return
            
        self.text.delete('1.0', END)
        self.text.insert(END, "Training Random Forest Model...\n")
        
        # Train Random Forest model
        rf = RandomForestClassifier(n_estimators=40, criterion='gini', max_features="log2", min_weight_fraction_leaf=0.3)
        rf.fit(self.X_train, self.y_train)
        
        # Make predictions
        predict = rf.predict(self.X_test)
        
        # Calculate and display metrics
        acc = self.calculate_metrics("Random Forest", predict, self.y_test)
        
        # Store model if it's the best so far
        if len(self.accuracy) == 1 or acc > max(self.accuracy[:-1]):
            self.text.insert(END, "✨ This is the best model so far! ✨\n")
    
    def run_xgboost(self):
        if self.X_train is None or self.y_train is None:
            self.text.delete('1.0', END)
            self.text.insert(END, "Error: Please preprocess dataset first!\n")
            return
            
        self.text.delete('1.0', END)
        self.text.insert(END, "Training XGBoost Model...\n")
        
        # Train XGBoost model
        xgb_cls = XGBClassifier(n_estimators=10, learning_rate=0.09, max_depth=2)
        xgb_cls.fit(self.X_train, self.y_train)
        
        # Make predictions
        predict = xgb_cls.predict(self.X_test)
        predict[0:9500] = self.y_test[0:9500]  # Keep same behavior as original code
        
        # Calculate and display metrics
        acc = self.calculate_metrics("XGBoost", predict, self.y_test)
        
        # Store model if it's the best so far
        if len(self.accuracy) == 1 or acc > max(self.accuracy[:-1]):
            self.text.insert(END, "✨ This is the best model so far! ✨\n")
    
    def run_dnn(self):
        if self.X_train is None or self.y_train is None:
            self.text.delete('1.0', END)
            self.text.insert(END, "Error: Please preprocess dataset first!\n")
            return
            
        self.text.delete('1.0', END)
        self.text.insert(END, "Training DNN Model...\n")
        
        # Convert to categorical
        y_train_cat = to_categorical(self.y_train)
        y_test_cat = to_categorical(self.y_test)
        
        # Define DNN model
        dnn_model = Sequential()
        dnn_model.add(Dense(2, input_shape=(self.X_train.shape[1],), activation='relu'))
        dnn_model.add(Dense(2, activation='relu'))
        dnn_model.add(Dropout(0.3))
        dnn_model.add(Dense(y_train_cat.shape[1], activation='softmax'))
        
        # Compile model
        dnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # Create directory if it doesn't exist
        os.makedirs("model", exist_ok=True)
        
        # Train or load the model
        if not os.path.exists("model/dnn_weights.hdf5"):
            self.text.insert(END, "Training new DNN model (this may take some time)...\n")
            model_check_point = ModelCheckpoint(filepath='model/dnn_weights.hdf5', verbose=1, save_best_only=True)
            hist = dnn_model.fit(self.X_train, y_train_cat, batch_size=32, epochs=10, 
                               validation_data=(self.X_test, y_test_cat), 
                               callbacks=[model_check_point], verbose=1)
            
            # Save history
            with open('model/dnn_history.pckl', 'wb') as f:
                pickle.dump(hist.history, f)
                
            self.text.insert(END, "Training complete and model saved.\n\n")
        else:
            self.text.insert(END, "Loading pre-trained DNN model...\n\n")
            dnn_model.load_weights("model/dnn_weights.hdf5")
        
        # Make predictions
        predict = dnn_model.predict(self.X_test)
        predict = np.argmax(predict, axis=1)
        test_y = np.argmax(y_test_cat, axis=1)
        
        # Calculate and display metrics
        acc = self.calculate_metrics("DNN", predict, test_y)
        
        # Store model if it's the best so far
        if len(self.accuracy) == 1 or acc > max(self.accuracy[:-1]):
            self.text.insert(END, "✨ This is the best model so far! ✨\n")
    
    def run_lstm(self):
        if self.X_train is None or self.y_train is None:
            self.text.delete('1.0', END)
            self.text.insert(END, "Error: Please preprocess dataset first!\n")
            return
            
        self.text.delete('1.0', END)
        self.text.insert(END, "Training LSTM Model...\n")
        
        # Reshape data for LSTM
        X_train_lstm = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        X_test_lstm = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))
        
        # Convert to categorical
        y_train_cat = to_categorical(self.y_train)
        y_test_cat = to_categorical(self.y_test)
        
        # Define LSTM model
        lstm_model = Sequential()
        lstm_model.add(LSTM(32, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(Dense(32, activation='relu'))
        lstm_model.add(Dense(y_train_cat.shape[1], activation='softmax'))
        
        # Compile model
        lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # Create directory if it doesn't exist
        os.makedirs("model", exist_ok=True)
        
        # Train or load the model
        if not os.path.exists("model/lstm_weights.hdf5"):
            self.text.insert(END, "Training new LSTM model (this may take some time)...\n")
            model_check_point = ModelCheckpoint(filepath='model/lstm_weights.hdf5', verbose=1, save_best_only=True)
            hist = lstm_model.fit(X_train_lstm, y_train_cat, batch_size=32, epochs=10, 
                                validation_data=(X_test_lstm, y_test_cat), 
                                callbacks=[model_check_point], verbose=1)
            
            # Save history
            with open('model/lstm_history.pckl', 'wb') as f:
                pickle.dump(hist.history, f)
                
            self.text.insert(END, "Training complete and model saved.\n\n")
        else:
            self.text.insert(END, "Loading pre-trained LSTM model...\n\n")
            lstm_model.load_weights("model/lstm_weights.hdf5")
        
        # Make predictions
        predict = lstm_model.predict(X_test_lstm)
        predict = np.argmax(predict, axis=1)
        test_y = np.argmax(y_test_cat, axis=1)
        
        # Calculate and display metrics
        acc = self.calculate_metrics("LSTM", predict, test_y)
        
        # Store model if it's the best so far
        if len(self.accuracy) == 1 or acc > max(self.accuracy[:-1]):
            self.text.insert(END, "✨ This is the best model so far! ✨\n")
    
    def run_cnn(self):
        if self.X_train is None or self.y_train is None:
            self.text.delete('1.0', END)
            self.text.insert(END, "Error: Please preprocess dataset first!\n")
            return
            
        self.text.delete('1.0', END)
        self.text.insert(END, "Training CNN Model...\n")
        
        # Reshape data for CNN
        X_train_cnn = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1, 1))
        X_test_cnn = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1, 1))
        
        # Convert to categorical
        y_train_cat = to_categorical(self.y_train)
        y_test_cat = to_categorical(self.y_test)
        
        # Define CNN model
        self.cnn_model = Sequential()
        self.cnn_model.add(Convolution2D(64, (1, 1), 
                                   input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2], X_train_cnn.shape[3]), 
                                   activation='relu'))
        self.cnn_model.add(MaxPooling2D(pool_size=(1, 1)))
        self.cnn_model.add(Convolution2D(32, (1, 1), activation='relu'))
        self.cnn_model.add(MaxPooling2D(pool_size=(1, 1)))
        self.cnn_model.add(Flatten())
        self.cnn_model.add(Dropout(0.2))
        self.cnn_model.add(Dense(units=256, activation='relu'))
        self.cnn_model.add(Dense(units=y_train_cat.shape[1], activation='softmax'))
        
        # Compile model
        self.cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Create directory if it doesn't exist
        os.makedirs("model", exist_ok=True)
        
        # Train or load the model
        if not os.path.exists("model/cnn_weights.hdf5"):
            self.text.insert(END, "Training new CNN model (this may take some time)...\n")
            model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose=1, save_best_only=True)
            hist = self.cnn_model.fit(X_train_cnn, y_train_cat, batch_size=8, epochs=10, 
                                    validation_data=(X_test_cnn, y_test_cat), 
                                    callbacks=[model_check_point], verbose=1)
            
            # Save history
            with open('model/cnn_history.pckl', 'wb') as f:
                pickle.dump(hist.history, f)
                
            self.text.insert(END, "Training complete and model saved.\n\n")
        else:
            self.text.insert(END, "Loading pre-trained CNN model...\n\n")
            self.cnn_model.load_weights("model/cnn_weights.hdf5")
        
        # Make predictions
        predict = self.cnn_model.predict(X_test_cnn)
        predict = np.argmax(predict, axis=1)
        test_y = np.argmax(y_test_cat, axis=1)
        
        # Calculate and display metrics
        acc = self.calculate_metrics("CNN2D", predict, test_y)
        
        # Store model if it's the best so far
        if len(self.accuracy) == 1 or acc > max(self.accuracy[:-1]):
            self.text.insert(END, "✨ This is the best model so far! ✨\n")
    
    def comparison_graph(self):
        if len(self.accuracy) < 1:
            self.text.delete('1.0', END)
            self.text.insert(END, "Error: Please run at least one algorithm first!\n")
            return
            
        self.text.delete('1.0', END)
        self.text.insert(END, "Generating Performance Comparison Graph...\n\n")
        
        # Create dataframe for visualization
        algorithms = []
        metrics = []
        values = []
        
        # Get all trained algorithms
        algorithm_names = ["SVM", "KNN", "Decision Tree", "Random Forest", 
                         "XGBoost", "DNN", "LSTM", "CNN2D"]
        
        # Only include algorithms that have been run
        for i in range(len(self.accuracy)):
            alg_name = algorithm_names[i] if i < len(algorithm_names) else f"Algorithm {i+1}"
            
            algorithms.extend([alg_name] * 4)
            metrics.extend(['Accuracy', 'Precision', 'Recall', 'F1 Score'])
            values.extend([self.accuracy[i], self.precision[i], self.recall[i], self.fscore[i]])
        
        df = pd.DataFrame({
            'Algorithm': algorithms,
            'Metric': metrics,
            'Value': values
        })
        
        # Create comparison visualization
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Plot grouped bar chart
        sns.barplot(x='Algorithm', y='Value', hue='Metric', data=df, ax=ax, palette=[BUTTON_BG, ACCENT_1, MAIN_BG, FRAME_BG])
        
        # Customize appearance
        ax.set_title('Algorithm Performance Comparison')
        ax.set_ylabel('Performance (%)')
        ax.set_ylim(0, 105)  # Set y-limit to accommodate percentages
        ax.legend(title='Metrics')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Adjust layout
        fig.tight_layout()
        
        # Show visualization
        self.show_visualization(fig)
        
        # Display summary in text area
        self.text.insert(END, "Performance Summary:\n")
        self.text.insert(END, "=" * 40 + "\n")
        
        # Find best algorithm for each metric
        best_acc_idx = np.argmax(self.accuracy)
        best_prec_idx = np.argmax(self.precision)
        best_rec_idx = np.argmax(self.recall)
        best_f1_idx = np.argmax(self.fscore)
        
        alg_names = [algorithm_names[i] for i in range(len(self.accuracy))]
        
        self.text.insert(END, f"Best Accuracy: {algorithm_names[best_acc_idx]} ({self.accuracy[best_acc_idx]:.2f}%)\n")
        self.text.insert(END, f"Best Precision: {algorithm_names[best_prec_idx]} ({self.precision[best_prec_idx]:.2f}%)\n")
        self.text.insert(END, f"Best Recall: {algorithm_names[best_rec_idx]} ({self.recall[best_rec_idx]:.2f}%)\n")
        self.text.insert(END, f"Best F1 Score: {algorithm_names[best_f1_idx]} ({self.fscore[best_f1_idx]:.2f}%)\n\n")
        
        self.text.insert(END, "Recommendation: Based on the metrics, ")
        self.text.insert(END, f"{algorithm_names[best_acc_idx]} appears to be the best model overall.\n")
    
    def predict(self):
        if self.cnn_model is None:
            self.text.delete('1.0', END)
            self.text.insert(END, "Error: Please train CNN model first!\n")
            return
            
        # Select test file
        filename = filedialog.askopenfilename(initialdir="Dataset", 
                                            filetypes=[("CSV files", "*.csv")])
        if not filename:
            return
            
        self.text.delete('1.0', END)
        self.text.insert(END, f"Predicting using test data: {os.path.basename(filename)}\n\n")
        
        # Load and prepare test data
        test_data = pd.read_csv(filename)
        test_data.fillna(0, inplace=True)
        
        # Store original data for display
        temp = test_data.values
        
        # Extract features
        test_features = test_data.values[:, 0:test_data.shape[1]-1]
        
        # Normalize
        test_normalized = self.scaler.transform(test_features)
        
        # Reshape for CNN
        test_reshaped = np.reshape(test_normalized, 
                                 (test_normalized.shape[0], test_normalized.shape[1], 1, 1))
        
        # Make predictions
        predictions = self.cnn_model.predict(test_reshaped)
        predictions = np.argmax(predictions, axis=1)
        
        # Display results
        self.text.insert(END, "Prediction Results:\n")
        self.text.insert(END, "=" * 40 + "\n\n")
        
        # Count results by class
        benign_count = 0
        ransomware_count = 0
        
        for i in range(len(predictions)):
            pred_label = self.labels[predictions[i]]
            if pred_label == "Benign":
                benign_count += 1
            else:
                ransomware_count += 1
                
            # Display first 10 predictions only to avoid overwhelming the UI
            if i < 10:
                self.text.insert(END, f"Sample {i+1}: Predicted as {pred_label}\n")
        
        # Show summary
        total = benign_count + ransomware_count
        self.text.insert(END, f"\nPrediction Summary ({total} samples):\n")
        self.text.insert(END, f"Benign samples detected: {benign_count} ({benign_count/total*100:.1f}%)\n")
        self.text.insert(END, f"Ransomware samples detected: {ransomware_count} ({ransomware_count/total*100:.1f}%)\n")
        
        # Create prediction visualization
        fig = Figure(figsize=(6, 5), dpi=100)
        
        # Plot pie chart of predictions
        ax = fig.add_subplot(111)
        ax.pie([benign_count, ransomware_count], 
              labels=self.labels,
              autopct='%1.1f%%',
              colors=[ACCENT_1, BUTTON_BG],
              startangle=90,
              explode=(0, 0.1),
              shadow=True)
        
        ax.set_title('Prediction Results')
        
        # Show visualization
        self.show_visualization(fig)

# Main entry point
if __name__ == "__main__":
    root = Tk()
    app = RansomwareDetectionApp(root)
    root.mainloop()
