import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
from datetime import datetime, timedelta
import os
import json
import logging
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_predictor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StockDataset(Dataset):
    """PyTorch Dataset for stock data"""
    
    def __init__(self, stock_data, seq_length):
        self.seq_length = seq_length
        
        # Prepare the data features and target
        x_data = []
        y_data = []
        
        for i in range(len(stock_data) - seq_length):
            x_data.append(stock_data[i:i+seq_length])
            y_data.append(stock_data[i+seq_length, 0])  # Predict the Close price
        
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.len = len(x_data)
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

class LSTMModel(nn.Module):
    """LSTM Neural Network for stock price prediction"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class StockModelTrainer:
    """Class for training stock price prediction models"""
    
    def __init__(self, symbol, data_dir='data', models_dir='models', seq_length=30, split_ratio=0.8):
        self.symbol = symbol
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.seq_length = seq_length
        self.split_ratio = split_ratio
        
        # Ensure directories exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        # Load config
        self.config = self._load_config()
        
        # Check if data file exists
        self.data_file = os.path.join(data_dir, f"{symbol.replace('.', '_')}.csv")
        if not os.path.exists(self.data_file):
            logger.error(f"Data file for {symbol} not found at {self.data_file}")
            raise FileNotFoundError(f"Data file for {symbol} not found. Please download the data first.")
        
        # Load and preprocess data
        self.data = None
        self.scaler = None
        self.features = None
        self.train_loader = None
        self.val_loader = None
        
        # Model file path
        self.model_file = os.path.join(models_dir, f"{symbol.replace('.', '_')}_model.pth")
        
        # Load or create stats files
        self.stats_file = os.path.join(models_dir, f"{symbol.replace('.', '_')}_stats.json")
    
    def _load_config(self):
        """Load configuration from config.json file"""
        try:
            with open('config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Config file not found. Using default configuration.")
            return {
                "model": {
                    "seq_length": 30,
                    "epochs": 50,
                    "batch_size": 32,
                    "learning_rate": 0.001
                }
            }
    
    def _load_and_preprocess_data(self):
        """Load and preprocess stock data"""
        logger.info(f"Loading data for {self.symbol}")
        
        # Load data
        df = pd.read_csv(self.data_file)
        
        # Ensure data is sorted by date
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
        
        # Select features
        features = ['Close', 'Volume', 'Open', 'High', 'Low']
        available_features = [f for f in features if f in df.columns]
        
        if len(available_features) < 1:
            raise ValueError(f"No valid features found in data file. Columns: {df.columns}")
        
        # Extract features
        data = df[available_features].values
        
        # Scale data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = self.scaler.fit_transform(data)
        
        self.data = data_scaled
        self.features = available_features
        
        logger.info(f"Data loaded and preprocessed. Shape: {data_scaled.shape}")
    
    def _prepare_data_loaders(self):
        """Prepare training and validation data loaders"""
        # Load and preprocess data if not done yet
        if self.data is None:
            self._load_and_preprocess_data()
        
        # Split data into training and validation sets
        train_size = int(len(self.data) * self.split_ratio)
        train_data = self.data[:train_size]
        val_data = self.data[train_size:]
        
        # Create datasets
        train_dataset = StockDataset(train_data, self.seq_length)
        val_dataset = StockDataset(val_data, self.seq_length)
        
        # Create data loaders
        batch_size = self.config["model"]["batch_size"]
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        logger.info(f"Data loaders prepared. Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    def train_model(self, epochs=None, learning_rate=None):
        """Train the stock prediction model"""
        # Prepare data loaders if not done yet
        if self.train_loader is None:
            self._prepare_data_loaders()
        
        # Set hyperparameters
        if epochs is None:
            epochs = self.config["model"]["epochs"]
        if learning_rate is None:
            learning_rate = self.config["model"]["learning_rate"]
        
        input_size = self.data.shape[1]  # Number of features
        hidden_size = 64
        num_layers = 2
        output_size = 1
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        model = model.to(device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        logger.info(f"Starting training for {self.symbol}. Epochs: {epochs}")
        
        best_val_loss = float('inf')
        training_stats = {
            "train_losses": [],
            "val_losses": [],
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": self.config["model"]["batch_size"],
            "features": self.features,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            for X_batch, y_batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Forward pass
                y_pred = model(X_batch)
                loss = criterion(y_pred.squeeze(), y_batch)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(self.train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0
            y_true = []
            y_pred = []
            
            with torch.no_grad():
                for X_batch, y_batch in tqdm(self.val_loader, desc="Validation"):
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    
                    # Forward pass
                    y_pred_batch = model(X_batch)
                    loss = criterion(y_pred_batch.squeeze(), y_batch)
                    
                    val_loss += loss.item()
                    
                    # Store predictions for metrics
                    y_true.extend(y_batch.cpu().numpy())
                    y_pred.extend(y_pred_batch.squeeze().cpu().numpy())
            
            avg_val_loss = val_loss / len(self.val_loader)
            
            # Calculate metrics
            y_true_unscaled = self.scaler.inverse_transform(np.array([y_true, [0] * len(y_true), [0] * len(y_true), [0] * len(y_true), [0] * len(y_true)]).T)[:, 0]
            y_pred_unscaled = self.scaler.inverse_transform(np.array([y_pred, [0] * len(y_pred), [0] * len(y_pred), [0] * len(y_pred), [0] * len(y_pred)]).T)[:, 0]
            
            mae = mean_absolute_error(y_true_unscaled, y_pred_unscaled)
            rmse = np.sqrt(mean_squared_error(y_true_unscaled, y_pred_unscaled))
            mape = mean_absolute_percentage_error(y_true_unscaled, y_pred_unscaled) * 100
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), self.model_file)
                logger.info(f"Saved best model at epoch {epoch+1} with validation loss {avg_val_loss:.4f}")
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
            
            # Store stats
            training_stats["train_losses"].append(avg_train_loss)
            training_stats["val_losses"].append(avg_val_loss)
        
        # Calculate final metrics on validation set
        model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred_batch = model(X_batch)
                
                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(y_pred_batch.squeeze().cpu().numpy())
        
        y_true_unscaled = self.scaler.inverse_transform(np.array([y_true, [0] * len(y_true), [0] * len(y_true), [0] * len(y_true), [0] * len(y_true)]).T)[:, 0]
        y_pred_unscaled = self.scaler.inverse_transform(np.array([y_pred, [0] * len(y_pred), [0] * len(y_pred), [0] * len(y_pred), [0] * len(y_pred)]).T)[:, 0]
        
        mae = mean_absolute_error(y_true_unscaled, y_pred_unscaled)
        rmse = np.sqrt(mean_squared_error(y_true_unscaled, y_pred_unscaled))
        mape = mean_absolute_percentage_error(y_true_unscaled, y_pred_unscaled) * 100
        
        # Save final stats
        training_stats.update({
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "final_train_loss": avg_train_loss,
            "final_val_loss": avg_val_loss
        })
        
        with open(self.stats_file, 'w') as f:
            json.dump(training_stats, f, indent=4)
        
        logger.info(f"Training completed for {self.symbol}")
        logger.info(f"Final metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        
        return {
            "mae": mae,
            "rmse": rmse,
            "mape": mape
        }

class StockPredictor:
    """Class for predicting stock prices and analyzing the market"""
    
    def __init__(self, symbol=None, data_dir='data', models_dir='models', predictions_dir='predictions', reports_dir='reports', charts_dir='charts'):
        self.symbol = symbol
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.predictions_dir = predictions_dir
        self.reports_dir = reports_dir
        self.charts_dir = charts_dir
        
        # Ensure directories exist
        for directory in [data_dir, models_dir, predictions_dir, reports_dir, charts_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Load config
        self.config = self._load_config()
        
        # Load market information
        self.market_info = self._load_market_info()
    
    def _load_config(self):
        """Load configuration from config.json file"""
        try:
            with open('config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Config file not found. Creating default configuration.")
            default_config = {
                "market": {
                    "default_period": "5y",
                    "update_frequency_days": 7
                },
                "model": {
                    "seq_length": 30,
                    "epochs": 50,
                    "batch_size": 32,
                    "learning_rate": 0.001
                },
                "prediction": {
                    "forecast_days": 180,
                    "confidence_interval": 0.9
                }
            }
            
            with open('config.json', 'w') as f:
                json.dump(default_config, f, indent=4)
            
            return default_config
    
    def _load_market_info(self):
        """Load Saudi market information"""
        logger.info("Loading Saudi market information")
        
        # Path to market info file
        market_info_file = os.path.join(self.data_dir, 'saudi_market_info.csv')
        
        # Check if file exists and is recent (less than a week old)
        if os.path.exists(market_info_file) and (datetime.now() - datetime.fromtimestamp(os.path.getmtime(market_info_file))).days < 7:
            logger.info("Loading market info from existing file")
            try:
                market_info = pd.read_csv(market_info_file)
                return market_info
            except Exception as e:
                logger.error(f"Error loading market info: {str(e)}")
        
        # If file doesn't exist or is old, fetch market info
        logger.info("Fetching Saudi market symbols and information")
        
        # This is a simplified example - in a real application, you would fetch the actual Saudi market data
        # Tadawul (Saudi Stock Exchange) doesn't have a free public API, so you might need to use alternative sources
        
        # Sample Saudi market stocks (for demonstration purposes)
        sample_stocks = [
            {"Symbol": "2222.SR", "Name": "Aramco", "Sector": "Energy"},
            {"Symbol": "1150.SR", "Name": "Alinma Bank", "Sector": "Banks"},
            {"Symbol": "1010.SR", "Name": "RIBL", "Sector": "Banks"},
            {"Symbol": "2010.SR", "Name": "SABIC", "Sector": "Materials"},
            {"Symbol": "2350.SR", "Name": "Saudi Kayan", "Sector": "Materials"},
            {"Symbol": "7010.SR", "Name": "STC", "Sector": "Communication Services"},
            {"Symbol": "4190.SR", "Name": "Jarir", "Sector": "Consumer Discretionary"},
            {"Symbol": "2280.SR", "Name": "Almarai", "Sector": "Consumer Staples"},
            {"Symbol": "2002.SR", "Name": "Sabic Agri-Nutrients", "Sector": "Materials"},
            {"Symbol": "2380.SR", "Name": "Petro Rabigh", "Sector": "Energy"}
        ]
        
        market_info = pd.DataFrame(sample_stocks)
        
        # Save market info to file
        market_info.to_csv(market_info_file, index=False)
        
        logger.info(f"Market info saved to {market_info_file}")
        return market_info
    
    def download_historical_data(self, symbols=None, period=None, force_update=False):
        """Download historical stock data from Yahoo Finance"""
        if period is None:
            period = self.config["market"]["default_period"]
        
        if symbols is None:
            if self.symbol:
                symbols = [self.symbol]
            else:
                symbols = self.market_info["Symbol"].tolist()
        
        if isinstance(symbols, str):
            symbols = [symbols]
        
        logger.info(f"Downloading historical data for {len(symbols)} stocks")
        
        for symbol in tqdm(symbols, desc="Downloading stock data"):
            output_file = os.path.join(self.data_dir, f"{symbol.replace('.', '_')}.csv")
            
            # Skip if data exists and is recent, unless force_update is True
            if not force_update and os.path.exists(output_file) and (datetime.now() - datetime.fromtimestamp(os.path.getmtime(output_file))).days < self.config["market"]["update_frequency_days"]:
                logger.debug(f"Skipping {symbol} - recent data exists")
                continue
            
            try:
                # Download data from Yahoo Finance
                stock_data = yf.download(symbol, period=period)
                
                if len(stock_data) == 0:
                    logger.warning(f"No data available for {symbol}")
                    continue
                
                # Save to CSV
                stock_data.to_csv(output_file)
                logger.debug(f"Data for {symbol} saved to {output_file}")
            
            except Exception as e:
                logger.error(f"Error downloading data for {symbol}: {str(e)}")
        
        logger.info("Historical data download completed")
    
    def prepare_market_features(self, symbols=None):
        """Prepare additional features for market analysis"""
        if symbols is None:
            if self.symbol:
                symbols = [self.symbol]
            else:
                symbols = self.market_info["Symbol"].tolist()
        
        if isinstance(symbols, str):
            symbols = [symbols]
        
        logger.info(f"Preparing features for {len(symbols)} stocks")
        
        for symbol in tqdm(symbols, desc="Preparing features"):
            data_file = os.path.join(self.data_dir, f"{symbol.replace('.', '_')}.csv")
            
            if not os.path.exists(data_file):
                logger.warning(f"Data file for {symbol} not found at {data_file}")
                continue
            
            try:
                # Load data
                df = pd.read_csv(data_file)
                
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                
                # Calculate technical indicators
                # 1. Moving Averages
                df['MA_5'] = df['Close'].rolling(window=5).mean()
                df['MA_20'] = df['Close'].rolling(window=20).mean()
                df['MA_50'] = df['Close'].rolling(window=50).mean()
                
                # 2. Relative Strength Index (RSI)
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # 3. MACD (Moving Average Convergence Divergence)
                exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = exp1 - exp2
                df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                
                # 4. Bollinger Bands
                df['BB_Middle'] = df['Close'].rolling(window=20).mean()
                df['BB_Std'] = df['Close'].rolling(window=20).std()
                df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
                df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
                
                # 5. Price Rate of Change
                df['ROC'] = df['Close'].pct_change(periods=5) * 100
                
                # Save enhanced data
                df.to_csv(data_file, index=False)
                logger.debug(f"Enhanced features for {symbol} saved")
            
            except Exception as e:
                logger.error(f"Error preparing features for {symbol}: {str(e)}")
        
        logger.info("Feature preparation completed")
    
    def train_models_for_sector(self, sector, epochs=None):
        """Train models for all stocks in a specific sector"""
        # Get stocks in the sector
        sector_stocks = self.market_info[self.market_info["Sector"] == sector]["Symbol"].tolist()
        
        if not sector_stocks:
            logger.warning(f"No stocks found for sector '{sector}'")
            return []
        
        logger.info(f"Training models for {len(sector_stocks)} stocks in {sector} sector")
        
        results = []
        for symbol in tqdm(sector_stocks, desc=f"Training models for {sector} sector"):
            try:
                trainer = StockModelTrainer(symbol, self.data_dir, self.models_dir)
                training_results = trainer.train_model(epochs=epochs)
                
                results.append({
                    "Symbol": symbol,
                    "MAE": training_results["mae"],
                    "RMSE": training_results["rmse"],
                    "MAPE": training_results["mape"]
                })
                
                logger.info(f"Model for {symbol} trained successfully")
            
            except Exception as e:
                logger.error(f"Error training model for {symbol}: {str(e)}")
        
        # Save sector training results
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_file = os.path.join(self.reports_dir, f"{sector}_training_results.csv")
            results_df.to_csv(results_file, index=False)
            logger.info(f"Sector training results saved to {results_file}")
        
        return results
    
    def train_all_models(self, epochs=None):
        """Train models for all stocks in the market"""
        symbols = self.market_info["Symbol"].tolist()
        
        logger.info(f"Training models for all {len(symbols)} stocks in the market")
        
        results = []
        for symbol in tqdm(symbols, desc="Training all models"):
            try:
                trainer = StockModelTrainer(symbol, self.data_dir, self.models_dir)
                training_results = trainer.train_model(epochs=epochs)
                
                results.append({
                    "Symbol": symbol,
                    "MAE": training_results["mae"],
                    "RMSE": training_results["rmse"],
                    "MAPE": training_results["mape"]
                })
                
                logger.info(f"Model for {symbol} trained successfully")
            
            except Exception as e:
                logger.error(f"Error training model for {symbol}: {str(e)}")
        
        # Save overall training results
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_file = os.path.join(self.reports_dir, "market_training_results.csv")
            results_df.to_csv(results_file, index=False)
            logger.info(f"Market training results saved to {results_file}")
        
        return results
    
    def predict_future(self, days=None, symbol=None):
        """Predict future prices for a specific stock"""
        if symbol is None:
            symbol = self.symbol
        
        if symbol is None:
            raise ValueError("Symbol must be provided")
        
        if days is None:
            days = self.config["prediction"]["forecast_days"]
        
        logger.info(f"Predicting future prices for {symbol} for {days} days")
        
        # Paths to data and model files
        data_file = os.path.join(self.data_dir, f"{symbol.replace('.', '_')}.csv")
        model_file = os.path.join(self.models_dir, f"{symbol.replace('.', '_')}_model.pth")
        
        # Check if files exist
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file for {symbol} not found at {data_file}")
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file for {symbol} not found at {model_file}")
        
        try:
            # Load data
            df = pd.read_csv(data_file)
            
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date')
            
            # Select features
            features = ['Close', 'Volume', 'Open', 'High', 'Low']
            available_features = [f for f in features if f in df.columns]
            
            data = df[available_features].values
            
            # Scale data
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_scaled = scaler.fit_transform(data)
            
            # Prepare model input (last seq_length days)
            seq_length = self.config["model"]["seq_length"]
            input_data = data_scaled[-seq_length:].reshape(1, seq_length, len(available_features))
            input_tensor = torch.FloatTensor(input_data)
            
            # Load model
            input_size = len(available_features)
            hidden_size = 64
            num_layers = 2
            output_size = 1
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = LSTMModel(input_size, hidden_size, num_layers, output_size)
            model.load_state_dict(torch.load(model_file, map_location=device))
            model = model.to(device)
            model.eval()
            
            # Get current price
            current_price = df['Close'].iloc[-1]
            
            # Predict future prices
            future_prices = []
            dates = []
            last_date = df['Date'].iloc[-1] if 'Date' in df.columns else datetime.now()
            
            # First input is the last seq_length days
            current_input = input_tensor.to(device)
            
            for i in range(days):
                # Predict next day
                with torch.no_grad():
                    next_price = model(current_input).item()
                
                # Store the predicted price
                future_prices.append(next_price)
                
                # Calculate next date
                next_date = last_date + timedelta(days=i+1)
                dates.append(next_date)
                
                # Update input for next prediction
                new_input = np.zeros((1, 1, len(available_features)))
                new_input[0, 0, 0] = next_price  # Set predicted Close price
                
                # For other features, we could use the last known values or predictions
                # Here we're just using zeros for simplicity
                
                # Remove the first element and append the new input
                new_seq = np.concatenate([current_input.cpu().numpy()[0, 1:, :], new_input], axis=0)
                current_input = torch.FloatTensor(new_seq.reshape(1, seq_length, len(available_features))).to(device)
            
            # Unscale predictions
            zeros_array = np.zeros((len(future_prices), len(available_features)))
            zeros_array[:, 0] = future_prices
            unscaled_predictions = scaler.inverse_transform(zeros_array)[:, 0]
            
            # Create dataframe with predictions
            future_df = pd.DataFrame({
                'Date': dates,
                'Predicted_Close': unscaled_predictions
            })
            
            # Generate plot
            plt.figure(figsize=(12, 6))
            
            # Plot historical data
            plt.plot(df['Date'].iloc[-90:], df['Close'].iloc[-90:], label='Historical Prices')
            
            # Plot predictions
            plt.plot(future_df['Date'], future_df['Predicted_Close'], label='Predicted Prices', linestyle='--')
            
            # Add info
            predicted_price = unscaled_predictions[-1]
            growth = ((predicted_price / current_price) - 1) * 100
            plt.title(f"{symbol} - Price Prediction\nCurrent: ${current_price:.2f}, Predicted (in {days} days): ${predicted_price:.2f}, Growth: {growth:.2f}%")
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save chart
            chart_file = os.path.join(self.charts_dir, f"{symbol.replace('.', '_')}_future.png")
            plt.savefig(chart_file, dpi=100, bbox_inches='tight')
            plt.close()
            
            # Save predictions to CSV
            predictions_file = os.path.join(self.predictions_dir, f"{symbol.replace('.', '_')}_future.csv")
            future_df.to_csv(predictions_file, index=False)
            
            logger.info(f"Predictions for {symbol} saved to {predictions_file}")
            logger.info(f"Chart for {symbol} saved to {chart_file}")
            
            # Return prediction results
            return {
                'Symbol': symbol,
                'Current_Price': current_price,
                'Predicted_Price': predicted_price,
                'Growth_Pct': growth,
                'Prices': unscaled_predictions.tolist(),
                'Dates': [d.strftime('%Y-%m-%d') for d in dates],
                'DataFrame': future_df,
                'Chart_Path': chart_file
            }
        
        except Exception as e:
            logger.error(f"Error predicting future prices for {symbol}: {str(e)}")
            raise
    
    def predict_all_stocks(self, days=None):
        """Predict future prices for all stocks in the market"""
        if days is None:
            days = self.config["prediction"]["forecast_days"]
        
        logger.info(f"Predicting future prices for all stocks for {days} days")
        
        # Get list of trained models
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('_model.pth')]
        symbols = [f.replace('_model.pth', '').replace('_', '.') for f in model_files]
        
        predictions = []
        for symbol in tqdm(symbols, desc="Predicting prices"):
            try:
                # Get stock name and sector
                stock_info = self.market_info[self.market_info['Symbol'] == symbol]
                if stock_info.empty:
                    name = symbol
                    sector = "Unknown"
                else:
                    name = stock_info.iloc[0]['Name']
                    sector = stock_info.iloc[0]['Sector']
                
                # Predict future prices
                prediction = self.predict_future(days=days, symbol=symbol)
                
                # Store prediction summary
                predictions.append({
                    'Symbol': symbol,
                    'Name': name,
                    'Sector': sector,
                    'Current_Price': prediction['Current_Price'],
                    'Future_Price': prediction['Predicted_Price'],
                    'Growth_Pct': prediction['Growth_Pct']
                })
                
                logger.info(f"Prediction for {symbol} completed")
            
            except Exception as e:
                logger.error(f"Error predicting prices for {symbol}: {str(e)}")
        
        # Save predictions summary
        predictions_df = pd.DataFrame(predictions)
        if not predictions_df.empty:
            summary_file = os.path.join(self.predictions_dir, "market_predictions_summary.csv")
            predictions_df.to_csv(summary_file, index=False)
            logger.info(f"Market predictions summary saved to {summary_file}")
        
        return predictions
    
    def generate_market_report(self):
        """Generate a comprehensive market report"""
        logger.info("Generating market report")
        
        # Check if predictions exist
        summary_file = os.path.join(self.predictions_dir, "market_predictions_summary.csv")
        if not os.path.exists(summary_file):
            logger.warning("Predictions summary not found. Run predict_all_stocks first.")
            return None
        
        # Load predictions
        predictions = pd.read_csv(summary_file)
        
        # Generate report content
        report = []
        report.append("# Saudi Stock Market Prediction Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Overall market outlook
        avg_growth = predictions['Growth_Pct'].mean()
        positive_growth = (predictions['Growth_Pct'] > 0).sum()
        negative_growth = (predictions['Growth_Pct'] < 0).sum()
        
        report.append("## Market Outlook")
        report.append(f"- Average Growth Forecast: {avg_growth:.2f}%")
        report.append(f"- Stocks with Positive Growth: {positive_growth} ({positive_growth/len(predictions)*100:.1f}%)")
        report.append(f"- Stocks with Negative Growth: {negative_growth} ({negative_growth/len(predictions)*100:.1f}%)\n")
        
        # Top performing stocks
        report.append("## Top 10 Stocks by Predicted Growth")
        top_stocks = predictions.sort_values('Growth_Pct', ascending=False).head(10)
        
        for i, (_, stock) in enumerate(top_stocks.iterrows()):
            report.append(f"{i+1}. **{stock['Name']}** ({stock['Symbol']}) - Sector: {stock['Sector']}")
            report.append(f"   - Current Price: ${stock['Current_Price']:.2f}")
            report.append(f"   - Predicted Price: ${stock['Future_Price']:.2f}")
            report.append(f"   - Growth: {stock['Growth_Pct']:.2f}%\n")
        
        # Sector analysis
        report.append("## Sector Analysis")
        sector_analysis = predictions.groupby('Sector').agg({
            'Growth_Pct': ['mean', 'min', 'max', 'count']
        }).reset_index()
        
        sector_analysis.columns = ['Sector', 'Avg_Growth', 'Min_Growth', 'Max_Growth', 'Stocks_Count']
        sector_analysis = sector_analysis.sort_values('Avg_Growth', ascending=False)
        
        for _, sector in sector_analysis.iterrows():
            report.append(f"### {sector['Sector']}")
            report.append(f"- Number of Stocks: {sector['Stocks_Count']}")
            report.append(f"- Average Growth: {sector['Avg_Growth']:.2f}%")
            report.append(f"- Range: {sector['Min_Growth']:.2f}% to {sector['Max_Growth']:.2f}%")
            
            # Top performers in sector
            sector_top = predictions[predictions['Sector'] == sector['Sector']].sort_values('Growth_Pct', ascending=False).head(3)
            report.append("- Top Performers:")
            
            for _, stock in sector_top.iterrows():
                report.append(f"  - **{stock['Name']}** ({stock['Symbol']}): {stock['Growth_Pct']:.2f}%")
            
            report.append("")
        
        # Save report to file
        report_content = "\n".join(report)
        report_file = os.path.join(self.reports_dir, "market_report.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Market report saved to {report_file}")
        return report_content

# Command-line interface for the predictor
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Saudi Stock Market Predictor")
    parser.add_argument("--mode", choices=["single", "sector", "market", "interactive"], default="interactive",
                      help="Operation mode: single stock, sector, entire market, or interactive")
    parser.add_argument("--symbol", type=str, help="Stock symbol (for single mode)")
    parser.add_argument("--sector", type=str, help="Sector name (for sector mode)")
    parser.add_argument("--days", type=int, default=180, help="Number of days to predict")
    parser.add_argument("--force-update", action="store_true", help="Force update of historical data")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        print("Interactive mode not supported in this script. Please use the web interface.")
    else:
        predictor = StockPredictor()
        
        # Download data
        if args.mode == "single":
            if not args.symbol:
                parser.error("--symbol is required for single mode")
            predictor.download_historical_data(symbols=[args.symbol], force_update=args.force_update)
            predictor.prepare_market_features(symbols=[args.symbol])
        elif args.mode == "sector":
            if not args.sector:
                parser.error("--sector is required for sector mode")
            sector_stocks = predictor.market_info[predictor.market_info['Sector'] == args.sector]['Symbol'].tolist()
            if not sector_stocks:
                print(f"No stocks found for sector '{args.sector}'")
                sys.exit(1)
            predictor.download_historical_data(symbols=sector_stocks, force_update=args.force_update)
            predictor.prepare_market_features(symbols=sector_stocks)
        else:  # market mode
            predictor.download_historical_data(force_update=args.force_update)
            predictor.prepare_market_features()
        
        # Train models
        if args.mode == "single":
            try:
                print(f"Training model for {args.symbol}...")
                trainer = StockModelTrainer(args.symbol)
                results = trainer.train_model(epochs=args.epochs)
                print(f"Training completed - MAE: {results['mae']:.2f}, RMSE: {results['rmse']:.2f}, MAPE: {results['mape']:.2f}%")
            except Exception as e:
                print(f"Error training model: {str(e)}")
                sys.exit(1)
        elif args.mode == "sector":
            print(f"Training models for {args.sector} sector...")
            predictor.train_models_for_sector(args.sector, epochs=args.epochs)
        else:  # market mode
            print("Training models for entire market...")
            predictor.train_all_models(epochs=args.epochs)
        
        # Make predictions
        if args.mode == "single":
            try:
                print(f"Predicting future prices for {args.symbol}...")
                prediction = predictor.predict_future(days=args.days, symbol=args.symbol)
                print(f"Current price: ${prediction['Current_Price']:.2f}")
                print(f"Predicted price (in {args.days} days): ${prediction['Predicted_Price']:.2f}")
                print(f"Growth: {prediction['Growth_Pct']:.2f}%")
                print(f"Chart saved to {prediction['Chart_Path']}")
            except Exception as e:
                print(f"Error predicting prices: {str(e)}")
                sys.exit(1)
        elif args.mode == "sector":
            print(f"Predicting prices for {args.sector} sector...")
            predictor.predict_all_stocks(days=args.days)
        else:  # market mode
            print("Predicting prices for entire market...")
            predictor.predict_all_stocks(days=args.days)
        
        # Generate report (for sector and market modes)
        if args.mode in ["sector", "market"]:
            print("Generating market report...")
            predictor.generate_market_report()