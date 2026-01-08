import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from .preprocessing import preprocess_stock_interval, create_sequences_interval, scaler

class VolatilityDataset:
    """
    Handles data loading, preprocessing, and sequence generation for the Volatility Transformer.
    Aggregates data from multiple stock CSVs.
    """
    def __init__(self, config):
        self.config = config
        self.data_dir = config["data"]["data_dir"]
        self.tickers = config["data"].get("tickers", [])
        
        # Hyperparams
        self.train_frac = config["data"]["train_frac"]
        self.val_frac = config["data"]["val_frac"]
        self.interval_minutes = config["data"]["interval_minutes"]
        self.n_days = config["data"]["n_days"]
        
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        
        self.X_test_dic = {}
        self.y_test_dic = {}
        self.har_train = {}
        self.har_test = {}

    def load_and_prepare(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads all CSVs, preprocesses them, creates sequences, and aggregates them.
        Returns scaled train, valid arrays and stores test dictionaries internally.
        """
        # 1. Get file list
        if not self.tickers:
            # Load all csvs in data_dir
            search_path = os.path.join(self.data_dir, "*.csv")
            file_list = glob.glob(search_path)
        else:
            file_list = [os.path.join(self.data_dir, f"{t}.csv") for t in self.tickers]
            
        if not file_list:
            raise ValueError(f"No CSV files found in {self.data_dir}")

        print(f"Found {len(file_list)} files. Processing...")

        Xtr_list, ytr_list = [], []
        Xva_list, yva_list = [], []
        
        for path in file_list:
            stock_name = os.path.splitext(os.path.basename(path))[0]
            print(f"Processing {stock_name}...")
            
            try:
                df_raw = pd.read_csv(path)
                
                # Check required columns
                if "Date Time" not in df_raw.columns or "Close" not in df_raw.columns:
                     print(f"Skipping {stock_name}: Missing 'Date Time' or 'Close' columns.")
                     continue

                # Preprocess
                train_df, val_df, test_df = preprocess_stock_interval(
                    df_raw, 
                    self.train_frac, 
                    self.val_frac, 
                    self.interval_minutes
                )

                # Create Sequences
                # Note: create_sequences_interval returns X, y_residual, y_true_RV, har_feature
                X_tr, y_tr, train_RV, train_har = create_sequences_interval(train_df, self.n_days)
                X_va, y_va, val_RV, val_har = create_sequences_interval(val_df, self.n_days)
                X_te, y_te, test_RV, test_har = create_sequences_interval(test_df, self.n_days)
                
                # Append to lists
                if len(X_tr) > 0:
                    Xtr_list.append(X_tr)
                    ytr_list.append(y_tr)
                if len(X_va) > 0:
                    Xva_list.append(X_va)
                    yva_list.append(y_va)
                
                # Store test data per stock
                if len(X_te) > 0:
                    self.X_test_dic[stock_name] = X_te
                    self.y_test_dic[stock_name] = test_RV
                    self.har_test[stock_name] = test_har
                    # For training HAR we combine train and val HAR features?
                    # Notebook: har_train[stock_name] = np.concatenate((train_har, val_har))
                    self.har_train[stock_name] = np.concatenate((train_har, val_har)) if len(train_har) > 0 and len(val_har) > 0 else np.array([])
                    
            except Exception as e:
                print(f"Error processing {stock_name}: {e}")

        # Concatenate
        if not Xtr_list:
             raise RuntimeError("No training data generated. Check data files and parameters.")

        self.X_train = np.concatenate(Xtr_list, axis=0).astype(np.float32)
        self.y_train = np.concatenate(ytr_list, axis=0).astype(np.float32)
        
        self.X_val = np.concatenate(Xva_list, axis=0).astype(np.float32) if Xva_list else np.empty((0, self.X_train.shape[1], 1))
        self.y_val = np.concatenate(yva_list, axis=0).astype(np.float32) if yva_list else np.empty((0,))

        print(f"Total Train Samples: {len(self.X_train)}")
        print(f"Total Val Samples: {len(self.X_val)}")

        # Scale
        # only 1 feature currently: log_rv
        d_in = 1 
        self.X_train, self.X_val, self.X_test_dic = scaler(
            self.X_train, self.X_val, self.X_test_dic, d=d_in
        )
        
        return self.X_train, self.y_train, self.X_val, self.y_val

    def get_dataloaders(self, batch_size: int, num_workers: int = 2) -> tuple[DataLoader, DataLoader]:
        """Returns Train and Valid DataLoaders"""
        
        def mk_loader(X, y, bs, shuffle, workers):
            x_t = torch.tensor(X, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.float32)
            ds = TensorDataset(x_t, y_t)
            return DataLoader(
                ds, batch_size=bs, shuffle=shuffle,
                pin_memory=True, num_workers=workers,
                persistent_workers=(workers > 0)
            )

        train_loader = mk_loader(self.X_train, self.y_train, batch_size, True, num_workers)
        val_loader = mk_loader(self.X_val, self.y_val, batch_size, False, num_workers)
        
        return train_loader, val_loader
