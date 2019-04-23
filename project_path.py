import os

PROJECT_ROOT = os.path.dirname(__file__)

data_dir = os.path.join(PROJECT_ROOT,'data_process_to_fingerprint')
train_data_raw_path = os.path.join(data_dir, 'train_data_raw.json')
test_data_raw_path = os.path.join(data_dir, 'test_data_raw.json')
