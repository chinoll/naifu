
from typing import Dict, Any
import random

class TextAugmentation:
    def __init__(self, config: Any):
        """
        Initialize the TextAugmentation class.

        Args:
            config: Can be one of:
                1. A dictionary with 'data_src' and 'config' keys.
                   Example: {'data_src': 'danbooru', 'config': {'field_0': {...}}}
                2. A list of such dictionaries for multiple sources.
                3. A dictionary mapping data source names to their configurations (backward compatibility).
                   Example: {'danbooru': {'field_0': {...}}}
        """
        self.config = config
        self.pipelines = {} # Changed to plural pipelines
        
        if isinstance(config, dict):
            if 'data_src' in config and 'config' in config:
                config_list = [config]
            else:
                # Treat as a map of data_src -> config
                config_list = [{'data_src': k, 'config': v} for k, v in config.items() if isinstance(v, dict)]
        elif isinstance(config, list):
            config_list = config
        else:
            config_list = []

        for item in config_list:
            if not isinstance(item, dict):
                continue
            data_src = item.get('data_src')
            data_src_config = item.get('config')
            
            if not data_src or not isinstance(data_src_config, dict):
                continue

            self.pipelines[data_src] = {}
            temp_pipeline = {}
            for key, value in data_src_config.items():
                if "_" in key and key.rsplit("_", 1)[-1].isdigit():
                    field_name, index = key.rsplit("_", 1)
                    if field_name not in temp_pipeline:
                        temp_pipeline[field_name] = []
                    temp_pipeline[field_name].append((int(index), value))
            
            for field_name in temp_pipeline:
                self.pipelines[data_src][field_name] = [x[1] for x in sorted(temp_pipeline[field_name], key=lambda x: x[0])]

    def execute_pipeline(self, data, pipeline):
        """
        Execute the augmentation pipeline on the data.

        Args:
            data: The input data to be augmented.
            pipeline: A list of augmentation configurations.

        Returns:
            The augmented data.
        """
        for config in pipeline:
            data = self._apply_augmentation(data, config)
        return data

    def _apply_augmentation(self, data, config):
        """
        Apply a single augmentation configuration to the data.

        Args:
            data: The input data.
            config: The augmentation configuration.

        Returns:
            The augmented data.
        """
        if isinstance(data, dict) and 'data' in data:
            target_data = data['data']
            count = data.get('count', 0)
        else:
            target_data = data
            count = 0

        method = config.get('method', '')
        if method == 'all':
            data_freq = config.get('data_freq')
            if data_freq is not None and isinstance(data_freq, int) and count > data_freq:
                print("yes?!",random.random(),data_freq / count)
                if random.random() > (data_freq / count):
                    print("yeah?")
                    return []

            prob = float(config.get('prob', 0.0))
            if prob > 0 and random.random() < prob:
                return []

        elif method == 'elem':
            prob = float(config.get('prob', 0.0))
            if prob > 0:
                target_data = [x for x in target_data if random.random() >= prob]
        
        return target_data

    def apply_to_item(self, data, data_src=None):
        """
        Apply augmentation to the input data based on data_src.
        
        Args:
            data: The input data to be augmented.
            data_src: The key identifying the source of the data to select the correct pipeline.
            
        Returns:
            The augmented data.
        """
        if not isinstance(data, dict):
            return data
            
        if data_src is None or data_src not in self.pipelines:
            return data

        pipeline = self.pipelines[data_src]
            
        for key, value in data.items():
            if key in pipeline:
                value = self.convert(value)
                data[key] = self.execute_pipeline(value, pipeline[key])
                
        return data

    def convert(self, data):
        """
        Convert input data to a list format if it's a string.
        
        Args:
            data: The input data (string, list, or dict).
            
        Returns:
            The data converted to a list or dict with list data.
        """
        if isinstance(data, str):
            return [data]
        return data
