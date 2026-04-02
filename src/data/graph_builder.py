"""
Graph Builder Module for TG-GAT DDoS Detection System.

This module handles the construction of dynamic graphs from network traffic data,
including node and edge feature extraction, temporal windowing, and graph serialization.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx
import logging
from collections import defaultdict
import ipaddress
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Dynamic graph builder for network traffic analysis.
    
    Constructs time-based graphs from network flow data where:
    - Nodes represent IP addresses (source/destination)
    - Edges represent connections between IPs
    - Graphs are created in temporal windows (default: 100ms)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the graph builder.
        
        Args:
            config: Configuration dictionary containing graph parameters
        """
        self.config = config
        self.window_ms = config['data']['graph_window_ms']
        self.max_nodes = config['data']['max_nodes_per_graph']
        self.node_features = {}
        self.edge_features = {}
        
        # Feature extraction functions
        self.node_feature_functions = {
            'packet_count': self._extract_packet_count,
            'byte_count': self._extract_byte_count,
            'connection_count': self._extract_connection_count,
            'avg_packet_size': self._extract_avg_packet_size,
            'flow_rate': self._extract_flow_rate,
            'protocol_distribution': self._extract_protocol_distribution,
            'port_entropy': self._extract_port_entropy,
            'temporal_features': self._extract_temporal_features
        }
        
        self.edge_feature_functions = {
            'packet_count': self._extract_edge_packet_count,
            'byte_count': self._extract_edge_byte_count,
            'duration': self._extract_edge_duration,
            'protocol': self._extract_edge_protocol,
            'flags': self._extract_edge_flags,
            'avg_packet_size': self._extract_edge_avg_packet_size,
            'connection_rate': self._extract_edge_connection_rate
        }
    
    def build_dynamic_graphs(self, df: pd.DataFrame) -> List[Data]:
        """
        Build dynamic graphs from network traffic data.
        
        Args:
            df: Preprocessed DataFrame with network traffic data
            
        Returns:
            List of PyTorch Geometric Data objects representing temporal graphs
        """
        logger.info("Building dynamic graphs from network traffic data")
        
        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        # Create temporal windows
        graphs = []
        window_start = df['timestamp'].min() if 'timestamp' in df.columns else 0
        
        while True:
            # Define window bounds
            if isinstance(window_start, (datetime, pd.Timestamp)):
                window_end = window_start + timedelta(milliseconds=self.window_ms)
            else:
                window_end = window_start + self.window_ms
            
            # Filter data for current window
            if 'timestamp' in df.columns:
                window_data = df[(df['timestamp'] >= window_start) & (df['timestamp'] < window_end)]
            else:
                # If no timestamp, create windows based on row indices
                start_idx = int((window_start / self.window_ms) * len(df))
                end_idx = min(int((window_end / self.window_ms) * len(df)), len(df))
                window_data = df.iloc[start_idx:end_idx]
            
            if len(window_data) == 0:
                break
            
            # Build graph for this window
            graph_data = self._build_single_graph(window_data, window_start)
            
            if graph_data is not None:
                graphs.append(graph_data)
            
            # Move to next window
            window_start = window_end
            
            # Stop if we've processed all data
            if isinstance(window_start, (datetime, pd.Timestamp)):
                if window_start >= df['timestamp'].max():
                    break
            else:
                if window_start >= len(df) * self.window_ms:
                    break
        
        logger.info(f"Built {len(graphs)} dynamic graphs")
        return graphs
    
    def _build_single_graph(self, window_data: pd.DataFrame, window_timestamp: Union[datetime, float]) -> Optional[Data]:
        """
        Build a single graph from window data.
        
        Args:
            window_data: Data for current temporal window
            window_timestamp: Timestamp of the current window
            
        Returns:
            PyTorch Geometric Data object or None if no valid graph
        """
        if len(window_data) == 0:
            return None
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Extract unique IPs as nodes
        src_ips = window_data['src_ip'].unique() if 'src_ip' in window_data.columns else []
        dst_ips = window_data['dst_ip'].unique() if 'dst_ip' in window_data.columns else []
        all_ips = list(set(src_ips) | set(dst_ips))
        
        # Limit number of nodes
        if len(all_ips) > self.max_nodes:
            # Keep most active IPs
            ip_activity = window_data.groupby('src_ip').size() + window_data.groupby('dst_ip').size()
            top_ips = ip_activity.nlargest(self.max_nodes).index
            all_ips = list(set(all_ips) & set(top_ips))
        
        # Add nodes with features
        for ip in all_ips:
            node_features = self._extract_node_features(ip, window_data)
            G.add_node(ip, **node_features)
        
        # Add edges with features
        for _, row in window_data.iterrows():
            src_ip = row.get('src_ip')
            dst_ip = row.get('dst_ip')
            
            if src_ip in all_ips and dst_ip in all_ips:
                edge_features = self._extract_edge_features(src_ip, dst_ip, row)
                
                # Handle multiple edges between same nodes
                if G.has_edge(src_ip, dst_ip):
                    # Aggregate edge features
                    for key, value in edge_features.items():
                        if isinstance(value, (int, float)):
                            G.edges[src_ip, dst_ip][key] += value
                else:
                    G.add_edge(src_ip, dst_ip, **edge_features)
        
        # Convert to PyTorch Geometric
        try:
            pyg_data = from_networkx(G)
            
            # Add temporal information
            pyg_data.window_timestamp = window_timestamp
            pyg_data.num_nodes = len(G.nodes)
            pyg_data.num_edges = len(G.edges)
            
            return pyg_data
            
        except Exception as e:
            logger.error(f"Error converting graph to PyTorch Geometric: {str(e)}")
            return None
    
    def _extract_node_features(self, ip: str, window_data: pd.DataFrame) -> Dict[str, float]:
        """
        Extract features for a node (IP address).
        
        Args:
            ip: IP address
            window_data: Window data
            
        Returns:
            Dictionary of node features
        """
        features = {}
        
        # Get all flows involving this IP
        src_flows = window_data[window_data['src_ip'] == ip] if 'src_ip' in window_data.columns else pd.DataFrame()
        dst_flows = window_data[window_data['dst_ip'] == ip] if 'dst_ip' in window_data.columns else pd.DataFrame()
        all_flows = pd.concat([src_flows, dst_flows], ignore_index=True)
        
        # Extract various node features
        for feature_name, extract_func in self.node_features_functions.items():
            try:
                features[feature_name] = extract_func(ip, src_flows, dst_flows, all_flows)
            except Exception as e:
                logger.warning(f"Error extracting node feature {feature_name}: {str(e)}")
                features[feature_name] = 0.0
        
        return features
    
    def _extract_edge_features(self, src_ip: str, dst_ip: str, flow_data: pd.Series) -> Dict[str, float]:
        """
        Extract features for an edge (connection between two IPs).
        
        Args:
            src_ip: Source IP
            dst_ip: Destination IP
            flow_data: Flow data for this connection
            
        Returns:
            Dictionary of edge features
        """
        features = {}
        
        # Extract various edge features
        for feature_name, extract_func in self.edge_feature_functions.items():
            try:
                features[feature_name] = extract_func(src_ip, dst_ip, flow_data)
            except Exception as e:
                logger.warning(f"Error extracting edge feature {feature_name}: {str(e)}")
                features[feature_name] = 0.0
        
        return features
    
    # Node feature extraction functions
    def _extract_packet_count(self, ip: str, src_flows: pd.DataFrame, dst_flows: pd.DataFrame, all_flows: pd.DataFrame) -> float:
        """Extract packet count for node."""
        count = 0
        if 'Tot Fwd Pkts' in src_flows.columns:
            count += src_flows['Tot Fwd Pkts'].sum()
        if 'Tot Bwd Pkts' in dst_flows.columns:
            count += dst_flows['Tot Bwd Pkts'].sum()
        return float(count)
    
    def _extract_byte_count(self, ip: str, src_flows: pd.DataFrame, dst_flows: pd.DataFrame, all_flows: pd.DataFrame) -> float:
        """Extract byte count for node."""
        bytes_count = 0.0
        if 'TotLen Fwd Pkts' in src_flows.columns:
            bytes_count += src_flows['TotLen Fwd Pkts'].sum()
        if 'TotLen Bwd Pkts' in dst_flows.columns:
            bytes_count += dst_flows['TotLen Bwd Pkts'].sum()
        return float(bytes_count)
    
    def _extract_connection_count(self, ip: str, src_flows: pd.DataFrame, dst_flows: pd.DataFrame, all_flows: pd.DataFrame) -> float:
        """Extract connection count for node."""
        return float(len(all_flows))
    
    def _extract_avg_packet_size(self, ip: str, src_flows: pd.DataFrame, dst_flows: pd.DataFrame, all_flows: pd.DataFrame) -> float:
        """Extract average packet size for node."""
        total_packets = self._extract_packet_count(ip, src_flows, dst_flows, all_flows)
        total_bytes = self._extract_byte_count(ip, src_flows, dst_flows, all_flows)
        return total_bytes / (total_packets + 1e-8)
    
    def _extract_flow_rate(self, ip: str, src_flows: pd.DataFrame, dst_flows: pd.DataFrame, all_flows: pd.DataFrame) -> float:
        """Extract flow rate for node."""
        if 'flow_duration' in all_flows.columns:
            total_duration = all_flows['flow_duration'].sum()
            return len(all_flows) / (total_duration + 1e-8)
        return 0.0
    
    def _extract_protocol_distribution(self, ip: str, src_flows: pd.DataFrame, dst_flows: pd.DataFrame, all_flows: pd.DataFrame) -> float:
        """Extract protocol distribution entropy for node."""
        if 'protocol' in all_flows.columns:
            protocol_counts = all_flows['protocol'].value_counts()
            if len(protocol_counts) > 0:
                probabilities = protocol_counts / len(all_flows)
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
                return float(entropy)
        return 0.0
    
    def _extract_port_entropy(self, ip: str, src_flows: pd.DataFrame, dst_flows: pd.DataFrame, all_flows: pd.DataFrame) -> float:
        """Extract port entropy for node."""
        ports = []
        if 'src_port' in src_flows.columns:
            ports.extend(src_flows['src_port'].tolist())
        if 'dst_port' in dst_flows.columns:
            ports.extend(dst_flows['dst_port'].tolist())
        
        if ports:
            port_counts = pd.Series(ports).value_counts()
            probabilities = port_counts / len(ports)
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
            return float(entropy)
        return 0.0
    
    def _extract_temporal_features(self, ip: str, src_flows: pd.DataFrame, dst_flows: pd.DataFrame, all_flows: pd.DataFrame) -> float:
        """Extract temporal features for node."""
        if 'timestamp' in all_flows.columns:
            timestamps = pd.to_datetime(all_flows['timestamp'])
            if len(timestamps) > 1:
                time_diffs = timestamps.diff().dropna()
                avg_time_diff = time_diffs.mean().total_seconds()
                return float(avg_time_diff)
        return 0.0
    
    # Edge feature extraction functions
    def _extract_edge_packet_count(self, src_ip: str, dst_ip: str, flow_data: pd.Series) -> float:
        """Extract packet count for edge."""
        count = 0
        if 'Tot Fwd Pkts' in flow_data:
            count += flow_data['Tot Fwd Pkts']
        if 'Tot Bwd Pkts' in flow_data:
            count += flow_data['Tot Bwd Pkts']
        return float(count)
    
    def _extract_edge_byte_count(self, src_ip: str, dst_ip: str, flow_data: pd.Series) -> float:
        """Extract byte count for edge."""
        bytes_count = 0.0
        if 'TotLen Fwd Pkts' in flow_data:
            bytes_count += flow_data['TotLen Fwd Pkts']
        if 'TotLen Bwd Pkts' in flow_data:
            bytes_count += flow_data['TotLen Bwd Pkts']
        return float(bytes_count)
    
    def _extract_edge_duration(self, src_ip: str, dst_ip: str, flow_data: pd.Series) -> float:
        """Extract duration for edge."""
        if 'flow_duration' in flow_data:
            return float(flow_data['flow_duration'])
        return 0.0
    
    def _extract_edge_protocol(self, src_ip: str, dst_ip: str, flow_data: pd.Series) -> float:
        """Extract protocol for edge."""
        if 'protocol' in flow_data:
            return float(flow_data['protocol'])
        return 0.0
    
    def _extract_edge_flags(self, src_ip: str, dst_ip: str, flow_data: pd.Series) -> float:
        """Extract TCP flags for edge."""
        flags = 0
        flag_columns = ['Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags']
        for col in flag_columns:
            if col in flow_data:
                flags += flow_data[col]
        return float(flags)
    
    def _extract_edge_avg_packet_size(self, src_ip: str, dst_ip: str, flow_data: pd.Series) -> float:
        """Extract average packet size for edge."""
        packet_count = self._extract_edge_packet_count(src_ip, dst_ip, flow_data)
        byte_count = self._extract_edge_byte_count(src_ip, dst_ip, flow_data)
        return byte_count / (packet_count + 1e-8)
    
    def _extract_edge_connection_rate(self, src_ip: str, dst_ip: str, flow_data: pd.Series) -> float:
        """Extract connection rate for edge."""
        duration = self._extract_edge_duration(src_ip, dst_ip, flow_data)
        if duration > 0:
            return 1.0 / duration
        return 0.0
    
    def create_graph_batch(self, graphs: List[Data]) -> Batch:
        """
        Create a batch of graphs for training.
        
        Args:
            graphs: List of graph Data objects
            
        Returns:
            Batched Data object
        """
        if not graphs:
            raise ValueError("No graphs to batch")
        
        try:
            batch = Batch.from_data_list(graphs)
            logger.info(f"Created batch with {len(graphs)} graphs, {batch.num_nodes} total nodes")
            return batch
        except Exception as e:
            logger.error(f"Error creating graph batch: {str(e)}")
            raise
    
    def save_graphs(self, graphs: List[Data], filepath: str):
        """
        Save graphs to file.
        
        Args:
            graphs: List of graph Data objects
            filepath: Path to save the graphs
        """
        try:
            torch.save(graphs, filepath)
            logger.info(f"Saved {len(graphs)} graphs to {filepath}")
        except Exception as e:
            logger.error(f"Error saving graphs: {str(e)}")
            raise
    
    def load_graphs(self, filepath: str) -> List[Data]:
        """
        Load graphs from file.
        
        Args:
            filepath: Path to load the graphs from
            
        Returns:
            List of graph Data objects
        """
        try:
            graphs = torch.load(filepath)
            logger.info(f"Loaded {len(graphs)} graphs from {filepath}")
            return graphs
        except Exception as e:
            logger.error(f"Error loading graphs: {str(e)}")
            raise
