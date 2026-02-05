"""
Episodic Memory module for HippoRAG.

This module provides:
- EpisodicMemory: A dataclass representing chunk-specific episodic memory
- EpisodicMemoryStore: A store for managing episodic memories and their embeddings, similar to EmbeddingStore
"""

import json
import os
import logging
import textwrap
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple
from copy import deepcopy
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .embedding_model.base import BaseEmbeddingModel
from .llm import BaseLLM
from .prompts.prompt_template_manager import PromptTemplateManager
from .utils.embed_utils import retrieve_knn

logger = logging.getLogger(__name__)


@dataclass
class EpisodicMemory:
    memory_id: str  # 唯一标识符，格式：memory-{hash}
    chunk_ids: List[str]  # 该情境记忆包含的所有chunk_id（多对一关系）
    summary: str
    # events 是一个列表，每个元素是一个字典，包含一个事件的结构化要素
    # 每个字典的格式: {"participants": [...], "action": [...], "time": str|None, "location": str|None, "reason": str|None, "method": str|None}
    events: List[Dict[str, Any]] = field(default_factory=list)
    related_memory_ids: List[str] = field(default_factory=list)  # 相关的情境记忆ID列表
    timestamp: Optional[datetime] = None  # 情境记忆生成/更新时间戳

    def __post_init__(self):
        """
        如果 timestamp 未提供，自动设置为当前时间。
        """
        if self.timestamp is None:
            self.timestamp = datetime.now()
        
        # 向后兼容：如果提供了chunk_id但没有chunk_ids，转换为chunk_ids
        if hasattr(self, 'chunk_id') and not hasattr(self, 'chunk_ids'):
            self.chunk_ids = [self.chunk_id]
        
        # 向后兼容：如果提供了related_chunk_ids但没有related_memory_ids，转换为related_memory_ids
        if hasattr(self, 'related_chunk_ids') and not hasattr(self, 'related_memory_ids'):
            self.related_memory_ids = getattr(self, 'related_chunk_ids', [])

    def get_keywords(self) -> List[str]:
        """
        从 events 中提取关键词用于检索。
        从每个事件的结构化要素中提取关键词。
        """
        keywords = []
        for event in self.events:
            # 从每个事件的结构化要素中提取关键词
            if isinstance(event, dict):
                # participants 和 action 通常是列表
                participants_list = event.get("participants", [])
                if isinstance(participants_list, list):
                    keywords.extend([str(w) for w in participants_list if w is not None])
                elif participants_list is not None:
                    keywords.append(str(participants_list))
                
                action_list = event.get("action", [])
                if isinstance(action_list, list):
                    keywords.extend([str(w) for w in action_list if w is not None])
                elif action_list is not None:
                    keywords.append(str(action_list))
                
                # time, location, reason, method 通常是字符串
                for key in ["time", "location", "reason", "method"]:
                    value = event.get(key)
                    if value:
                        keywords.append(str(value))
        return list(set(keywords))  # 去重

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典，将 datetime 对象序列化为 ISO 格式字符串。
        """
        data = asdict(self)
        if self.timestamp is not None:
            data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodicMemory":
        """
        从字典创建对象，将 ISO 格式字符串转换为 datetime 对象。
        支持向后兼容：如果数据中有旧格式，会尝试转换。
        """
        # 向后兼容：如果数据中有chunk_id但没有chunk_ids，转换为chunk_ids
        if 'chunk_id' in data and 'chunk_ids' not in data:
            logger.warning("Found old format 'chunk_id', converting to 'chunk_ids' format")
            data['chunk_ids'] = [data['chunk_id']]
        
        # 向后兼容：如果没有memory_id，基于chunk_ids生成
        if 'memory_id' not in data:
            from .utils.misc_utils import compute_mdhash_id
            chunk_ids = data.get('chunk_ids', data.get('chunk_id', ['']))
            if isinstance(chunk_ids, str):
                chunk_ids = [chunk_ids]
            # 对chunk_ids排序后生成hash，确保相同组合生成相同ID
            sorted_chunk_ids = sorted(chunk_ids)
            data['memory_id'] = compute_mdhash_id(','.join(sorted_chunk_ids), prefix="memory-")
        
        # 向后兼容：如果数据中有related_chunk_ids但没有related_memory_ids，转换为related_memory_ids
        if 'related_chunk_ids' in data and 'related_memory_ids' not in data:
            logger.warning("Found old format 'related_chunk_ids', converting to 'related_memory_ids' format")
            # 注意：旧格式的related_chunk_ids是chunk_id列表，新格式的related_memory_ids是memory_id列表
            # 这里需要查找对应的memory_id，但为了简化，先直接使用chunk_ids（后续需要迁移）
            data['related_memory_ids'] = []  # 暂时为空，需要后续迁移
        
        # 向后兼容：如果数据中有 episodic_elements，尝试转换为 events
        if 'episodic_elements' in data and 'events' not in data:
            logger.warning("Found old format 'episodic_elements', converting to 'events' format")
            # 将旧的 episodic_elements 转换为一个事件字典
            old_elements = data['episodic_elements']
            if isinstance(old_elements, dict):
                data['events'] = [{
                    "participants": old_elements.get("who", []),
                    "action": old_elements.get("what", []),
                    "time": old_elements.get("when"),
                    "location": old_elements.get("where"),
                    "reason": old_elements.get("why"),
                    "method": old_elements.get("how")
                }]
            else:
                data['events'] = []
        
        # 向后兼容：如果 events 是字符串列表，转换为字典列表
        if 'events' in data and isinstance(data['events'], list) and len(data['events']) > 0:
            if isinstance(data['events'][0], str):
                logger.warning("Found old string format 'events', converting to structured format")
                # 将字符串事件转换为字典格式（无法提取结构化要素，只能放在action中）
                data['events'] = [{"action": [event], "participants": [], "time": None, "location": None, "reason": None, "method": None} 
                                 for event in data['events']]
        
        # 处理 timestamp
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            try:
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            except (ValueError, TypeError):
                logger.warning(f"Failed to parse timestamp: {data['timestamp']}, using current time")
                data['timestamp'] = None
        
        return cls(**data)


class EpisodicMemoryStore:
    def __init__(self, working_dir: str, embedding_model: Optional[BaseEmbeddingModel] = None,
                 store_name: str = "episodic_memories", embedding_batch_size: int = 32):
        self.working_dir = working_dir
        self.store_name = store_name
        self.store_path = os.path.join(working_dir, store_name)
        self.embedding_model = embedding_model
        self.embedding_batch_size = embedding_batch_size

        if not os.path.exists(self.store_path):
            logger.info(f"Creating episodic memory store directory: {self.store_path}")
            os.makedirs(self.store_path, exist_ok=True)

        self.memories_filename = os.path.join(self.store_path, "episodic_memories.json")
        self.chunk_to_memories_filename = os.path.join(self.store_path, "chunk_to_memories.json")
        self.embedding_filename = os.path.join(self.store_path, "episodic_memory_embeddings.parquet")
        
        # 向后兼容：保留旧的memory_filename引用
        self.memory_filename = self.memories_filename

        self._load_data()

    def _load_data(self):
        # 核心数据结构：多对一关系
        self.memory_id_to_memory: Dict[str, EpisodicMemory] = {}  # memory_id -> EpisodicMemory
        self.chunk_id_to_memory_ids: Dict[str, List[str]] = {}  # chunk_id -> List[memory_id]
        
        # Embedding相关（使用memory_id）
        self.memory_ids_emb: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.memory_id_to_idx: Dict[str, int] = {}
        
        # 向后兼容：保留旧的chunk_id_to_memory引用（用于兼容旧代码）
        self.chunk_id_to_memory: Dict[str, EpisodicMemory] = {}
        self.chunk_ids_emb: List[str] = []
        self.chunk_id_to_idx: Dict[str, int] = {}
        
        # 加载记忆数据
        if os.path.exists(self.memories_filename):
            try:
                with open(self.memories_filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 检查数据格式：新格式（memory_id为key）还是旧格式（chunk_id为key）
                if data and isinstance(data, dict):
                    first_key = list(data.keys())[0]
                    if first_key.startswith('memory-'):
                        # 新格式：memory_id -> EpisodicMemory
                        self.memory_id_to_memory = {
                            memory_id: EpisodicMemory.from_dict(mem_dict)
                            for memory_id, mem_dict in data.items()
                        }
                        # 构建chunk_id_to_memory_ids索引
                        # 确保一个chunk_id只对应一个memory_id（如果数据中有重复，保留最后一个）
                        for memory_id, memory in self.memory_id_to_memory.items():
                            for chunk_id in memory.chunk_ids:
                                # 如果chunk_id已经属于其他memory，记录警告并覆盖
                                if chunk_id in self.chunk_id_to_memory_ids:
                                    old_memory_id = self.chunk_id_to_memory_ids[chunk_id][0] if self.chunk_id_to_memory_ids[chunk_id] else None
                                    if old_memory_id and old_memory_id != memory_id:
                                        logger.warning(f"Chunk {chunk_id} belongs to multiple memories ({old_memory_id} and {memory_id}), keeping {memory_id}")
                                # 确保一个chunk_id只对应一个memory_id
                                self.chunk_id_to_memory_ids[chunk_id] = [memory_id]
                        # 向后兼容：为每个chunk_id创建映射（取第一个memory）
                        for chunk_id, memory_ids in self.chunk_id_to_memory_ids.items():
                            if memory_ids:
                                self.chunk_id_to_memory[chunk_id] = self.memory_id_to_memory[memory_ids[0]]
                        logger.info(f"Loaded {len(self.memory_id_to_memory)} episodic memories (new format) from {self.memories_filename}")
                    else:
                        # 旧格式：chunk_id -> EpisodicMemory（一对一）
                        logger.warning("Found old format episodic memories, converting to new format")
                        for chunk_id, mem_dict in data.items():
                            memory = EpisodicMemory.from_dict(mem_dict)
                            # 确保memory_id存在
                            if not hasattr(memory, 'memory_id') or not memory.memory_id:
                                from .utils.misc_utils import compute_mdhash_id
                                sorted_chunk_ids = sorted(memory.chunk_ids)
                                memory.memory_id = compute_mdhash_id(','.join(sorted_chunk_ids), prefix="memory-")
                            self.memory_id_to_memory[memory.memory_id] = memory
                            # 构建索引
                            # 确保一个chunk_id只对应一个memory_id
                            for cid in memory.chunk_ids:
                                # 如果chunk_id已经属于其他memory，记录警告并覆盖
                                if cid in self.chunk_id_to_memory_ids:
                                    old_memory_id = self.chunk_id_to_memory_ids[cid][0] if self.chunk_id_to_memory_ids[cid] else None
                                    if old_memory_id and old_memory_id != memory.memory_id:
                                        logger.warning(f"Chunk {cid} belongs to multiple memories ({old_memory_id} and {memory.memory_id}), keeping {memory.memory_id}")
                                # 确保一个chunk_id只对应一个memory_id
                                self.chunk_id_to_memory_ids[cid] = [memory.memory_id]
                            # 向后兼容
                            self.chunk_id_to_memory[chunk_id] = memory
                        logger.info(f"Converted {len(self.memory_id_to_memory)} episodic memories from old format")
            except Exception as e:
                logger.error(f"Error loading episodic memories: {e}")
                self.memory_id_to_memory = {}
                self.chunk_id_to_memory_ids = {}
                self.chunk_id_to_memory = {}
        else:
            logger.info(f"No existing episodic memory file found, starting fresh")

        # 加载chunk_to_memories索引（如果存在）
        if os.path.exists(self.chunk_to_memories_filename):
            try:
                with open(self.chunk_to_memories_filename, 'r', encoding='utf-8') as f:
                    chunk_to_memories_data = json.load(f)
                # 更新chunk_id_to_memory_ids
                # 确保一个chunk_id只对应一个memory_id（如果有多个，只保留第一个）
                for chunk_id, memory_ids in chunk_to_memories_data.items():
                    if memory_ids:
                        # 如果chunk_id已经属于其他memory，记录警告
                        if chunk_id in self.chunk_id_to_memory_ids:
                            old_memory_id = self.chunk_id_to_memory_ids[chunk_id][0] if self.chunk_id_to_memory_ids[chunk_id] else None
                            new_memory_id = memory_ids[0]
                            if old_memory_id and old_memory_id != new_memory_id:
                                logger.warning(f"Chunk {chunk_id} belongs to multiple memories ({old_memory_id} and {new_memory_id}), keeping {new_memory_id}")
                        # 确保一个chunk_id只对应一个memory_id（取第一个）
                        self.chunk_id_to_memory_ids[chunk_id] = [memory_ids[0]]
            except Exception as e:
                logger.warning(f"Error loading chunk_to_memories index: {e}")

        # 加载embedding数据
        if os.path.exists(self.embedding_filename):
            try:
                df = pd.read_parquet(self.embedding_filename)
                # 检查列名：新格式（memory_id）还是旧格式（chunk_id）
                if "memory_id" in df.columns:
                    # 新格式
                    self.memory_ids_emb = df["memory_id"].values.tolist()
                    self.embeddings = df["embedding"].values.tolist()
                    self.memory_id_to_idx = {
                        memory_id: idx for idx, memory_id in enumerate(self.memory_ids_emb)
                    }
                    logger.info(
                        f"Loaded {len(self.memory_ids_emb)} episodic memory embeddings (new format) "
                        f"from {self.embedding_filename}"
                    )
                elif "chunk_id" in df.columns:
                    # 旧格式：需要转换
                    logger.warning("Found old format embeddings, converting to new format")
                    self.chunk_ids_emb = df["chunk_id"].values.tolist()
                    embeddings_list = df["embedding"].values.tolist()
                    # 将chunk_id映射到memory_id
                    for chunk_id, embedding in zip(self.chunk_ids_emb, embeddings_list):
                        if chunk_id in self.chunk_id_to_memory_ids:
                            # 使用第一个memory_id
                            memory_id = self.chunk_id_to_memory_ids[chunk_id][0]
                            if memory_id not in self.memory_id_to_idx:
                                self.memory_ids_emb.append(memory_id)
                                self.embeddings.append(embedding)
                                self.memory_id_to_idx[memory_id] = len(self.memory_ids_emb) - 1
                    logger.info(f"Converted {len(self.memory_ids_emb)} embeddings from old format")
                else:
                    logger.warning("Unknown embedding format, skipping")
            except Exception as e:
                logger.error(f"Error loading episodic memory embeddings: {e}")
                self.memory_ids_emb = []
                self.embeddings = []
                self.memory_id_to_idx = {}
        else:
            self.memory_ids_emb = []
            self.embeddings = []
            self.memory_id_to_idx = {}

    def _save_memory_data(self):
        try:
            # 保存memory_id -> EpisodicMemory映射
            data_to_save = {
                memory_id: memory.to_dict()
                for memory_id, memory in self.memory_id_to_memory.items()
            }
            temp_filename = self.memories_filename + ".tmp"
            with open(temp_filename, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            if os.path.exists(self.memories_filename):
                os.remove(self.memories_filename)
            os.rename(temp_filename, self.memories_filename)
            
            # 保存chunk_id -> memory_ids映射
            chunk_to_memories_data = {
                chunk_id: memory_ids
                for chunk_id, memory_ids in self.chunk_id_to_memory_ids.items()
            }
            temp_chunk_filename = self.chunk_to_memories_filename + ".tmp"
            with open(temp_chunk_filename, 'w', encoding='utf-8') as f:
                json.dump(chunk_to_memories_data, f, ensure_ascii=False, indent=2)
            if os.path.exists(self.chunk_to_memories_filename):
                os.remove(self.chunk_to_memories_filename)
            os.rename(temp_chunk_filename, self.chunk_to_memories_filename)
            
            logger.info(f"Saved {len(self.memory_id_to_memory)} episodic memories to {self.memories_filename}")
        except Exception as e:
            logger.error(f"Error saving episodic memories: {e}")
            raise

    def _save_embedding_data(self):
        try:
            if not self.memory_ids_emb:
                return
            data_to_save = pd.DataFrame({
                "memory_id": self.memory_ids_emb,
                "embedding": self.embeddings
            })
            data_to_save.to_parquet(self.embedding_filename, index=False)
            self.memory_id_to_idx = {memory_id: idx for idx, memory_id in enumerate(self.memory_ids_emb)}
            logger.info(f"Saved {len(self.memory_ids_emb)} episodic memory embeddings to {self.embedding_filename}")
        except Exception as e:
            logger.error(f"Error saving episodic memory embeddings: {e}")
            raise

    def get(self, chunk_id: str) -> Optional[EpisodicMemory]:
        """
        通过chunk_id获取记忆（向后兼容方法）。
        如果chunk_id对应多个memory，返回第一个。
        """
        if chunk_id in self.chunk_id_to_memory_ids:
            memory_ids = self.chunk_id_to_memory_ids[chunk_id]
            if memory_ids:
                return self.memory_id_to_memory.get(memory_ids[0])
        # 向后兼容
        return self.chunk_id_to_memory.get(chunk_id)
    
    def get_by_memory_id(self, memory_id: str) -> Optional[EpisodicMemory]:
        """通过memory_id获取记忆"""
        return self.memory_id_to_memory.get(memory_id)
    
    def get_memories_by_chunk_ids(self, chunk_ids: List[str]) -> Dict[str, List[EpisodicMemory]]:
        """
        通过chunk_id列表检索对应的情境记忆（多对一关系）。
        
        Args:
            chunk_ids: chunk_id列表
        
        Returns:
            Dict[chunk_id, List[EpisodicMemory]]: 每个chunk_id对应的情境记忆列表
        """
        result = {}
        for chunk_id in chunk_ids:
            memory_ids = self.chunk_id_to_memory_ids.get(chunk_id, [])
            memories = [
                self.memory_id_to_memory[mid] 
                for mid in memory_ids 
                if mid in self.memory_id_to_memory
            ]
            result[chunk_id] = memories
        return result

    def batch_get(self, chunk_ids: List[str]) -> Dict[str, EpisodicMemory]:
        """
        批量获取记忆（向后兼容方法）。
        如果chunk_id对应多个memory，返回第一个。
        """
        result = {}
        for chunk_id in chunk_ids:
            memory = self.get(chunk_id)
            if memory:
                result[chunk_id] = memory
        return result

    def save(self, memories: Dict[str, EpisodicMemory]):
        """
        保存记忆（支持两种格式：memory_id或chunk_id作为key）。
        如果key是memory_id，直接保存；如果key是chunk_id，需要从memory中获取memory_id。
        """
        if not memories:
            return
        
        for key, memory in memories.items():
            # 确保memory_id存在
            if not hasattr(memory, 'memory_id') or not memory.memory_id:
                from .utils.misc_utils import compute_mdhash_id
                sorted_chunk_ids = sorted(memory.chunk_ids)
                memory.memory_id = compute_mdhash_id(','.join(sorted_chunk_ids), prefix="memory-")
            
            # 保存到memory_id_to_memory
            self.memory_id_to_memory[memory.memory_id] = memory
            
            # 更新chunk_id_to_memory_ids索引
            # 确保一个chunk_id只属于一个memory：如果chunk_id已经属于其他memory，需要先移除
            for chunk_id in memory.chunk_ids:
                # 如果chunk_id已经属于其他memory，先移除旧的关联
                if chunk_id in self.chunk_id_to_memory_ids:
                    existing_memory_ids = self.chunk_id_to_memory_ids[chunk_id]
                    # 移除不是当前memory的ID
                    for existing_memory_id in existing_memory_ids:
                        if existing_memory_id != memory.memory_id:
                            logger.warning(f"Chunk {chunk_id} already belongs to memory {existing_memory_id}, removing old association (now belongs to {memory.memory_id})")
                            # 从旧memory中移除这个chunk_id（如果旧memory还存在）
                            if existing_memory_id in self.memory_id_to_memory:
                                old_memory = self.memory_id_to_memory[existing_memory_id]
                                if chunk_id in old_memory.chunk_ids:
                                    old_memory.chunk_ids.remove(chunk_id)
                                    # 如果旧memory没有chunk_ids了，可能需要删除它
                                    if not old_memory.chunk_ids:
                                        logger.warning(f"Memory {existing_memory_id} has no chunk_ids after removing {chunk_id}, consider deleting it")
                                        if existing_memory_id in self.memory_id_to_memory:
                                            del self.memory_id_to_memory[existing_memory_id]
                    # 清空列表，只保留当前memory_id
                    self.chunk_id_to_memory_ids[chunk_id] = [memory.memory_id]
                else:
                    # 新chunk_id，直接添加
                    self.chunk_id_to_memory_ids[chunk_id] = [memory.memory_id]
            
            # 向后兼容：更新chunk_id_to_memory（取第一个memory）
            for chunk_id in memory.chunk_ids:
                if chunk_id not in self.chunk_id_to_memory or key == chunk_id:
                    self.chunk_id_to_memory[chunk_id] = memory
        
        logger.info(f"Saving {len(memories)} episodic memories (total: {len(self.memory_id_to_memory)})")
        self._save_memory_data()

    def save_one(self, memory: EpisodicMemory):
        """保存单个记忆"""
        if not hasattr(memory, 'memory_id') or not memory.memory_id:
            from .utils.misc_utils import compute_mdhash_id
            sorted_chunk_ids = sorted(memory.chunk_ids)
            memory.memory_id = compute_mdhash_id(','.join(sorted_chunk_ids), prefix="memory-")
        self.save({memory.memory_id: memory})

    def load(self) -> Dict[str, EpisodicMemory]:
        """加载所有记忆（返回memory_id -> memory的映射）"""
        return deepcopy(self.memory_id_to_memory)

    def delete(self, chunk_ids: List[str]):
        """
        删除指定chunk_ids对应的记忆。
        注意：如果多个chunk_id指向同一个memory，删除所有chunk_id后才会删除memory。
        """
        deleted_memory_ids = set()
        for chunk_id in chunk_ids:
            if chunk_id in self.chunk_id_to_memory_ids:
                memory_ids = self.chunk_id_to_memory_ids[chunk_id]
                deleted_memory_ids.update(memory_ids)
                # 从索引中移除
                del self.chunk_id_to_memory_ids[chunk_id]
            # 向后兼容
            if chunk_id in self.chunk_id_to_memory:
                del self.chunk_id_to_memory[chunk_id]
        
        # 检查哪些memory需要删除（所有chunk_id都被删除的memory）
        for memory_id in list(deleted_memory_ids):
            memory = self.memory_id_to_memory.get(memory_id)
            if memory:
                # 检查是否还有chunk_id指向这个memory
                should_delete = True
                for chunk_id in memory.chunk_ids:
                    if chunk_id in self.chunk_id_to_memory_ids and memory_id in self.chunk_id_to_memory_ids[chunk_id]:
                        should_delete = False
                        break
                if should_delete:
                    del self.memory_id_to_memory[memory_id]
        
        if deleted_memory_ids:
            logger.info(f"Deleted {len(deleted_memory_ids)} episodic memories")
            self._save_memory_data()
            self._delete_embeddings(list(deleted_memory_ids))

    def get_all_ids(self) -> List[str]:
        """获取所有memory_id（向后兼容：也返回chunk_ids）"""
        return list(self.memory_id_to_memory.keys())

    def get_all_memories(self) -> Dict[str, EpisodicMemory]:
        """获取所有记忆（返回memory_id -> memory的映射）"""
        return deepcopy(self.memory_id_to_memory)

    def exists(self, chunk_id: str) -> bool:
        """检查chunk_id是否存在对应的记忆"""
        return chunk_id in self.chunk_id_to_memory_ids or chunk_id in self.chunk_id_to_memory

    def get_missing_chunk_ids(self, chunk_ids: List[str]) -> List[str]:
        """获取缺失的chunk_ids"""
        return [chunk_id for chunk_id in chunk_ids if chunk_id not in self.chunk_id_to_memory_ids]

    def _delete_embeddings(self, memory_ids: List[str]):
        """删除指定memory_ids的embeddings"""
        if not memory_ids:
            return
        indices_to_delete = []
        for memory_id in memory_ids:
            if memory_id in self.memory_id_to_idx:
                indices_to_delete.append(self.memory_id_to_idx[memory_id])
        if not indices_to_delete:
            return
        sorted_indices = sorted(set(indices_to_delete), reverse=True)
        for idx in sorted_indices:
            if idx < len(self.memory_ids_emb):
                self.memory_ids_emb.pop(idx)
                self.embeddings.pop(idx)
        self.memory_id_to_idx = {memory_id: idx for idx, memory_id in enumerate(self.memory_ids_emb)}
        logger.info(f"Deleted {len(sorted_indices)} episodic memory embeddings")
        self._save_embedding_data()

    def insert_embeddings(self, memory_ids: List[str], embeddings: List[np.ndarray]):
        """插入或更新memory_ids的embeddings"""
        if not memory_ids or not embeddings:
            return
        if len(memory_ids) != len(embeddings):
            raise ValueError(f"Mismatch: {len(memory_ids)} memory_ids vs {len(embeddings)} embeddings")
        new_memory_ids = []
        new_embeddings = []
        for memory_id, embedding in zip(memory_ids, embeddings):
            if memory_id not in self.memory_id_to_idx:
                new_memory_ids.append(memory_id)
                new_embeddings.append(np.array(embedding, dtype=np.float32))
            else:
                idx = self.memory_id_to_idx[memory_id]
                self.embeddings[idx] = np.array(embedding, dtype=np.float32)
        if new_memory_ids:
            self.memory_ids_emb.extend(new_memory_ids)
            self.embeddings.extend(new_embeddings)
            self.memory_id_to_idx = {memory_id: idx for idx, memory_id in enumerate(self.memory_ids_emb)}
            logger.info(f"Inserted {len(new_memory_ids)} new episodic memory embeddings, updated {len(memory_ids) - len(new_memory_ids)} existing")
            self._save_embedding_data()

    def compute_and_insert_embeddings(self, memory_ids: List[str], memories: List[EpisodicMemory]):
        """计算并插入memory_ids的embeddings"""
        if not self.embedding_model:
            raise ValueError("Embedding model is not set. Cannot compute embeddings.")
        if len(memory_ids) != len(memories):
            raise ValueError(f"Mismatch: {len(memory_ids)} memory_ids vs {len(memories)} memories")
        texts_to_embed = []
        for memory in memories:
            text_parts = [memory.summary]
            # 从每个事件的结构化要素中构建文本
            if memory.events:
                for event in memory.events:
                    if isinstance(event, dict):
                        event_parts = []
                        participants = event.get("participants", [])
                        # 兼容旧数据: 如果没有participants但有who
                        if not participants and "who" in event:
                            participants = event.get("who", [])

                        if participants:
                            # 过滤掉 None 值，并确保所有元素都是字符串
                            if isinstance(participants, list):
                                participants_clean = [str(p) for p in participants if p is not None]
                                participants_str = ', '.join(participants_clean) if participants_clean else ''
                            else:
                                participants_str = str(participants) if participants is not None else ''
                            if participants_str:
                                event_parts.append(f"Participants: {participants_str}")
                        
                        action = event.get("action", [])
                        # 兼容旧数据: 如果没有action但有what
                        if not action and "what" in event:
                            action = event.get("what", [])
                        
                        if action:
                            # 过滤掉 None 值，并确保所有元素都是字符串
                            if isinstance(action, list):
                                action_clean = [str(a) for a in action if a is not None]
                                action_str = ', '.join(action_clean) if action_clean else ''
                            else:
                                action_str = str(action) if action is not None else ''
                            if action_str:
                                event_parts.append(f"Action: {action_str}")
                        
                        time = event.get("time") or event.get("when")
                        if time:
                            event_parts.append(f"Time: {time}")
                        
                        location = event.get("location") or event.get("where")
                        if location:
                            event_parts.append(f"Location: {location}")
                        
                        reason = event.get("reason") or event.get("why")
                        if reason:
                            event_parts.append(f"Reason: {reason}")
                        
                        method = event.get("method") or event.get("how")
                        if method:
                            event_parts.append(f"Method: {method}")
                        
                        if event_parts:
                            text_parts.append(" | ".join(event_parts))
            # 不再将原文档内容添加到embedding计算中，以保持嵌入表示的一致性
            # 原文可以从chunk_embedding_store中获取，不需要在情境记忆中重复存储
            texts_to_embed.append(" | ".join(text_parts))
        all_embeddings = []
        for i in range(0, len(texts_to_embed), self.embedding_batch_size):
            batch_texts = texts_to_embed[i:i + self.embedding_batch_size]
            batch_embeddings = self.embedding_model.batch_encode(batch_texts, norm=True)
            if isinstance(batch_embeddings, list):
                all_embeddings.extend(batch_embeddings)
            else:
                all_embeddings.extend(batch_embeddings.tolist() if hasattr(batch_embeddings, 'tolist') else [batch_embeddings])
        self.insert_embeddings(memory_ids, all_embeddings)

    def _compute_memory_embedding(self, memory: EpisodicMemory, embedding_model: Optional[BaseEmbeddingModel] = None) -> np.ndarray:
        """计算单个memory的embedding"""
        model = embedding_model or self.embedding_model
        if not model:
            raise ValueError("Embedding model is not set. Cannot compute embeddings.")
        text_parts = [memory.summary]
        # 从每个事件的结构化要素中构建文本
        if memory.events:
            for event in memory.events:
                if isinstance(event, dict):
                    event_parts = []
                    participants = event.get("participants", [])
                    # 兼容旧数据
                    if not participants and "who" in event:
                        participants = event.get("who", [])
                    
                    if participants:
                        # 过滤掉 None 值，并确保所有元素都是字符串
                        if isinstance(participants, list):
                            participants_clean = [str(p) for p in participants if p is not None]
                            participants_str = ', '.join(participants_clean) if participants_clean else ''
                        else:
                            participants_str = str(participants) if participants is not None else ''
                        if participants_str:
                            event_parts.append(f"Participants: {participants_str}")
                    
                    action = event.get("action", [])
                    # 兼容旧数据
                    if not action and "what" in event:
                        action = event.get("what", [])
                    
                    if action:
                        # 过滤掉 None 值，并确保所有元素都是字符串
                        if isinstance(action, list):
                            action_clean = [str(a) for a in action if a is not None]
                            action_str = ', '.join(action_clean) if action_clean else ''
                        else:
                            action_str = str(action) if action is not None else ''
                        if action_str:
                            event_parts.append(f"Action: {action_str}")
                    
                    time = event.get("time") or event.get("when")
                    if time:
                        event_parts.append(f"Time: {time}")
                    
                    location = event.get("location") or event.get("where")
                    if location:
                        event_parts.append(f"Location: {location}")
                    
                    reason = event.get("reason") or event.get("why")
                    if reason:
                        event_parts.append(f"Reason: {reason}")
                    
                    method = event.get("method") or event.get("how")
                    if method:
                        event_parts.append(f"Method: {method}")
                    
                    if event_parts:
                        text_parts.append(" | ".join(event_parts))
        text_to_embed = " | ".join(text_parts)
        embedding = model.batch_encode([text_to_embed], norm=True)
        if isinstance(embedding, list):
            return np.array(embedding[0], dtype=np.float32)
        elif hasattr(embedding, '__len__') and len(embedding) > 0:
            return np.array(embedding[0] if isinstance(embedding[0], np.ndarray) else embedding, dtype=np.float32)
        else:
            return np.array(embedding, dtype=np.float32)
    
    def _get_memory_embedding(self, memory_id: str, dtype=np.float32) -> Optional[np.ndarray]:
        """获取memory_id对应的embedding"""
        if memory_id not in self.memory_id_to_idx:
            return None
        idx = self.memory_id_to_idx[memory_id]
        return np.array(self.embeddings[idx], dtype=dtype)

    def get_embedding(self, chunk_id: str, dtype=np.float32) -> Optional[np.ndarray]:
        """
        通过chunk_id获取embedding（向后兼容方法）。
        如果chunk_id对应多个memory，返回第一个的embedding。
        """
        if chunk_id in self.chunk_id_to_memory_ids:
            memory_ids = self.chunk_id_to_memory_ids[chunk_id]
            if memory_ids:
                return self._get_memory_embedding(memory_ids[0], dtype)
        # 向后兼容
        if chunk_id in self.chunk_id_to_idx:
            idx = self.chunk_id_to_idx[chunk_id]
            return np.array(self.embeddings[idx], dtype=dtype)
        return None

    def get_embeddings(self, chunk_ids: List[str], dtype=np.float32) -> List[Optional[np.ndarray]]:
        """批量获取embeddings（向后兼容方法）"""
        if not chunk_ids:
            return []
        results = []
        for chunk_id in chunk_ids:
            results.append(self.get_embedding(chunk_id, dtype))
        return results

    def get_all_embeddings(self, dtype=np.float32) -> Tuple[List[str], List[np.ndarray]]:
        """获取所有embeddings（返回memory_ids和embeddings）"""
        embeddings = [np.array(emb, dtype=dtype) for emb in self.embeddings]
        return self.memory_ids_emb.copy(), embeddings

    def extract_episodic_memory(self, chunk_id: str, chunk_text: str, llm: BaseLLM,
                                 prompt_template_manager: PromptTemplateManager) -> EpisodicMemory:
        try:
            prompt = prompt_template_manager.render(
                name='episodic_memory_extraction',
                chunk_text=chunk_text
            )
            response, metadata, cache_hit = llm.infer(prompt)
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
            else:
                parsed = json.loads(response)
            
            # 从 events 字段读取，确保每个事件都是字典格式
            events = parsed.get("events", [])
            # 验证并规范化 events 格式
            normalized_events = []
            for event in events:
                if isinstance(event, dict):
                    # 确保所有必需的键都存在
                    normalized_event = {
                        "participants": event.get("participants", event.get("who", [])),
                        "action": event.get("action", event.get("what", [])),
                        "time": event.get("time", event.get("when")),
                        "location": event.get("location", event.get("where")),
                        "reason": event.get("reason", event.get("why")),
                        "method": event.get("method", event.get("how"))
                    }
                    normalized_events.append(normalized_event)
                else:
                    logger.warning(f"Event is not a dict, skipping: {event}")
            
            # 生成memory_id
            from .utils.misc_utils import compute_mdhash_id
            memory_id = compute_mdhash_id(chunk_id, prefix="memory-")
            
            memory = EpisodicMemory(
                memory_id=memory_id,
                chunk_ids=[chunk_id],
                summary=parsed.get("summary", ""),
                events=normalized_events
                # 不再保存原文档内容，原文从chunk_embedding_store中获取
            )
            return memory
        except Exception as e:
            logger.error(f"Error extracting episodic memory for chunk {chunk_id}: {e}")
            # 返回空 events 列表，timestamp 会自动设置
            from .utils.misc_utils import compute_mdhash_id
            memory_id = compute_mdhash_id(chunk_id, prefix="memory-")
            return EpisodicMemory(
                memory_id=memory_id,
                chunk_ids=[chunk_id],
                summary="",
                events=[]
                # 不再保存原文档内容，原文从chunk_embedding_store中获取
            )


    def extract_batch_episodic_memories(self, chunk_ids: List[str], chunk_texts: List[str],
                                        llm: BaseLLM, prompt_template_manager: PromptTemplateManager,
                                        batch_size: int = 10) -> Dict[str, EpisodicMemory]:
        """
        并发提取 Episodic Memory。
        """
        if len(chunk_ids) != len(chunk_texts):
            raise ValueError(f"Mismatch: {len(chunk_ids)} chunk_ids vs {len(chunk_texts)} texts")

        # 引入必要的工具函数 (保持原有的引用方式)
        from .utils.misc_utils import compute_mdhash_id
        
        results = {}
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0
        
        # 定义单个任务的处理函数
        def _process_single_chunk(c_id, c_text):
            """
            处理单个 chunk 并返回结果和统计信息。
            返回: (chunk_id, memory_object, metadata, is_cache_hit)
            """
            try:
                prompt = prompt_template_manager.render(
                    name='episodic_memory_extraction',
                    chunk_text=c_text
                )
                response, metadata, cache_hit = llm.infer(prompt)
                
                # 解析 JSON
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed = json.loads(json_str)
                else:
                    parsed = json.loads(response)
                
                # 规范化 events
                events = parsed.get("events", [])
                normalized_events = []
                for event in events:
                    if isinstance(event, dict):
                        normalized_event = {
                            "participants": event.get("participants", event.get("who", [])),
                            "action": event.get("action", event.get("what", [])),
                            "time": event.get("time", event.get("when")),
                            "location": event.get("location", event.get("where")),
                            "reason": event.get("reason", event.get("why")),
                            "method": event.get("method", event.get("how"))
                        }
                        normalized_events.append(normalized_event)
                    else:
                        logger.warning(f"Event is not a dict, skipping: {event}")
                
                memory_id = compute_mdhash_id(c_id, prefix="memory-")
                
                memory = EpisodicMemory(
                    memory_id=memory_id,
                    chunk_ids=[c_id],
                    summary=parsed.get("summary", ""),
                    events=normalized_events
                )
                return c_id, memory, metadata, cache_hit

            except Exception as e:
                logger.error(f"Error extracting episodic memory for chunk {c_id}: {e}")
                # 出错时返回空 Memory
                memory_id = compute_mdhash_id(c_id, prefix="memory-")
                empty_memory = EpisodicMemory(
                    memory_id=memory_id,
                    chunk_ids=[c_id],
                    summary="",
                    events=[]
                )
                return c_id, empty_memory, {}, False

        # 开始并发执行
        total_count = len(chunk_ids)
        
        # 使用 ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            # 提交所有任务
            future_to_chunk = {
                executor.submit(_process_single_chunk, chunk_ids[i], chunk_texts[i]): chunk_ids[i]
                for i in range(total_count)
            }
            
            # 使用 tqdm 显示进度，按照任务完成的顺序处理结果
            pbar = tqdm(total=total_count, desc="Generating episodic memories (Concurrent)")
            
            for future in as_completed(future_to_chunk):
                c_id, memory, metadata, cache_hit = future.result()
                
                # 收集结果
                results[c_id] = memory
                
                # 聚合统计信息
                if metadata:
                    total_prompt_tokens += metadata.get('prompt_tokens', 0)
                    total_completion_tokens += metadata.get('completion_tokens', 0)
                if cache_hit:
                    num_cache_hit += 1
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    'prompt_tokens': total_prompt_tokens,
                    'completion_tokens': total_completion_tokens,
                    'cache_hits': num_cache_hit
                })
                
            pbar.close()

        return results

    def find_related_chunks_by_knn(self, target_chunk_id: str, k: int = 5,
                                    query_batch_size: int = 1000, key_batch_size: int = 10000,
                                    candidate_chunk_ids: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        基于相似度查找关联的 chunk。
        
        Args:
            target_chunk_id: 目标 chunk ID
            k: 返回 top-k 个关联 chunk
            query_batch_size: 查询批次大小
            key_batch_size: 键批次大小
            candidate_chunk_ids: 可选的候选 chunk ID 列表。如果为 None，则从所有 chunk 中搜索（全局搜索）
        
        Returns:
            List[Tuple[str, float]]: (chunk_id, similarity_score) 列表
        """
        if target_chunk_id not in self.chunk_id_to_idx:
            logger.warning(f"Target chunk {target_chunk_id} does not have embedding")
            return []
        
        target_embedding = self.get_embedding(target_chunk_id)
        if target_embedding is None:
            return []
        
        # 如果指定了候选集合，只从候选集合中搜索；否则从所有 chunk 中搜索
        if candidate_chunk_ids is not None:
            # 增量式搜索：只在指定的候选集合中搜索
            candidate_chunk_ids = [cid for cid in candidate_chunk_ids 
                                 if cid != target_chunk_id and cid in self.chunk_id_to_idx]
        else:
            # 全局搜索：从所有 chunk 中搜索（保持向后兼容）
            candidate_chunk_ids = [cid for cid in self.chunk_ids_emb if cid != target_chunk_id]
        
        if not candidate_chunk_ids:
            return []
        
        if len(candidate_chunk_ids) < 1:
            return []
        
        candidate_embeddings = [self.get_embedding(cid) for cid in candidate_chunk_ids]
        valid_pairs = [(cid, emb) for cid, emb in zip(candidate_chunk_ids, candidate_embeddings) if emb is not None]
        if not valid_pairs:
            return []
        candidate_chunk_ids, candidate_embeddings = zip(*valid_pairs)
        candidate_chunk_ids = list(candidate_chunk_ids)
        candidate_embeddings = list(candidate_embeddings)
        knn_results = retrieve_knn(
            query_ids=[target_chunk_id],
            key_ids=candidate_chunk_ids,
            query_vecs=[target_embedding],
            key_vecs=candidate_embeddings,
            k=min(k, len(candidate_chunk_ids)),
            query_batch_size=query_batch_size,
            key_batch_size=key_batch_size
        )
        if target_chunk_id not in knn_results:
            return []
        related_chunk_ids, scores = knn_results[target_chunk_id]
        return list(zip(related_chunk_ids, scores))

    def filter_related_chunks_by_llm(self, target_chunk_id: str, candidate_chunk_ids: List[str],
                                      llm: BaseLLM, prompt_template_manager: PromptTemplateManager) -> List[str]:
        if target_chunk_id not in self.chunk_id_to_memory:
            logger.warning(f"Target chunk {target_chunk_id} does not have episodic memory")
            return []
        if not candidate_chunk_ids:
            return []
        target_memory = self.chunk_id_to_memory[target_chunk_id]
        candidate_memories = {}
        for cid in candidate_chunk_ids:
            if cid in self.chunk_id_to_memory:
                candidate_memories[cid] = self.chunk_id_to_memory[cid]
        if not candidate_memories:
            return []
        try:
            candidate_chunks_jsonl = []
            for cid, mem in candidate_memories.items():
                # 传递 events（已经是字典列表格式）
                candidate_chunks_jsonl.append({
                    "chunk_id": cid,
                    "summary": mem.summary,
                    "events": mem.events
                })
            # 传递 target_events_json（已经是字典列表格式）
            prompt = prompt_template_manager.render(
                name='episodic_memory_relation_filter',
                target_chunk_id=target_chunk_id,
                target_summary=target_memory.summary,
                target_events_json=json.dumps(target_memory.events, ensure_ascii=False),
                candidate_chunks_jsonl="\n".join([json.dumps(c, ensure_ascii=False) for c in candidate_chunks_jsonl]),
                k_value=len(candidate_chunk_ids)
            )
            response, metadata, cache_hit = llm.infer(prompt)
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
            else:
                parsed = json.loads(response)
            related_chunk_ids = parsed.get("related_chunk_ids", [])
            related_chunk_ids = [cid for cid in related_chunk_ids if cid in candidate_chunk_ids]
            return related_chunk_ids
        except Exception as e:
            logger.error(f"Error filtering related chunks for {target_chunk_id}: {e}")
            return candidate_chunk_ids

    def find_related_chunks_incremental(self, new_chunk_ids: List[str],
                                         llm: Optional[BaseLLM] = None,
                                         prompt_template_manager: Optional[PromptTemplateManager] = None,
                                         related_chunks_top_k: int = 5,
                                         related_chunks_llm_filter: bool = True,
                                         max_related_chunks: int = 10) -> Dict[str, List[str]]:
        """
        增量式关联生成：基于相似度进行增量更新。
        
        流程：
        1. 批次内关联：新批次内的 chunk 之间先进行关联
        2. 跨批次关联：新批次与历史记忆进行关联
        
        Args:
            new_chunk_ids: 新批次的情境记忆 chunk ID 列表
            llm: LLM 模型（用于过滤）
            prompt_template_manager: Prompt 模板管理器
            related_chunks_top_k: KNN 检索的 top-k
            related_chunks_llm_filter: 是否使用 LLM 过滤
        
        Returns:
            Dict[str, List[str]]: 每个 chunk_id 对应的 related_chunk_ids 列表
        """
        result = {}  # chunk_id -> List[related_chunk_id]
        
        if not new_chunk_ids:
            return result
        
        # 获取所有已存在的 chunk IDs（用于跨批次关联）
        all_existing_chunk_ids = set(self.chunk_id_to_memory.keys())
        new_chunk_ids_set = set(new_chunk_ids)
        old_chunk_ids = list(all_existing_chunk_ids - new_chunk_ids_set)
        
        logger.info(f"Incremental relationship generation: {len(new_chunk_ids)} new chunks, {len(old_chunk_ids)} old chunks")
        logger.info(f"Generating relationships for {len(new_chunk_ids)} new chunks (max_related={max_related_chunks}, llm_filter={related_chunks_llm_filter})")
        
        # 使用 tqdm 显示总体进度
        valid_new_chunk_ids = [cid for cid in new_chunk_ids if cid in self.chunk_id_to_memory]
        total_chunks = len(valid_new_chunk_ids)
        
        if total_chunks == 0:
            logger.warning("No valid new chunks found in memory store")
            return result
        
        pbar = tqdm(total=total_chunks, desc="Finding related chunks", unit="chunk")
        processed_count = 0
        chunks_with_relations = 0
        total_relations_found = 0
        
        for i, chunk_id in enumerate(new_chunk_ids):
            if chunk_id not in self.chunk_id_to_memory:
                continue

            combined_candidates: List[str] = []

            # Step 1: candidates from current batch (only previously processed new chunks)
            batch_candidates = new_chunk_ids[:i]
            if batch_candidates:
                knn_results = self.find_related_chunks_by_knn(
                    target_chunk_id=chunk_id,
                    k=min(related_chunks_top_k, len(batch_candidates)),
                    candidate_chunk_ids=batch_candidates
                )
                if knn_results:
                    combined_candidates.extend([cid for cid, _ in knn_results])

            # Step 2: candidates from old chunks
            if old_chunk_ids:
                knn_results = self.find_related_chunks_by_knn(
                    target_chunk_id=chunk_id,
                    k=min(related_chunks_top_k, len(old_chunk_ids)),
                    candidate_chunk_ids=old_chunk_ids
                )
                if knn_results:
                    combined_candidates.extend([cid for cid, _ in knn_results])

            # Deduplicate while preserving order and remove self references
            seen = set()
            ordered_candidates = []
            for cid in combined_candidates:
                if cid == chunk_id or cid in seen:
                    continue
                seen.add(cid)
                ordered_candidates.append(cid)

            if not ordered_candidates:
                result[chunk_id] = []
                processed_count += 1
                pbar.update(1)
                pbar.set_postfix({
                    'processed': f'{processed_count}/{total_chunks}',
                    'with_relations': chunks_with_relations,
                    'total_relations': total_relations_found
                })
                continue

            # LLM filtering on combined candidates (if enabled)
            if related_chunks_llm_filter and llm is not None and prompt_template_manager is not None:
                filtered_ids = self.filter_related_chunks_by_llm(
                    target_chunk_id=chunk_id,
                    candidate_chunk_ids=ordered_candidates,
                    llm=llm,
                    prompt_template_manager=prompt_template_manager
                )
                ordered_candidates = filtered_ids

            final_related_chunks = ordered_candidates[:max_related_chunks] if ordered_candidates else []
            result[chunk_id] = final_related_chunks
            
            # 更新统计信息
            processed_count += 1
            if final_related_chunks:
                chunks_with_relations += 1
                total_relations_found += len(final_related_chunks)
            
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                'processed': f'{processed_count}/{total_chunks}',
                'with_relations': chunks_with_relations,
                'total_relations': total_relations_found
            })
        
        pbar.close()

        for chunk_id in new_chunk_ids:
            if chunk_id not in result:
                result[chunk_id] = []
        
        # 输出最终统计信息
        chunks_with_relations = sum(1 for rels in result.values() if rels)
        total_relations = sum(len(rels) for rels in result.values())
        logger.info(f"Incremental relationship generation completed:")
        logger.info(f"  - Total chunks processed: {len(result)}")
        logger.info(f"  - Chunks with relations: {chunks_with_relations} ({chunks_with_relations/len(result)*100:.1f}%)")
        logger.info(f"  - Total relations found: {total_relations} (avg: {total_relations/len(result):.2f} per chunk)")
        
        return result

    # ========== 多对一整合相关方法 ==========
    
    def _generate_memory_id(self, chunk_ids: List[str]) -> str:
        """生成memory_id（基于chunk_ids的hash）"""
        from .utils.misc_utils import compute_mdhash_id
        sorted_chunk_ids = sorted(chunk_ids)
        return compute_mdhash_id(','.join(sorted_chunk_ids), prefix="memory-")
    
    def _llm_judge_integration(
        self,
        memory1: EpisodicMemory,
        memory2: EpisodicMemory,
        llm: BaseLLM,
        prompt_template_manager: PromptTemplateManager
    ) -> bool:
        """
        使用LLM判断两个情境记忆是否应该整合。
        
        Returns:
            True: 应该整合
            False: 不应该整合
        """
        try:
            prompt = prompt_template_manager.render(
                name='episodic_memory_integration_judge',
                memory1_summary=memory1.summary,
                memory1_events_json=json.dumps(memory1.events, ensure_ascii=False),
                memory2_summary=memory2.summary,
                memory2_events_json=json.dumps(memory2.events, ensure_ascii=False)
            )
            
            response, metadata, cache_hit = llm.infer(prompt)
            
            # 解析LLM响应
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                parsed = json.loads(response)
            
            should_integrate = parsed.get("should_integrate", False)
            if isinstance(should_integrate, str):
                should_integrate = should_integrate.lower() == "true"
            
            reason = parsed.get("reason", "")
            
            logger.info(f"LLM integration judgment: {should_integrate}, reason: {reason}")
            return bool(should_integrate)
        except Exception as e:
            logger.error(f"Error parsing LLM integration judgment: {e}")
            return False  # 默认不整合
    
    def _integrate_memories(
        self,
        memory1: EpisodicMemory,
        memory2: EpisodicMemory,
        llm: BaseLLM,
        prompt_template_manager: PromptTemplateManager,
        chunk_to_rows: Optional[Dict[str, Dict]] = None
    ) -> EpisodicMemory:
        """
        整合两个情境记忆，生成新的整合后的情境记忆。
        
        重要说明：
        - 整合后的event list长度不一定等于两个记忆的events数量相加
        - LLM需要分析事件关联，合并相同或相关的事件
        - 可能减少事件数量（去重、合并），也可能增加（发现新关联）
        
        Returns:
            整合后的EpisodicMemory对象
        """
        try:
            # 合并chunk_ids
            integrated_chunk_ids = list(set(memory1.chunk_ids + memory2.chunk_ids))
            
            # 获取原始文本（如果可用）
            memory1_original_texts = ""
            memory2_original_texts = ""
            if chunk_to_rows is not None:
                # 获取Memory 1的原始文本
                memory1_texts = []
                for chunk_id in memory1.chunk_ids:
                    if chunk_id in chunk_to_rows and "content" in chunk_to_rows[chunk_id]:
                        memory1_texts.append(f"[Chunk {chunk_id}]:\n{chunk_to_rows[chunk_id]['content']}")
                memory1_original_texts = "\n\n".join(memory1_texts) if memory1_texts else "Original text not available."
                
                # 获取Memory 2的原始文本
                memory2_texts = []
                for chunk_id in memory2.chunk_ids:
                    if chunk_id in chunk_to_rows and "content" in chunk_to_rows[chunk_id]:
                        memory2_texts.append(f"[Chunk {chunk_id}]:\n{chunk_to_rows[chunk_id]['content']}")
                memory2_original_texts = "\n\n".join(memory2_texts) if memory2_texts else "Original text not available."
            else:
                memory1_original_texts = "Original text not available."
                memory2_original_texts = "Original text not available."
            
            # 使用LLM整合summary和events（智能合并，而非简单拼接）
            prompt = prompt_template_manager.render(
                name='episodic_memory_integration',
                memory1_summary=memory1.summary,
                memory1_events_json=json.dumps(memory1.events, ensure_ascii=False),
                memory1_original_texts=memory1_original_texts,
                memory2_summary=memory2.summary,
                memory2_events_json=json.dumps(memory2.events, ensure_ascii=False),
                memory2_original_texts=memory2_original_texts
            )
            
            response, metadata, cache_hit = llm.infer(prompt)
            
            # 解析整合后的结果
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                parsed = json.loads(response)
            
            # 原逻辑如下：
            # integrated_summary = parsed.get("integrated_summary", "")
            # integrated_events = parsed.get("integrated_events", [])
            integrated_summary = parsed.get("integrated_summary", "")
            integrated_events = parsed.get("integrated_events", [])

            # 确保整合后的事件仍然覆盖所有来源事件，减少信息缺失
            integrated_events = self._ensure_event_coverage(
                integrated_events,
                (memory1.events or []) + (memory2.events or [])
            )
            
            # 生成新的memory_id（基于整合后的chunk_ids）
            integrated_memory_id = self._generate_memory_id(integrated_chunk_ids)
            
            # 合并related_memory_ids
            # 注意：整合后的记忆应该继承两个原记忆的关联关系
            integrated_related_ids = list(set(
                memory1.related_memory_ids + 
                memory2.related_memory_ids
            ))
            # 移除自己的ID（如果存在）
            if integrated_memory_id in integrated_related_ids:
                integrated_related_ids.remove(integrated_memory_id)
            # 注意：不添加memory1和memory2的ID，因为它们已经被整合，不再是独立的记忆
            
            # 使用较新的timestamp
            integrated_timestamp = max(
                memory1.timestamp or datetime.min,
                memory2.timestamp or datetime.min
            )
            
            integrated_memory = EpisodicMemory(
                memory_id=integrated_memory_id,
                chunk_ids=integrated_chunk_ids,
                summary=integrated_summary,
                events=integrated_events,
                related_memory_ids=integrated_related_ids,
                timestamp=integrated_timestamp
            )
            
            return integrated_memory
        except Exception as e:
            logger.error(f"Error integrating memories: {e}")
            # 如果整合失败，返回memory1（保留原有记忆）
            return memory1

    def _ensure_event_coverage(
        self,
        integrated_events: List[Dict[str, Any]],
        source_events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        合并LLM输出的事件与原始事件，避免遗漏。
        """
        merged_events: List[Dict[str, Any]] = []
        seen_signatures = set()

        def _append_event(event_obj):
            signature = self._event_signature(event_obj)
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                merged_events.append(event_obj)

        for event in integrated_events or []:
            normalized = self._normalize_event(event)
            if normalized:
                _append_event(normalized)

        # 防失真设计修改：不再强制追加原始事件，避免因描述微小差异导致的重复
        # 信任LLM的整合结果，如果LLM生成的事件列表非空，就不再追加原始事件
        # 仅当integrated_events为空时，才使用source_events作为兜底
        if not merged_events and source_events:
            for event in source_events or []:
                normalized = self._normalize_event(event)
                if normalized:
                    _append_event(normalized)

        # 原代码：强制追加原始事件
        # for event in source_events or []:
        #     normalized = self._normalize_event(event)
        #     if normalized:
        #         _append_event(normalized)

        return merged_events

    def _normalize_event(self, event: Any) -> Optional[Dict[str, Any]]:
        """将事件统一为dict结构，过滤空事件"""
        if isinstance(event, dict):
            # 确保有新键
            normalized = {
                "participants": event.get("participants", event.get("who", [])),
                "action": event.get("action", event.get("what", [])),
                "time": event.get("time", event.get("when")),
                "location": event.get("location", event.get("where")),
                "reason": event.get("reason", event.get("why")),
                "method": event.get("method", event.get("how"))
            }
            return normalized
        if event is None:
            return None
        # Fallback: 将字符串或其它结构转为action字段
        return {
            "participants": [],
            "action": [str(event)],
            "time": None,
            "location": None,
            "reason": None,
            "method": None
        }

    def _event_signature(self, event: Dict[str, Any]) -> str:
        """构建事件签名用于去重"""
        parts = []
        # 兼容新旧键
        participants = event.get("participants", event.get("who", []))
        action = event.get("action", event.get("what", []))
        
        for values in [participants, action]:
            if isinstance(values, list):
                normalized = [str(v).strip() for v in values if v is not None]
                parts.append("|".join(sorted(normalized)))
            elif values is not None:
                parts.append(str(values).strip())
            else:
                parts.append("")
        
        for key in ["time", "location", "reason", "method", "when", "where", "why", "how"]:
            # 优先使用新键，然后旧键
            value = event.get(key)
            # 避免重复添加（如果新旧键都存在且值相同，或者已经处理过）
            # 这里简化处理：只处理新键，因为 _normalize_event 已经统一了
            # 但为了健壮性，我们只处理新键的对应值
            pass
        
        # 重新按照标准顺序处理
        for key in ["time", "location", "reason", "method"]:
            value = event.get(key)
            if not value:
                # 尝试旧键
                old_key_map = {"time": "when", "location": "where", "reason": "why", "method": "how"}
                value = event.get(old_key_map.get(key))
            parts.append(str(value).strip() if value else "")
            
        return "||".join(parts)
    
    def _update_indices_after_integration(
        self,
        old_memory: EpisodicMemory,
        new_memory: EpisodicMemory
    ):
        """
        在记忆整合后更新索引。
        """
        # 如果old_memory被整合到new_memory中
        if old_memory.memory_id != new_memory.memory_id:
            # 1. 从memory_id_to_memory中移除旧记忆
            if old_memory.memory_id in self.memory_id_to_memory:
                del self.memory_id_to_memory[old_memory.memory_id]
            
            # 2. 更新chunk_id_to_memory_ids索引
            # 确保一个chunk_id只属于一个memory
            for chunk_id in old_memory.chunk_ids:
                # 将chunk_id从旧memory转移到新memory
                if chunk_id in self.chunk_id_to_memory_ids:
                    # 移除旧的memory_id
                    if old_memory.memory_id in self.chunk_id_to_memory_ids[chunk_id]:
                        self.chunk_id_to_memory_ids[chunk_id].remove(old_memory.memory_id)
                # 确保chunk_id只属于新memory
                self.chunk_id_to_memory_ids[chunk_id] = [new_memory.memory_id]
            
            # 3. 更新embedding索引
            if old_memory.memory_id in self.memory_id_to_idx:
                old_idx = self.memory_id_to_idx[old_memory.memory_id]
                # 移除旧的embedding
                if old_idx < len(self.memory_ids_emb):
                    self.memory_ids_emb.pop(old_idx)
                    self.embeddings.pop(old_idx)
                # 重建索引
                self.memory_id_to_idx = {
                    mid: idx for idx, mid in enumerate(self.memory_ids_emb)
                }
        
        # 添加/更新新记忆
        self.memory_id_to_memory[new_memory.memory_id] = new_memory
        
        # 更新chunk_id索引
        # 确保一个chunk_id只属于一个memory
        for chunk_id in new_memory.chunk_ids:
            # 如果chunk_id已经属于其他memory，先移除旧的关联
            if chunk_id in self.chunk_id_to_memory_ids:
                existing_memory_ids = self.chunk_id_to_memory_ids[chunk_id]
                for existing_memory_id in existing_memory_ids:
                    if existing_memory_id != new_memory.memory_id:
                        logger.warning(f"Chunk {chunk_id} already belongs to memory {existing_memory_id}, removing old association (now belongs to {new_memory.memory_id})")
            # 确保chunk_id只属于新memory
            self.chunk_id_to_memory_ids[chunk_id] = [new_memory.memory_id]
    
    def _recompute_embedding(self, memory: EpisodicMemory, embedding_model: BaseEmbeddingModel):
        """重新计算memory的embedding并更新"""
        embedding = self._compute_memory_embedding(memory, embedding_model)
        self.insert_embeddings([memory.memory_id], [embedding])
    
    def integrate_within_batch(
        self,
        new_memories: List[EpisodicMemory],
        embedding_model: BaseEmbeddingModel,
        llm: BaseLLM,
        prompt_template_manager: PromptTemplateManager,
        similarity_threshold: float = 0.7,
        chunk_to_rows: Optional[Dict[str, Dict]] = None
    ) -> List[EpisodicMemory]:
        """
        并发版批次内整合：
        1. 并发计算所有 Embedding。
        2. 并发进行两两相似度判断 (LLM Judge)。
        3. 基于判断结果构建连通分量。
        4. 并发对每个分量内的记忆进行归约合并。
        """
        if not new_memories:
            return []

        count = len(new_memories)
        if count == 1:
            return new_memories

        # ======================================================
        # Step 1: 并发计算所有 Embedding
        # ======================================================
        embeddings = [None] * count
        
        with ThreadPoolExecutor(max_workers=min(count, 20)) as executor:
            future_to_idx = {
                executor.submit(self._compute_memory_embedding, mem, embedding_model): i 
                for i, mem in enumerate(new_memories)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    embeddings[idx] = future.result()
                except Exception as e:
                    logger.error(f"Failed to compute embedding for memory {idx}: {e}")
                    embeddings[idx] = np.zeros(1) 

        # ======================================================
        # Step 2: 寻找合并候选 (Pair Generation)
        # 对于每个 item i (i>0)，在 0...i-1 中找一个最相似的 top-1
        # ======================================================
        # 存储待判定任务：(index_source, index_target)
        judge_tasks = []
        
        for i in range(1, count):
            current_emb = embeddings[i]
            best_sim = -1.0
            best_target_idx = -1
            
            # 在 i 之前的所有项中找 Top-1
            # 注意：这里使用的是"原始"状态的 embedding 进行初步筛选
            for j in range(i):
                prev_emb = embeddings[j]
                # 简单向量维度检查
                if current_emb.shape != prev_emb.shape: 
                    continue
                    
                sim = float(np.dot(current_emb, prev_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_target_idx = j
            
            if best_target_idx != -1:
                judge_tasks.append((i, best_target_idx))

        # ======================================================
        # Step 3: 并发 LLM 判决 (Parallel Judgment)
        # ======================================================
        # 邻接表：记录 LLM 认为应该合并的边
        adj_list = {i: set() for i in range(count)}
        
        def _run_judge(idx_a, idx_b):
            # 辅助函数：调用 LLM 判断
            mem_a = new_memories[idx_a] # Target (older)
            mem_b = new_memories[idx_b] # Source (newer)
            should = self._llm_judge_integration(
                memory1=mem_a,
                memory2=mem_b,
                llm=llm,
                prompt_template_manager=prompt_template_manager
            )
            return idx_a, idx_b, should

        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(_run_judge, t, s) for s, t in judge_tasks] # target(t) is older, source(s) is newer
            
            for future in as_completed(futures):
                try:
                    t_idx, s_idx, should_integrate = future.result()
                    if should_integrate:
                        # 建立无向边
                        adj_list[t_idx].add(s_idx)
                        adj_list[s_idx].add(t_idx)
                except Exception as e:
                    logger.error(f"Error during integration judgment: {e}")

        # ======================================================
        # Step 4: 查找连通分量 (Connected Components)
        # 使用 BFS/DFS 将所有连通的节点分为一组
        # ======================================================
        visited = set()
        groups = []

        for i in range(count):
            if i not in visited:
                component = []
                stack = [i]
                visited.add(i)
                while stack:
                    node = stack.pop()
                    component.append(node)
                    for neighbor in adj_list[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            stack.append(neighbor)
                # 保持时间顺序排序 (index 小的在前)
                component.sort()
                groups.append(component)

        # ======================================================
        # Step 5: 并发归约合并 (Parallel Reduce)
        # 每个组内部串行合并，但不同组并行处理
        # ======================================================
        final_memories = [None] * len(groups)

        def _merge_group(group_indices):
            # 取出该组的第一个作为基底
            base_idx = group_indices[0]
            merged_memory = new_memories[base_idx]
            
            # 依次将后续记忆合并到基底中
            for next_idx in group_indices[1:]:
                next_memory = new_memories[next_idx]
                merged_memory = self._integrate_memories(
                    memory1=merged_memory,
                    memory2=next_memory,
                    llm=llm,
                    prompt_template_manager=prompt_template_manager,
                    chunk_to_rows=chunk_to_rows
                )
            return merged_memory

        with ThreadPoolExecutor(max_workers=32) as executor:
            future_to_group_idx = {
                executor.submit(_merge_group, group): g_idx 
                for g_idx, group in enumerate(groups)
            }
            
            for future in as_completed(future_to_group_idx):
                g_idx = future_to_group_idx[future]
                try:
                    final_memories[g_idx] = future.result()
                except Exception as e:
                    logger.error(f"Error merging group {groups[g_idx]}: {e}")
                    # 出错兜底：返回该组第一个未合并的原始记忆
                    final_memories[g_idx] = new_memories[groups[g_idx][0]]

        # 过滤掉 None (理论上不应存在) 并返回
        return [m for m in final_memories if m is not None]
    
    def integrate_with_existing_memories(
        self,
        new_memories: List[EpisodicMemory],
        embedding_model: BaseEmbeddingModel,
        llm: BaseLLM,
        prompt_template_manager: PromptTemplateManager,
        similarity_threshold: float = 0.7,
        chunk_to_rows: Optional[Dict[str, Dict]] = None,
        chunk_embedding_store: Optional[Any] = None
    ) -> List[EpisodicMemory]:
        """
        跨批次整合：将新批次的情境记忆与已有记忆进行整合判断。
        
        算法逻辑：对每个新记忆，在已有记忆中找最相似的一个（top-1）。
        只要找到了最相似的候选（无论相似度如何），就调用LLM判断是否整合。
        只整合到最相似的一个记忆，不进行多候选比较。
        
        Args:
            new_memories: 新批次整合后的情境记忆列表
            embedding_model: 嵌入模型
            llm: LLM模型
            prompt_template_manager: Prompt模板管理器
            similarity_threshold: 已弃用（保留参数兼容性，不再使用）
        
        Returns:
            最终整合后的新记忆列表（可能数量减少）
        """
        if not self.memory_ids_emb:
            # 如果没有已有记忆，直接返回
            return new_memories
        
        # 获取所有已有记忆的embeddings
        existing_memory_ids, existing_embeddings = self.get_all_embeddings()
        existing_memories = {mid: self.memory_id_to_memory[mid] for mid in existing_memory_ids if mid in self.memory_id_to_memory}
        
        final_memories = []
        
        for new_memory in new_memories:
            # 计算新记忆的embedding
            new_embedding = self._compute_memory_embedding(new_memory, embedding_model)
            
            # 在已有记忆中查找最相似的一个（直接找top-1）
            if existing_embeddings:
                existing_embeddings_array = np.array(existing_embeddings)
                similarities = np.dot(existing_embeddings_array, new_embedding)
                best_idx = np.argmax(similarities)
                best_similarity = float(similarities[best_idx])
                best_candidate = existing_memories.get(existing_memory_ids[best_idx])
            else:
                best_candidate = None
                best_similarity = -1.0
            
            # 判断是否需要整合
            should_integrate = False
            if best_candidate:
                should_integrate = self._llm_judge_integration(
                    memory1=best_candidate,
                    memory2=new_memory,
                    llm=llm,
                    prompt_template_manager=prompt_template_manager
                )
            
            if should_integrate:
                # 准备chunk_to_rows（合并新记忆和已有记忆的chunk文本）
                merged_chunk_to_rows = {}
                if chunk_to_rows is not None:
                    merged_chunk_to_rows.update(chunk_to_rows)
                # 如果提供了chunk_embedding_store，尝试获取已有记忆的原始文本
                if chunk_embedding_store is not None:
                    for chunk_id in best_candidate.chunk_ids:
                        if chunk_id not in merged_chunk_to_rows:
                            try:
                                # get_row 返回 {"hash_id": chunk_id, "content": text}
                                row = chunk_embedding_store.get_row(chunk_id)
                                if row and "content" in row:
                                    merged_chunk_to_rows[chunk_id] = {"content": row["content"]}
                            except KeyError:
                                # chunk_id 不存在于 embedding_store 中，跳过
                                logger.debug(f"Chunk {chunk_id} not found in chunk_embedding_store, skipping original text retrieval")
                            except Exception as e:
                                logger.debug(f"Could not retrieve original text for chunk {chunk_id}: {e}")
                
                # 整合到已有记忆
                integrated_memory = self._integrate_memories(
                    memory1=best_candidate,
                    memory2=new_memory,
                    llm=llm,
                    prompt_template_manager=prompt_template_manager,
                    chunk_to_rows=merged_chunk_to_rows if merged_chunk_to_rows else None
                )
                # 更新已有记忆
                self.memory_id_to_memory[best_candidate.memory_id] = integrated_memory
                # 更新索引
                self._update_indices_after_integration(best_candidate, integrated_memory)
                # 重新计算embedding
                self._recompute_embedding(integrated_memory, embedding_model)
            else:
                # 保持独立，添加到最终列表
                final_memories.append(new_memory)
        
        return final_memories
    
    def _llm_judge_relationship(
        self,
        memory1: EpisodicMemory,
        memory2: EpisodicMemory,
        llm: BaseLLM,
        prompt_template_manager: PromptTemplateManager
    ) -> bool:
        """
        使用LLM判断两个情境记忆是否应该建立关联关系。
        
        与整合判断的区别：
        - 整合判断：两个记忆是否应该合并为一个记忆
        - 关联判断：两个记忆是否相关，但不应该合并（例如：不同时间/地点的事件）
        
        Returns:
            True: 应该建立关联
            False: 不应该建立关联
        """
        try:
            prompt = prompt_template_manager.render(
                name='episodic_memory_relationship_judge',
                memory1_summary=memory1.summary,
                memory1_events_json=json.dumps(memory1.events, ensure_ascii=False),
                memory1_chunk_ids=memory1.chunk_ids,
                memory2_summary=memory2.summary,
                memory2_events_json=json.dumps(memory2.events, ensure_ascii=False),
                memory2_chunk_ids=memory2.chunk_ids
            )
            
            response, metadata, cache_hit = llm.infer(prompt)
            
            # 解析LLM响应
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                parsed = json.loads(response)
            
            should_relate = parsed.get("should_relate", False)
            if isinstance(should_relate, str):
                should_relate = should_relate.lower() == "true"
            
            reason = parsed.get("reason", "")
            
            logger.info(f"LLM relationship judgment: {should_relate}, reason: {reason}")
            return bool(should_relate)
        except Exception as e:
            logger.error(f"Error parsing LLM relationship judgment: {e}")
            return False  # 默认不建立关联
    
    def build_related_memory_relationships(
        self,
        new_memories: List[EpisodicMemory],
        embedding_model: BaseEmbeddingModel,
        llm: BaseLLM,
        prompt_template_manager: PromptTemplateManager,
        relation_similarity_min: float = 0.6,  # 关联关系的最小相似度阈值
        relation_similarity_max: float = 0.7,  # 关联关系的最大相似度阈值（低于整合阈值）
        max_related_per_memory: int = 5
    ) -> Dict[str, List[str]]:
        """
        为新记忆构建关联关系（related_memory_ids）。
        
        策略：
        1. 相似度 >= relation_similarity_max：应该整合 → 已在整合阶段处理，不建立关联
        2. 相似度在 [relation_similarity_min, relation_similarity_max)：相关但不整合 → 建立关联
        3. 相似度 < relation_similarity_min：不相关 → 不建立关联
        
        Args:
            new_memories: 新批次整合后的情境记忆列表
            embedding_model: 嵌入模型
            llm: LLM模型
            prompt_template_manager: Prompt模板管理器
            relation_similarity_min: 关联关系的最小相似度阈值
            relation_similarity_max: 关联关系的最大相似度阈值（应低于整合阈值）
            max_related_per_memory: 每个记忆最多关联的记忆数量
        
        Returns:
            Dict[memory_id, List[related_memory_id]]: 每个记忆的关联记忆ID列表
        """
        # 已禁用：不再生成记忆间的关联关系
        # 返回空字典，保持接口兼容性
        result = {}
        for new_memory in new_memories:
            result[new_memory.memory_id] = []
            # 确保 related_memory_ids 为空列表
            new_memory.related_memory_ids = []
        return result