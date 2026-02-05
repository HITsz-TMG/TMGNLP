import json
import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Union, Optional, List, Set, Dict, Any, Tuple, Literal
import numpy as np
import importlib
from collections import defaultdict
from transformers import HfArgumentParser
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from igraph import Graph
import igraph as ig
import numpy as np
from collections import defaultdict
import re
import time

from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .information_extraction import OpenIE
from .information_extraction.openie_vllm_offline import VLLMOfflineOpenIE
from .information_extraction.openie_transformers_offline import TransformersOfflineOpenIE
from .evaluation.retrieval_eval import RetrievalRecall
from .evaluation.qa_eval import QAExactMatch, QAF1Score, QABLEU1
from .evaluation.qa_llm_judge import QALLMJudge
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .rerank import DSPyFilter, Qwen3Reranker
from .episodic_memory import EpisodicMemoryStore, EpisodicMemory
from .utils.misc_utils import *
from .utils.misc_utils import NerRawOutput, TripleRawOutput
from .utils.embed_utils import retrieve_knn
from .utils.typing import Triple
from .utils.config_utils import BaseConfig

logger = logging.getLogger(__name__)

class HippoRAG:

    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 llm_base_url=None,
                 embedding_model_name=None,
                 embedding_base_url=None,
                 azure_endpoint=None,
                 azure_embedding_endpoint=None):
        """
        Initializes an instance of the class and its related components.

        Attributes:
            global_config (BaseConfig): The global configuration settings for the instance. An instance
                of BaseConfig is used if no value is provided.
            saving_dir (str): The directory where specific HippoRAG instances will be stored. This defaults
                to `outputs` if no value is provided.
            llm_model (BaseLLM): The language model used for processing based on the global
                configuration settings.
            openie (Union[OpenIE, VLLMOfflineOpenIE]): The Open Information Extraction module
                configured in either online or offline mode based on the global settings.
            graph: The graph instance initialized by the `initialize_graph` method.
            embedding_model (BaseEmbeddingModel): The embedding model associated with the current
                configuration.
            chunk_embedding_store (EmbeddingStore): The embedding store handling chunk embeddings.
            entity_embedding_store (EmbeddingStore): The embedding store handling entity embeddings.
            fact_embedding_store (EmbeddingStore): The embedding store handling fact embeddings.
            prompt_template_manager (PromptTemplateManager): The manager for handling prompt templates
                and roles mappings.
            openie_results_path (str): The file path for storing Open Information Extraction results
                based on the dataset and LLM name in the global configuration.
            rerank_filter (Optional[DSPyFilter]): The filter responsible for reranking information
                when a rerank file path is specified in the global configuration.
            ready_to_retrieve (bool): A flag indicating whether the system is ready for retrieval
                operations.

        Parameters:
            global_config: The global configuration object. Defaults to None, leading to initialization
                of a new BaseConfig object.
            working_dir: The directory for storing working files. Defaults to None, constructing a default
                directory based on the class name and timestamp.
            llm_model_name: LLM model name, can be inserted directly as well as through configuration file.
            embedding_model_name: Embedding model name, can be inserted directly as well as through configuration file.
            llm_base_url: LLM URL for a deployed LLM model, can be inserted directly as well as through configuration file.
        """
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        #Overwriting Configuration if Specified
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        if embedding_base_url is not None:
            self.global_config.embedding_base_url = embedding_base_url

        if azure_endpoint is not None:
            self.global_config.azure_endpoint = azure_endpoint

        if azure_embedding_endpoint is not None:
            self.global_config.azure_embedding_endpoint = azure_embedding_endpoint

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"HippoRAG init with config:\n  {_print_config}\n")

        #LLM and embedding model specific working directories are created under every specified saving directories
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.llm_model: BaseLLM = _get_llm_class(self.global_config)

        if self.global_config.openie_mode == 'online':
            self.openie = OpenIE(llm_model=self.llm_model)
        elif self.global_config.openie_mode == 'offline':
            self.openie = VLLMOfflineOpenIE(self.global_config)
        elif self.global_config.openie_mode ==  'Transformers-offline':
            self.openie = TransformersOfflineOpenIE(self.global_config)

        self.graph = self.initialize_graph()

        if self.global_config.openie_mode == 'offline':
            self.embedding_model = None
        else:
            self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
                embedding_model_name=self.global_config.embedding_model_name)(global_config=self.global_config,
                                                                              embedding_model_name=self.global_config.embedding_model_name)
        self.chunk_embedding_store = EmbeddingStore(self.embedding_model,
                                                    os.path.join(self.working_dir, "chunk_embeddings"),
                                                    self.global_config.embedding_batch_size, 'chunk')
        self.entity_embedding_store = EmbeddingStore(self.embedding_model,
                                                     os.path.join(self.working_dir, "entity_embeddings"),
                                                     self.global_config.embedding_batch_size, 'entity')
        self.fact_embedding_store = EmbeddingStore(self.embedding_model,
                                                   os.path.join(self.working_dir, "fact_embeddings"),
                                                   self.global_config.embedding_batch_size, 'fact')

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})

        self.openie_results_path = os.path.join(self.global_config.save_dir,f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.json')

        if getattr(self.global_config, 'rerank_llm_base_url', None) is not None:
            self.rerank_filter = Qwen3Reranker(self.global_config)
        else:
            self.rerank_filter = DSPyFilter(self)

        # Initialize EpisodicMemoryStore (no compression involved)
        if self.embedding_model is not None:
            self.episodic_memory_store = EpisodicMemoryStore(
                working_dir=self.working_dir,
                embedding_model=self.embedding_model,
                store_name="episodic_memories",
                embedding_batch_size=self.global_config.episodic_memory_embedding_batch_size
            )
        else:
            self.episodic_memory_store = None
            logger.warning("Embedding model is None, EpisodicMemoryStore will not be initialized. Episodic memory features will be disabled.")

        self.ready_to_retrieve = False

        self.ppr_time = 0
        self.rerank_time = 0
        self.all_retrieval_time = 0

        self.ent_node_to_chunk_ids = None

        self.fact_metadata: Dict[str, Dict[str, str]] = {}
        self.fact_metadata_path = os.path.join(self.working_dir, "fact_metadata.json")
        self.load_fact_metadata()

        # Statistics for latest retrieve() call
        self.last_retrieve_dpr_fallback_count = 0
        self.last_retrieve_total_queries = 0
        self.last_retrieve_dpr_fallback_ratio = 0.0


    def load_fact_metadata(self):
        """Load fact metadata from JSON file."""
        if os.path.exists(self.fact_metadata_path):
            try:
                with open(self.fact_metadata_path, 'r') as f:
                    self.fact_metadata = json.load(f)
                logger.info(f"Loaded fact metadata from {self.fact_metadata_path} with {len(self.fact_metadata)} facts")
            except Exception as e:
                logger.error(f"Error loading fact metadata: {e}")
                self.fact_metadata = {}
        else:
            self.fact_metadata = {}

    def save_fact_metadata(self):
        """Save fact metadata to JSON file."""
        try:
            with open(self.fact_metadata_path, 'w') as f:
                json.dump(self.fact_metadata, f, indent=2)
            logger.info(f"Saved fact metadata to {self.fact_metadata_path}")
        except Exception as e:
            logger.error(f"Error saving fact metadata: {e}")

    def initialize_graph(self):
        """
        Initializes a graph using a Pickle file if available or creates a new graph.

        The function attempts to load a pre-existing graph stored in a Pickle file. If the file
        is not present or the graph needs to be created from scratch, it initializes a new directed
        or undirected graph based on the global configuration. If the graph is loaded successfully
        from the file, pertinent information about the graph (number of nodes and edges) is logged.

        Returns:
            ig.Graph: A pre-loaded or newly initialized graph.

        Raises:
            None
        """
        self._graph_pickle_filename = os.path.join(
            self.working_dir, f"graph.pickle"
        )

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graph_pickle_filename):
                preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)

        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(
                f"Loaded graph from {self._graph_pickle_filename} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
            )
            return preloaded_graph

    def pre_openie(self,  docs: List[str]):
        logger.info(f"Indexing Documents")
        logger.info(f"Performing OpenIE Offline")

        chunks = self.chunk_embedding_store.get_missing_string_hash_ids(docs)

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k : chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows, triple_prompt_name='triple_extraction_with_time')
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        assert False, logger.info('Done with OpenIE, run online indexing for future retrieval.')

    def index(self, docs: List[str]):
        """
        Indexes the given documents based on the HippoRAG 2 framework which generates an OpenIE knowledge graph
        based on the given documents and encodes passages, entities and facts separately for later retrieval.

        Parameters:
            docs : List[str]
                A list of documents to be indexed.
        """

        logger.info(f"Indexing Documents")

        logger.info(f"Performing OpenIE")

        if self.global_config.openie_mode == 'offline':
            self.pre_openie(docs)

        self.chunk_embedding_store.insert_strings(docs)
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunk_to_rows.keys())
        new_openie_rows = {k : chunk_to_rows[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows, triple_prompt_name='triple_extraction_with_time')
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

        assert len(chunk_to_rows) == len(ner_results_dict) == len(triple_results_dict), f"len(chunk_to_rows): {len(chunk_to_rows)}, len(ner_results_dict): {len(ner_results_dict)}, len(triple_results_dict): {len(triple_results_dict)}"

        # prepare data_store
        chunk_ids = list(chunk_to_rows.keys())

        # Generate episodic memories before graph construction (optional)
        if self.episodic_memory_store is not None and self.global_config.generate_episodic_memory:
            self._generate_episodic_memories(chunk_ids, chunk_to_rows)

        # Process triples and extract time metadata
        raw_chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
        
        chunk_triples = []
        total_quadruples = 0
        total_time_stored = 0
        time_storage_by_chunk = {}
        
        for chunk_idx, triples in enumerate(raw_chunk_triples):
            chunk_id = chunk_ids[chunk_idx]
            sanitized_triples = []
            chunk_quadruples = 0
            chunk_time_stored = 0
            
            for t in triples:
                if len(t) >= 4:
                    # It's a quadruple (S, P, O, T)
                    total_quadruples += 1
                    chunk_quadruples += 1
                    s, p, o, time_val = t[0], t[1], t[2], t[3]
                    pure_triple = (s, p, o)
                    sanitized_triples.append(pure_triple)
                    
                    # Store metadata if time exists and is not null/empty
                    # Check for None, empty string, or string representations of null
                    is_valid_time = (
                        time_val is not None and 
                        time_val != "" and 
                        str(time_val).lower().strip() not in ("none", "null", "")
                    )
                    if is_valid_time:
                        # Compute hash id that matches what EmbeddingStore will generate
                        # Note: EmbeddingStore uses prefix="fact-" for fact store
                        fact_hash_id = compute_mdhash_id(str(pure_triple), prefix="fact-")
                        
                        if fact_hash_id not in self.fact_metadata:
                            self.fact_metadata[fact_hash_id] = {}
                        self.fact_metadata[fact_hash_id][chunk_id] = time_val
                        total_time_stored += 1
                        chunk_time_stored += 1
                        logger.debug(f"[Time Metadata] Stored time '{time_val}' for fact ({s}, {p}, {o}) in chunk {chunk_id}")
                    else:
                        logger.debug(f"[Time Metadata] No time value for quadruple ({s}, {p}, {o}) in chunk {chunk_id}")
                elif len(t) == 3:
                    sanitized_triples.append(tuple(t))
                else:
                    # Handle other cases if necessary, or skip
                    if len(t) > 0:
                        sanitized_triples.append(tuple(t[:3])) # Truncate if weirdly long but not 4 structured
            
            if chunk_quadruples > 0:
                time_storage_by_chunk[chunk_id] = {
                    'quadruples': chunk_quadruples,
                    'time_stored': chunk_time_stored
                }
            
            chunk_triples.append(sanitized_triples)
        
        # Log summary statistics
        logger.info(f"[Time Metadata] Indexing Summary: {total_quadruples} quadruples extracted, {total_time_stored} time values stored")
        if time_storage_by_chunk:
            logger.info(f"[Time Metadata] Time storage by chunk: {len(time_storage_by_chunk)} chunks with quadruples")
            for chunk_id, stats in list(time_storage_by_chunk.items())[:5]:  # Log first 5 chunks as examples
                logger.debug(f"[Time Metadata] Chunk {chunk_id}: {stats['quadruples']} quadruples, {stats['time_stored']} time values stored")

        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        logger.info(f"Encoding Entities")
        self.entity_embedding_store.insert_strings(entity_nodes)

        logger.info(f"Encoding Facts")
        self.fact_embedding_store.insert_strings([str(fact) for fact in facts])
        
        # Save metadata after encoding facts
        self.save_fact_metadata()

        logger.info(f"Constructing Graph")

        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}

        self.add_fact_edges(chunk_ids, chunk_triples)
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

        if num_new_chunks > 0:
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            self.add_synonymy_edges()

            self.augment_graph()
            self.save_igraph()

    def delete(self, docs_to_delete: List[str]):
        """
        Deletes the given documents from all data structures within the HippoRAG class.
        Note that triples and entities which are indexed from chunks that are not being removed will not be removed.

        Parameters:
            docs : List[str]
                A list of documents to be deleted.
        """

        #Making sure that all the necessary structures have been built.
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        current_docs = set(self.chunk_embedding_store.get_all_texts())
        docs_to_delete = [doc for doc in docs_to_delete if doc in current_docs]

        #Get ids for chunks to delete
        chunk_ids_to_delete = set(
            [self.chunk_embedding_store.text_to_hash_id[chunk] for chunk in docs_to_delete])

        #Find triples in chunks to delete
        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])
        triples_to_delete = []

        all_openie_info_with_deletes = []

        for openie_doc in all_openie_info:
            if openie_doc['idx'] in chunk_ids_to_delete:
                triples_to_delete.append(openie_doc['extracted_triples'])
            else:
                all_openie_info_with_deletes.append(openie_doc)

        triples_to_delete = flatten_facts(triples_to_delete)

        #Filter out triples that appear in unaltered chunks
        true_triples_to_delete = []

        for triple in triples_to_delete:
            proc_triple = tuple(text_processing(list(triple)))

            doc_ids = self.proc_triples_to_docs[str(proc_triple)]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                true_triples_to_delete.append(triple)

        processed_true_triples_to_delete = [[text_processing(list(triple)) for triple in true_triples_to_delete]]
        entities_to_delete, _ = extract_entity_nodes(processed_true_triples_to_delete)
        processed_true_triples_to_delete = flatten_facts(processed_true_triples_to_delete)

        triple_ids_to_delete = set([self.fact_embedding_store.text_to_hash_id[str(triple)] for triple in processed_true_triples_to_delete])

        #Filter out entities that appear in unaltered chunks
        ent_ids_to_delete = [self.entity_embedding_store.text_to_hash_id[ent] for ent in entities_to_delete]

        filtered_ent_ids_to_delete = []

        for ent_node in ent_ids_to_delete:
            doc_ids = self.ent_node_to_chunk_ids[ent_node]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                filtered_ent_ids_to_delete.append(ent_node)

        logger.info(f"Deleting {len(chunk_ids_to_delete)} Chunks")
        logger.info(f"Deleting {len(triple_ids_to_delete)} Triples")
        logger.info(f"Deleting {len(filtered_ent_ids_to_delete)} Entities")

        self.save_openie_results(all_openie_info_with_deletes)

        self.entity_embedding_store.delete(filtered_ent_ids_to_delete)
        self.fact_embedding_store.delete(triple_ids_to_delete)
        self.chunk_embedding_store.delete(chunk_ids_to_delete)

        #Delete Nodes from Graph
        self.graph.delete_vertices(list(filtered_ent_ids_to_delete) + list(chunk_ids_to_delete))
        self.save_igraph()

        self.ready_to_retrieve = False

    def _generate_episodic_memories(self, chunk_ids: List[str], chunk_to_rows: Dict[str, Dict]):
        if self.episodic_memory_store is None:
            logger.warning("EpisodicMemoryStore is not initialized, skipping episodic memory generation.")
            return
        logger.info(f"Generating episodic memories for {len(chunk_ids)} chunks")
        missing_chunk_ids = self.episodic_memory_store.get_missing_chunk_ids(chunk_ids)
        if not missing_chunk_ids:
            logger.info("All chunks already have episodic memories, skipping generation.")
            return
        logger.info(f"Generating episodic memories for {len(missing_chunk_ids)} new chunks")
        chunk_texts = [chunk_to_rows[chunk_id]["content"] for chunk_id in missing_chunk_ids]
        episodic_memories = self.episodic_memory_store.extract_batch_episodic_memories(
            chunk_ids=missing_chunk_ids,
            chunk_texts=chunk_texts,
            llm=self.llm_model,
            prompt_template_manager=self.prompt_template_manager,
            batch_size=self.global_config.episodic_memory_batch_size
        )
        # 检查是否启用多对一模式
        enable_multi_chunk = getattr(self.global_config, 'enable_multi_chunk_memory', False)
        
        if enable_multi_chunk:
            # 多对一模式：进行整合和关联
            logger.info("Multi-chunk memory mode enabled: performing integration and relationship building")
            
            # 1. 将提取的记忆转换为列表（按chunk顺序）
            memory_list = [episodic_memories[chunk_id] for chunk_id in missing_chunk_ids]
            
            # 2. 批次内整合
            similarity_threshold = getattr(self.global_config, 'memory_integration_similarity_threshold', 0.7)
            use_llm_filter = getattr(self.global_config, 'memory_integration_llm_filter', True)
            
            if use_llm_filter:
                logger.info(f"Performing within-batch integration (similarity_threshold={similarity_threshold})")
                integrated_memories = self.episodic_memory_store.integrate_within_batch(
                    new_memories=memory_list,
                    embedding_model=self.embedding_model,
                    llm=self.llm_model,
                    prompt_template_manager=self.prompt_template_manager,
                    similarity_threshold=similarity_threshold,
                    chunk_to_rows=chunk_to_rows
                )
                logger.info(f"Within-batch integration: {len(memory_list)} -> {len(integrated_memories)} memories")
            else:
                integrated_memories = memory_list
            
            # 3. 计算整合后记忆的embeddings
            logger.info("Computing episodic memory embeddings for integrated memories")
            memory_ids = [mem.memory_id for mem in integrated_memories]
            self.episodic_memory_store.compute_and_insert_embeddings(memory_ids, integrated_memories)
            
            # 4. 跨批次整合
            if use_llm_filter:
                logger.info(f"Performing cross-batch integration (similarity_threshold={similarity_threshold})")
                final_memories = self.episodic_memory_store.integrate_with_existing_memories(
                    new_memories=integrated_memories,
                    embedding_model=self.embedding_model,
                    llm=self.llm_model,
                    prompt_template_manager=self.prompt_template_manager,
                    similarity_threshold=similarity_threshold,
                    chunk_to_rows=chunk_to_rows,
                    chunk_embedding_store=self.chunk_embedding_store
                )
                logger.info(f"Cross-batch integration: {len(integrated_memories)} -> {len(final_memories)} new memories")
            else:
                final_memories = integrated_memories
            
            # 5. 构建关联关系（已禁用）
            # relation_min = getattr(self.global_config, 'memory_relation_similarity_min', 0.6)
            # relation_max = getattr(self.global_config, 'memory_relation_similarity_max', 0.7)
            # max_related = getattr(self.global_config, 'memory_relation_max_per_memory', 5)
            # relation_llm_filter = getattr(self.global_config, 'memory_relation_llm_filter', True)
            # 
            # if relation_llm_filter:
            #     logger.info(f"Building related memory relationships (similarity_range=[{relation_min}, {relation_max}), max_related={max_related})")
            #     relationships = self.episodic_memory_store.build_related_memory_relationships(
            #         new_memories=final_memories,
            #         embedding_model=self.embedding_model,
            #         llm=self.llm_model,
            #         prompt_template_manager=self.prompt_template_manager,
            #         relation_similarity_min=relation_min,
            #         relation_similarity_max=relation_max,
            #         max_related_per_memory=max_related
            #     )
            #     logger.info(f"Built relationships for {len(relationships)} memories")
            
            # 6. 保存最终的记忆
            logger.info("Saving integrated episodic memories")
            memories_dict = {mem.memory_id: mem for mem in final_memories}
            self.episodic_memory_store.save(memories_dict)
            
            logger.info(f"Successfully generated and saved {len(final_memories)} integrated episodic memories")
            return
        # 原有的一对一模式：保持向后兼容
        logger.info("Single-chunk memory mode: using original one-to-one approach")
        logger.info("Saving episodic memories metadata to store (before computing embeddings)")
        self.episodic_memory_store.save(episodic_memories)
        logger.info("Computing episodic memory embeddings")
        memory_list = [episodic_memories[chunk_id] for chunk_id in missing_chunk_ids]
        memory_ids = [mem.memory_id for mem in memory_list]
        self.episodic_memory_store.compute_and_insert_embeddings(memory_ids, memory_list)
        
        # 使用增量式关联生成方法
        related_chunks_top_k = self.global_config.related_chunks_top_k
        related_chunks_llm_filter = self.global_config.related_chunks_llm_filter
        logger.info(f"Finding related chunks incrementally (k={related_chunks_top_k}, llm_filter={related_chunks_llm_filter})")
        
        # 调用增量式关联生成方法
        # 该方法会：
        # 1. 先在新批次内进行关联（批次内关联）
        # 2. 然后新批次与历史记忆进行关联（跨批次关联）
        relationships = self.episodic_memory_store.find_related_chunks_incremental(
            new_chunk_ids=missing_chunk_ids,
            llm=self.llm_model,
            prompt_template_manager=self.prompt_template_manager,
            related_chunks_top_k=related_chunks_top_k,
            related_chunks_llm_filter=related_chunks_llm_filter,
            max_related_chunks=10
        )
        
        # 更新新 chunk 的 related_memory_ids（向后兼容：也更新related_chunk_ids）
        for chunk_id, related_chunk_ids in relationships.items():
            if chunk_id in episodic_memories:
                memory = episodic_memories[chunk_id]
                # 需要将chunk_ids转换为memory_ids（简化处理：暂时保持chunk_ids）
                # 注意：在旧模式下，related_chunk_ids和related_memory_ids可能不一致
                memory.related_memory_ids = related_chunk_ids  # 简化：直接使用chunk_ids
                try:
                    logger.info(f"[Episodic] chunk_id={chunk_id} related_memory_ids_len={len(related_chunk_ids)} related_memory_ids={related_chunk_ids}")
                except Exception:
                    logger.info(f"[Episodic] chunk_id={chunk_id} related_memory_ids logging failed.")
        
        logger.info("Updating episodic memories with incremental relationships and saving to store")
        # 保存新 chunk 的记忆
        self.episodic_memory_store.save(episodic_memories)
        
        logger.info(f"Successfully generated and saved {len(episodic_memories)} episodic memories with incremental relationships")

    def _get_episodic_memories_with_single_hop(self, chunk_ids: List[str]) -> List[EpisodicMemory]:
        """
        获取 chunk_ids 对应的 EpisodicMemory，并进行 single-hop 搜索扩展。
        支持多对一关系：一个chunk_id可能对应多个memory。
        通过related_memory_ids进行单跳扩展。

        Args:
            chunk_ids: 初始检索到的 chunk ID 列表

        Returns:
            List[EpisodicMemory]: 扩展后的 episodic memory 列表（包含初始的和单跳邻居的）
        """
        if self.episodic_memory_store is None:
            return []

        # 支持多对一关系：通过chunk_ids获取所有对应的memories
        chunk_to_memories = self.episodic_memory_store.get_memories_by_chunk_ids(chunk_ids)
        
        # 收集所有初始记忆（去重）
        all_memory_ids = set()
        initial_memories = []
        for chunk_id, memories in chunk_to_memories.items():
            for memory in memories:
                if memory.memory_id not in all_memory_ids:
                    all_memory_ids.add(memory.memory_id)
                    initial_memories.append(memory)
        
        episodic_memories: List[EpisodicMemory] = initial_memories.copy()
        seen_memory_ids = all_memory_ids.copy()

        # single-hop 扩展：通过related_memory_ids
        related_memory_ids_to_add = set()
        for memory in episodic_memories:
            if hasattr(memory, 'related_memory_ids') and memory.related_memory_ids:
                for related_memory_id in memory.related_memory_ids:
                    if related_memory_id not in seen_memory_ids:
                        related_memory_ids_to_add.add(related_memory_id)
            # 向后兼容：也支持related_chunk_ids（旧格式）
            elif hasattr(memory, 'related_chunk_ids') and memory.related_chunk_ids:
                # 将chunk_ids转换为memory_ids（简化处理：通过chunk_id查找对应的memory）
                for related_chunk_id in memory.related_chunk_ids:
                    related_memories = self.episodic_memory_store.get_memories_by_chunk_ids([related_chunk_id])
                    if related_chunk_id in related_memories:
                        for related_memory in related_memories[related_chunk_id]:
                            if related_memory.memory_id not in seen_memory_ids:
                                related_memory_ids_to_add.add(related_memory.memory_id)

        if related_memory_ids_to_add:
            for related_memory_id in related_memory_ids_to_add:
                related_memory = self.episodic_memory_store.get_by_memory_id(related_memory_id)
                if related_memory and related_memory.memory_id not in seen_memory_ids:
                    episodic_memories.append(related_memory)
                    seen_memory_ids.add(related_memory.memory_id)

        return episodic_memories

    def retrieve(self,
                 queries: List[str],
                 num_to_retrieve: int = None,
                 gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using the HippoRAG 2 framework, which consists of several steps:
        - Fact Retrieval
        - Recognition Memory for improved fact selection
        - Dense passage scoring
        - Personalized PageRank based re-ranking

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        # Track DPR fallback stats for this retrieve() call
        num_queries = len(queries)
        self.last_retrieve_total_queries = num_queries
        num_dpr_fallback = 0

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            rerank_start = time.time()
            query_fact_scores = self.get_fact_scores(query)
            top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, query_fact_scores)
            rerank_end = time.time()

            self.rerank_time += rerank_end - rerank_start

            if top_k_facts:
                logger.info(f"[Rerank Facts] Query: {query}")
                for idx, fact in enumerate(top_k_facts, 1):
                    logger.info(f"[Rerank Facts]   Fact {idx}: {fact}")
            else:
                logger.info(f"[Rerank Facts] Query: {query} - No reranked facts available")

            if len(top_k_facts) == 0:
                logger.info('No facts found after reranking, return DPR results')
                logger.info(f"Query: {query}")
                # 显示query_fact_scores中分数最高的Top_k个fact的分数
                if len(query_fact_scores) > 0:
                    top_k_scores = np.sort(query_fact_scores)[-self.global_config.linking_top_k:][::-1]
                    logger.info(f"Top {min(len(query_fact_scores), self.global_config.linking_top_k)} fact scores: {top_k_scores}")
                # logger.info(f"query_fact_scores: {query_fact_scores}")
                num_dpr_fallback += 1
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
                # When using DPR fallback, set top_k_facts to empty list
                top_k_facts = []
            else:
                sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(query=query,
                                                                                         link_top_k=self.global_config.linking_top_k,
                                                                                         query_fact_scores=query_fact_scores,
                                                                                         top_k_facts=top_k_facts,
                                                                                         top_k_fact_indices=top_k_fact_indices,
                                                                                         passage_node_weight=self.global_config.passage_node_weight)

            top_k_idx = sorted_doc_ids[:num_to_retrieve]
            top_k_chunk_ids = [self.passage_node_keys[idx] for idx in top_k_idx]
            top_k_docs = [self.chunk_embedding_store.get_row(cid)["content"] for cid in top_k_chunk_ids]

            # Attach episodic memories (directly from top_k_chunk_ids, no single-hop expansion) if available and enabled
            episodic_memories = []
            if self.global_config.use_episodic_memory and self.episodic_memory_store is not None:
                # 支持多对一关系：通过chunk_ids获取所有对应的memories
                chunk_to_memories = self.episodic_memory_store.get_memories_by_chunk_ids(top_k_chunk_ids)
                # 收集所有记忆（去重）
                all_memory_ids = set()
                for chunk_id, memories in chunk_to_memories.items():
                    for memory in memories:
                        if memory.memory_id not in all_memory_ids:
                            all_memory_ids.add(memory.memory_id)
                            episodic_memories.append(memory)
                # 可以按相关性排序
                episodic_memories.sort(key=lambda m: len(set(m.chunk_ids) & set(top_k_chunk_ids)), reverse=True)

            retrieval_results.append(QuerySolution(
                question=query,
                docs=top_k_docs,
                doc_scores=sorted_doc_scores[:num_to_retrieve],
                episodic_memories=episodic_memories if episodic_memories else None,
                top_k_facts=top_k_facts if top_k_facts else None
            ))

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")
        logger.info(f"Total Recognition Memory Time {self.rerank_time:.2f}s")
        logger.info(f"Total PPR Time {self.ppr_time:.2f}s")
        logger.info(f"Total Misc Time {self.all_retrieval_time - (self.rerank_time + self.ppr_time):.2f}s")

        # Log DPR fallback statistics for this retrieval
        self.last_retrieve_dpr_fallback_count = num_dpr_fallback
        self.last_retrieve_dpr_fallback_ratio = (num_dpr_fallback / num_queries) if num_queries > 0 else 0.0
        logger.info(
            f"DPR fallback occurred for {num_dpr_fallback}/{num_queries} queries ({self.last_retrieve_dpr_fallback_ratio:.2%})."
        )

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results], k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None,
               qa_items: Optional[List[Dict[str, Any]]] = None
               ) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using the HippoRAG 2 framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k、Exact Match、F1 与 BLEU-1 等指标。

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics（Exact Match、F1、BLEU-1 等）。

        """
        # LoCoMo adaptation: only enabled when qa_items carries category/adversarial metadata.
        is_locomo_format = (
            qa_items is not None
            and len(qa_items) == len(queries)
            and any(isinstance(qa, dict) and ("category" in qa or "adversarial_answer" in qa) for qa in qa_items)
        )

        # Optional LoCoMo preprocessing (question rewriting + metadata attachment)
        locomo_metadata_list: Optional[List[Dict[str, Any]]] = None
        if is_locomo_format:
            from hashlib import md5
            from .utils.locomo_qa_utils import preprocess_question

            locomo_metadata_list = []
            processed_questions: List[str] = []

            # Build processed question strings aligned with qa_items order
            for idx, qa in enumerate(qa_items):
                qa_id = str(qa.get("id", idx))
                seed = int(md5(qa_id.encode("utf-8")).hexdigest()[:8], 16)
                processed_q, meta = preprocess_question(qa, random_seed=seed)
                locomo_metadata_list.append(meta)
                processed_questions.append(processed_q)

            if isinstance(queries[0], QuerySolution):
                # Update questions in-place (retrieval already done)
                for qsol, new_q, meta in zip(queries, processed_questions, locomo_metadata_list):
                    qsol.question = new_q
                    qsol.locomo_metadata = meta
            else:
                # Replace raw queries (retrieval will be performed)
                queries = processed_questions

        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)
            qa_bleu1_evaluator = QABLEU1(global_config=self.global_config)
            # qa_llm_judge_evaluator = QALLMJudge(global_config=self.global_config, llm_model=self.llm_model)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve(queries=queries)
            # Attach LoCoMo metadata after retrieval (if enabled)
            if is_locomo_format and locomo_metadata_list is not None:
                for qsol, meta in zip(queries, locomo_metadata_list):
                    qsol.locomo_metadata = meta

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_bleu1_result, example_qa_bleu1_results = qa_bleu1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            
            # LLM as a Judge evaluation
            # questions = [qa_result.question for qa_result in queries_solutions]
            # overall_qa_llm_judge_result, example_qa_llm_judge_results = qa_llm_judge_evaluator.calculate_metric_scores(
            #     gold_answers=gold_answers, 
            #     predicted_answers=[qa_result.answer for qa_result in queries_solutions],
            #     questions=questions,
            #     aggregation_fn=np.max)

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_em_result.update(overall_qa_bleu1_result)
            # overall_qa_em_result.update(overall_qa_llm_judge_result)
            overall_qa_results = overall_qa_em_result

            # LoCoMo category-aware evaluation (additive; does not replace existing metrics)
            if is_locomo_format and qa_items is not None:
                try:
                    from .evaluation.locomo_eval import LocomoQAEvaluator

                    locomo_evaluator = LocomoQAEvaluator(global_config=self.global_config)
                    gold_answers_list = [list(g) if not isinstance(g, list) else g for g in gold_answers]
                    locomo_overall, _ = locomo_evaluator.calculate_metric_scores(
                        gold_answers=gold_answers_list,
                        predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                        qa_items=qa_items,
                        aggregation_fn=np.max,
                    )
                    overall_qa_results.update(locomo_overall)
                except Exception as e:
                    logger.warning(f"LoCoMo evaluator failed, skipping LoCoMo metrics. Error: {e}")

            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Log incorrect QA cases (EM=0 or F1 < threshold)
            f1_threshold = 0.5  # Threshold for F1 score to consider as incorrect
            incorrect_cases = []
            for idx, (query_solution, em_result, f1_result, bleu1_result) in enumerate(zip(queries_solutions, example_qa_em_results, example_qa_f1_results, example_qa_bleu1_results)):
                em_score = em_result.get("ExactMatch", 0.0)
                f1_score = f1_result.get("F1", 0.0)
                bleu1_score = bleu1_result.get("BLEU-1", 0.0)
                
                # Consider as incorrect if F1 < threshold
                if f1_score < f1_threshold:
                    case_dict = {
                        "index": idx,
                        "question": query_solution.question,
                        "predicted_answer": query_solution.answer,
                        "gold_answers": list(gold_answers[idx]),
                        "em_score": em_score,
                        "f1_score": f1_score,
                        "bleu1_score": bleu1_score
                    }
                    # Add gold_docs if available
                    if gold_docs is not None:
                        case_dict["gold_docs"] = gold_docs[idx]
                    incorrect_cases.append(case_dict)
            
            # Log incorrect cases summary
            if incorrect_cases:
                logger.info(f"\n{'='*80}")
                logger.info(f"Found {len(incorrect_cases)} incorrect QA cases (out of {len(queries_solutions)} total)")
                logger.info(f"{'='*80}")
                
                # Log each incorrect case in detail
                for case in incorrect_cases:
                    logger.info(f"\n[Incorrect QA Case #{case['index']}]")
                    logger.info(f"Question: {case['question']}")
                    logger.info(f"Predicted Answer: {case['predicted_answer']}")
                    logger.info(f"Gold Answers: {case['gold_answers']}")
                    if 'gold_docs' in case:
                        logger.info(f"Gold Docs: {case['gold_docs']}")
                    logger.info(f"EM Score: {case['em_score']:.4f}, F1 Score: {case['f1_score']:.4f}, BLEU-1: {case['bleu1_score']:.4f}")
                    logger.info(f"{'-'*80}")
            else:
                logger.info(f"All {len(queries_solutions)} QA cases are correct (EM=1.0 and F1 >= {f1_threshold})")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata

    def retrieve_dpr(self,
                     queries: List[str],
                     num_to_retrieve: int = None,
                     gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using a DPR framework, which consists of several steps:
        - Dense passage scoring

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            logger.info('No facts found after reranking, return DPR results')
            logger.info(f"Query: {query}")
            sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)

            top_k_idx = sorted_doc_ids[:num_to_retrieve]
            top_k_chunk_ids = [self.passage_node_keys[idx] for idx in top_k_idx]
            top_k_docs = [self.chunk_embedding_store.get_row(cid)["content"] for cid in top_k_chunk_ids]

            episodic_memories = []
            if self.global_config.use_episodic_memory and self.episodic_memory_store is not None:
                # 支持多对一关系：通过chunk_ids获取所有对应的memories
                chunk_to_memories = self.episodic_memory_store.get_memories_by_chunk_ids(top_k_chunk_ids)
                # 收集所有记忆（去重）
                all_memory_ids = set()
                for chunk_id, memories in chunk_to_memories.items():
                    for memory in memories:
                        if memory.memory_id not in all_memory_ids:
                            all_memory_ids.add(memory.memory_id)
                            episodic_memories.append(memory)
                # 可以按相关性排序
                episodic_memories.sort(key=lambda m: len(set(m.chunk_ids) & set(top_k_chunk_ids)), reverse=True)

            retrieval_results.append(
                QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve], episodic_memories=episodic_memories if episodic_memories else None))

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(
                gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results],
                k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa_dpr(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using a standard DPR framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k、Exact Match、F1 与 BLEU-1 等指标。

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics（Exact Match、F1、BLEU-1 等）。

        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)
            qa_bleu1_evaluator = QABLEU1(global_config=self.global_config)
            qa_llm_judge_evaluator = QALLMJudge(global_config=self.global_config, llm_model=self.llm_model)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve_dpr(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve_dpr(queries=queries)

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_bleu1_result, example_qa_bleu1_results = qa_bleu1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            
            # LLM as a Judge evaluation
            questions = [qa_result.question for qa_result in queries_solutions]
            overall_qa_llm_judge_result, example_qa_llm_judge_results = qa_llm_judge_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, 
                predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                questions=questions,
                aggregation_fn=np.max)

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_em_result.update(overall_qa_bleu1_result)
            overall_qa_em_result.update(overall_qa_llm_judge_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata

    def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Executes question-answering (QA) inference using a provided set of query solutions and a language model.

        Parameters:
            queries: List[QuerySolution]
                A list of QuerySolution objects that contain the user queries, retrieved documents, and other related information.

        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict]]
                A tuple containing:
                - A list of updated QuerySolution objects with the predicted answers embedded in them.
                - A list of raw response messages from the language model.
                - A list of metadata dictionaries associated with the results.
        """
        #Running inference for QA
        all_qa_messages = []

        for query_solution in tqdm(queries, desc="Collecting QA prompts"):

            # obtain the retrieved docs
            retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]

            prompt_user = ''
            
            # Mode 1: Use episodic memories (when enabled)
            # Mode 2: Use original passages (when episodic memory is disabled)
            if self.global_config.use_episodic_memory:
                # MODE 1: Episodic memory mode - use passages as primary evidence, episodic memories/facts as secondary evidence
                # Pre-compute chunk IDs for the top qa_top_k passages (used by evidence backfill and fact time injection).
                top_k_chunk_ids: List[str] = []
                for doc in retrieved_passages:
                    try:
                        top_k_chunk_ids.append(self.chunk_embedding_store.get_hash_id(doc))
                    except KeyError:
                        # doc text not found in store mapping (should be rare); skip
                        continue
                    
                episodic_memories: List[EpisodicMemory] = []
                memory_prompt_section = ""
                try:
                    initial_episodic_memories = []
                    if self.episodic_memory_store is not None and top_k_chunk_ids:
                        # Step 1: Get memories strictly corresponding to the top qa_top_k chunks
                        chunk_to_memories = self.episodic_memory_store.get_memories_by_chunk_ids(top_k_chunk_ids)
                        
                        # Collect memories while preserving order of chunks (relevance)
                        seen_memory_ids = set()
                        for chunk_id in top_k_chunk_ids:
                            if chunk_id in chunk_to_memories:
                                for memory in chunk_to_memories[chunk_id]:
                                    if memory.memory_id not in seen_memory_ids:
                                        initial_episodic_memories.append(memory)
                                        seen_memory_ids.add(memory.memory_id)

                    if initial_episodic_memories:
                        episodic_memories = list(initial_episodic_memories)  # Start with initial memories
                        added_memory_ids = []
                        
                        # 记录使用的情境记忆信息
                        initial_chunk_ids_list = []
                        for memory in initial_episodic_memories:
                            initial_chunk_ids_list.extend(memory.chunk_ids)
                        logger.info(f"[QA Episodic Memory] Query: {query_solution.question}")
                        logger.info(f"[QA Episodic Memory] Initial {len(initial_episodic_memories)} episodic memories (qa_top_k={self.global_config.qa_top_k}) from chunks: {initial_chunk_ids_list}")
                        if added_memory_ids:
                            logger.info(f"[QA Episodic Memory] Single-hop expansion: Added {len(added_memory_ids)} related memories from memory_ids: {added_memory_ids}")
                        else:
                            logger.info(f"[QA Episodic Memory] Single-hop expansion: No additional related memories found")
                        logger.info(f"[QA Episodic Memory] Total {len(episodic_memories)} episodic memories used (initial: {len(initial_episodic_memories)}, single-hop: {len(added_memory_ids)})")
                        
                        # 记录每个情境记忆的详细信息
                        for i, memory in enumerate(episodic_memories, 1):
                            is_initial = i <= len(initial_episodic_memories)
                            memory_type = "Initial" if is_initial else "Single-hop"
                            chunk_ids_str = ', '.join(memory.chunk_ids) if memory.chunk_ids else 'N/A'
                            logger.info(f"[QA Episodic Memory] Memory {i}/{len(episodic_memories)} ({memory_type}) - Memory ID: {memory.memory_id}, Chunk IDs: {chunk_ids_str}, Summary: {memory.summary[:100]}...")
                            if hasattr(memory, 'related_memory_ids') and memory.related_memory_ids:
                                logger.info(f"[QA Episodic Memory]   Related memory IDs: {memory.related_memory_ids}")
                            elif hasattr(memory, 'related_chunk_ids') and memory.related_chunk_ids:
                                logger.info(f"[QA Episodic Memory]   Related chunk IDs: {memory.related_chunk_ids}")
                        
                        # 按timestamp排序情境记忆（如果时间戳对应文档顺序，这样可以保持时间顺序）
                        # 对于没有timestamp的记忆，将其放在最后
                        episodic_memories_sorted = sorted(
                            episodic_memories,
                            key=lambda m: m.timestamp if m.timestamp is not None else datetime.max
                        )
                        if episodic_memories_sorted != episodic_memories:
                            logger.info(f"[QA Episodic Memory] Sorted {len(episodic_memories)} episodic memories by timestamp")
                        episodic_memories = episodic_memories_sorted
                        
                        # Build episodic memory prompt section (secondary evidence; passages are primary).
                        memory_prompt_section += '\n--- Episodic Memory Summary ---\n'
                        memory_prompt_section += (
                            "The following episodic memories contain Structured Event Attributes extracted from the retrieved passages. "
                            "They can help you locate relevant information, but if they conflict with the Original Passages (Grounded Evidence), "
                            "trust the passages.\n"
                        )
                        for i, memory in enumerate(episodic_memories, 1):
                            memory_prompt_section += f'\n[Memory {i}]\n'
                            
                            # 按事件组织结构化要素，每个事件的要素对应在一起
                            if memory.events:
                                memory_prompt_section += 'Structured Event Details:\n'
                                for j, event in enumerate(memory.events, 1):
                                    if isinstance(event, dict):
                                        memory_prompt_section += f'  Event {j}:\n'
                                        
                                        # 优先使用新键，兼容旧数据
                                        participants = event.get('participants', event.get('who', []))
                                        if participants:
                                            participants_str = ', '.join(participants) if isinstance(participants, list) else str(participants)
                                            memory_prompt_section += f'    Participants: {participants_str}\n'
                                            
                                        action = event.get('action', event.get('what', []))
                                        if action:
                                            action_str = ', '.join(action) if isinstance(action, list) else str(action)
                                            memory_prompt_section += f'    Action: {action_str}\n'
                                            
                                        time = event.get('time') or event.get('when')
                                        if time:
                                            memory_prompt_section += f'    Time: {time}\n'
                                            
                                        location = event.get('location') or event.get('where')
                                        if location:
                                            memory_prompt_section += f'    Location: {location}\n'
                                            
                                        reason = event.get('reason') or event.get('why')
                                        if reason:
                                            memory_prompt_section += f'    Reason: {reason}\n'
                                            
                                        method = event.get('method') or event.get('how')
                                        if method:
                                            memory_prompt_section += f'    Method: {method}\n'
                                    else:
                                        # 向后兼容：如果事件不是字典，直接显示
                                        memory_prompt_section += f'  {j}. {event}\n'
                        memory_prompt_section += '\n--- End of Episodic Memory Summary ---\n\n'
                    else:
                        logger.info(f"[QA Episodic Memory] Query: {query_solution.question} - No episodic memories available")
                except Exception as e:
                    logger.warning(f"[QA Episodic Memory] Error processing episodic memories for query '{query_solution.question}': {str(e)}")
                
                # Grounded evidence (Original Passages): Scheme A
                # Expand evidence passages by backfilling chunks referenced by episodic memories.
                qa_backfill_max_additional = int(getattr(self.global_config, "qa_backfill_max_additional_chunks", 20))

                # Always prioritize retrieved chunks (keep their original order)
                expanded_chunk_ids: List[str] = []
                seen_chunk_ids: Set[str] = set()
                for cid in top_k_chunk_ids:
                    if cid not in seen_chunk_ids:
                        expanded_chunk_ids.append(cid)
                        seen_chunk_ids.add(cid)

                # Append backfilled chunk_ids from episodic memories in order, capped by qa_backfill_max_additional
                backfilled_chunk_ids: List[str] = []
                if 'episodic_memories' in locals() and episodic_memories:
                    for mem in episodic_memories:
                        if not getattr(mem, "chunk_ids", None):
                            continue
                        for cid in mem.chunk_ids:
                            if cid in seen_chunk_ids:
                                continue
                            if len(backfilled_chunk_ids) >= qa_backfill_max_additional:
                                break
                            expanded_chunk_ids.append(cid)
                            backfilled_chunk_ids.append(cid)
                            seen_chunk_ids.add(cid)
                        if len(backfilled_chunk_ids) >= qa_backfill_max_additional:
                            break

                # Convert expanded chunk ids back to passage texts (skip missing ids)
                expanded_evidence_passages: List[str] = []
                missing_in_store: List[str] = []
                for cid in expanded_chunk_ids:
                    try:
                        expanded_evidence_passages.append(self.chunk_embedding_store.get_row(cid)["content"])
                    except KeyError:
                        missing_in_store.append(cid)
                        continue

                if backfilled_chunk_ids or missing_in_store:
                    logger.info(
                        f"[QA Evidence Backfill] retrieved_chunks={len(top_k_chunk_ids)} "
                        f"memories_used={len(episodic_memories) if ('episodic_memories' in locals() and episodic_memories) else 0} "
                        f"expanded_chunks_total={len(expanded_chunk_ids)} backfilled={len(backfilled_chunk_ids)} "
                        f"missing_in_store={len(missing_in_store)} max_additional={qa_backfill_max_additional}"
                    )
                    if backfilled_chunk_ids:
                        logger.info(f"[QA Evidence Backfill] backfilled_chunk_ids={backfilled_chunk_ids[:50]}")

                evidence_passages = expanded_evidence_passages

                # Build evidence / memory / facts sections in a consistent order:
                # 1) Original Passages (Grounded Evidence) [primary]
                # 2) Episodic Memory Summary [secondary]
                # 3) Relevant Facts [secondary]
                evidence_prompt_section = ""
                if evidence_passages:
                    evidence_prompt_section += '--- Original Passages (Grounded Evidence) ---\n'
                    evidence_prompt_section += (
                        'The following passages are the grounded evidence retrieved for this question. '
                        'Prefer answering from these passages. If other evidence conflicts with them, trust these passages.\n'
                    )
                    for idx, passage in enumerate(evidence_passages, 1):
                        evidence_prompt_section += f'Passage {idx}: {passage}\n\n'
                    evidence_prompt_section += '--- End of Original Passages ---\n\n'
                    
                facts_prompt_section = ""
                # Add top-k facts to the prompt if available (secondary evidence; verify against passages first).
                if getattr(query_solution, 'top_k_facts', None) and query_solution.top_k_facts:
                    facts_prompt_section += '--- Relevant Facts ---\n'
                    facts_prompt_section += (
                        'The following facts were identified as highly relevant to the question through retrieval and reranking. '
                        'Use them as hints and verify them against the Original Passages (Grounded Evidence) first. '
                        'If passages are incomplete, you may additionally use Episodic Memory Summary when it does not conflict with the passages.\n'
                    )
                        
                    facts_with_time = 0
                    facts_without_time = 0
                    time_lookup_details = []
                    facts_added_to_prompt = []  # 记录最终加入prompt的fact内容

                    for i, fact in enumerate(query_solution.top_k_facts, 1):
                        if isinstance(fact, (tuple, list)) and len(fact) >= 3:
                            subject, predicate, obj = fact[0], fact[1], fact[2]
                            fact_tuple = (subject, predicate, obj)
                            fact_str = f'({subject}, {predicate}, {obj})'

                            # Time Injection Logic
                            time_source = None
                            try:
                                if hasattr(self, 'fact_embedding_store') and hasattr(self, 'fact_metadata'):
                                    fact_hash_id = self.fact_embedding_store.text_to_hash_id.get(str(fact_tuple))

                                    if fact_hash_id and fact_hash_id in self.fact_metadata:
                                        time_val = None
                                        # Priority 1: Use time from the retrieved chunks (top_k_chunk_ids)
                                        for cid in top_k_chunk_ids:
                                            if cid in self.fact_metadata[fact_hash_id]:
                                                time_val = self.fact_metadata[fact_hash_id][cid]
                                                time_source = f"chunk {cid} (retrieved)"
                                                break

                                        # Priority 2: Fallback to any available time for this fact
                                        if not time_val and self.fact_metadata[fact_hash_id]:
                                            first_chunk_id = next(iter(self.fact_metadata[fact_hash_id].keys()))
                                            time_val = self.fact_metadata[fact_hash_id][first_chunk_id]
                                            time_source = f"chunk {first_chunk_id} (fallback)"

                                        is_valid_time = (
                                            time_val is not None
                                            and time_val != ""
                                            and str(time_val).lower().strip() not in ("none", "null", "")
                                        )
                                        if is_valid_time:
                                            fact_str += f' [Time: {time_val}]'
                                            facts_with_time += 1
                                            time_lookup_details.append(f"Fact {i}: time='{time_val}' from {time_source}")
                                        else:
                                            facts_without_time += 1
                                            logger.debug(
                                                f"[QA Time] Fact {i} ({subject}, {predicate}, {obj}): Invalid time value '{time_val}' (null/empty/invalid)"
                                            )
                                    else:
                                        facts_without_time += 1
                                        if fact_hash_id:
                                            logger.debug(
                                                f"[QA Time] Fact {i} ({subject}, {predicate}, {obj}): fact_hash_id={fact_hash_id} not in fact_metadata"
                                            )
                                        else:
                                            logger.debug(
                                                f"[QA Time] Fact {i} ({subject}, {predicate}, {obj}): fact_hash_id not found in fact_embedding_store"
                                            )
                            except Exception as e:
                                facts_without_time += 1
                                logger.warning(f"[QA Time] Error injecting time for fact {fact}: {e}")

                        else:
                            fact_str = str(fact)
                            facts_without_time += 1

                        facts_added_to_prompt.append(f"Fact {i}: {fact_str}")
                        facts_prompt_section += f'Fact {i}: {fact_str}\n'

                    facts_prompt_section += '--- End of Relevant Facts ---\n\n'
                    logger.info(f"[QA Facts] Query: {query_solution.question} - Added {len(query_solution.top_k_facts)} facts to prompt")
                    logger.info(f"[QA Time] Time injection summary: {facts_with_time} facts with time, {facts_without_time} facts without time")
                    logger.info(f"[QA Facts] Facts added to prompt (final form):")
                    for fact_entry in facts_added_to_prompt:
                        logger.info(f"[QA Facts]   {fact_entry}")
                    if time_lookup_details:
                        logger.info(f"[QA Time] Facts with timestamps ({len(time_lookup_details)} total):")
                        for detail in time_lookup_details:
                            logger.info(f"[QA Time]   {detail}")
                else:
                    logger.info(f"[QA Facts] Query: {query_solution.question} - No facts available")

                # Assemble prompt in the desired order
                prompt_user += evidence_prompt_section
                prompt_user += memory_prompt_section
                prompt_user += facts_prompt_section
            else:
                # MODE 2: Original passages mode - use original retrieved passages when episodic memory is disabled
                logger.info(f"[QA Original Passages] Query: {query_solution.question} - Using original passages (episodic memory disabled)")
                for passage in retrieved_passages:
                    prompt_user += f'Wikipedia Title: {passage}\n\n'
            
            prompt_user += 'Question: ' + query_solution.question + '\n'
            prompt_user += 'Thought: '

            # Choose prompt template based on use_episodic_memory flag
            if self.global_config.use_episodic_memory:
                # Use episodic memory specific template
                if self.prompt_template_manager.is_template_name_valid(name='rag_qa_episodic'):
                    prompt_template_name = 'rag_qa_episodic'
                else:
                    logger.warning("rag_qa_episodic template not found, falling back to rag_qa_musique")
                    prompt_template_name = 'rag_qa_musique'
            else:
                # Use original passages template
                if self.prompt_template_manager.is_template_name_valid(name=f'rag_qa_{self.global_config.dataset}'):
                    # find the corresponding prompt for this dataset
                    prompt_template_name = f'rag_qa_{self.global_config.dataset}'
                else:
                    # the dataset does not have a customized prompt template yet
                    logger.debug(
                        f"rag_qa_{self.global_config.dataset} does not have a customized prompt template. Using MUSIQUE's prompt template instead.")
                    prompt_template_name = 'rag_qa_musique'
            all_qa_messages.append(
                self.prompt_template_manager.render(name=prompt_template_name, prompt_user=prompt_user))

        all_qa_results = []
        with ThreadPoolExecutor() as executor:
            results_generator = executor.map(self.llm_model.infer, all_qa_messages)
            all_qa_results = list(tqdm(results_generator, total=len(all_qa_messages), desc="QA Reading"))

        all_response_message, all_metadata, all_cache_hit = zip(*all_qa_results)
        all_response_message, all_metadata = list(all_response_message), list(all_metadata)

        #Process responses and extract predicted answers.
        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="Extraction Answers from LLM Response"):
            response_content = all_response_message[query_solution_idx]
            
            # Try to extract answer using multiple patterns
            pred_ans = None
            
            # Pattern 1: Standard "Answer:" format
            if 'Answer:' in response_content:
                parts = response_content.split('Answer:')
                if len(parts) > 1:
                    answer_part = parts[1].strip()
                    if answer_part:  # Only use if not empty
                        pred_ans = answer_part
            
            # Pattern 2: Try alternative formats if standard format failed
            if pred_ans is None or not pred_ans:
                # Try "答案:" (Chinese)
                if '答案:' in response_content:
                    parts = response_content.split('答案:')
                    if len(parts) > 1:
                        answer_part = parts[1].strip()
                        if answer_part:
                            pred_ans = answer_part
                
                # Try "The answer is" format
                if (pred_ans is None or not pred_ans) and 'The answer is' in response_content:
                    parts = response_content.split('The answer is')
                    if len(parts) > 1:
                        answer_part = parts[1].strip()
                        if answer_part:
                            pred_ans = answer_part
                
                # Try to extract from the last line if it looks like an answer
                if pred_ans is None or not pred_ans:
                    lines = response_content.strip().split('\n')
                    if lines:
                        last_line = lines[-1].strip()
                        # If last line is short and doesn't contain "Thought:" or "Answer:", use it
                        if last_line and len(last_line) < 200 and 'Thought:' not in last_line and 'Answer:' not in last_line:
                            pred_ans = last_line
            
            # Fallback: use entire response if no answer found
            if pred_ans is None or not pred_ans:
                logger.warning(f"Could not extract answer from response, using full response. Response length: {len(response_content)}")
                pred_ans = response_content.strip()

            # Save raw answer for optional dataset-specific postprocessing
            query_solution.raw_answer = pred_ans

            # LoCoMo answer postprocessing (only when locomo_metadata is attached)
            if getattr(query_solution, "locomo_metadata", None):
                try:
                    from .utils.locomo_qa_utils import postprocess_answer

                    query_solution.answer = postprocess_answer(pred_ans, query_solution.locomo_metadata)
                except Exception as e:
                    logger.warning(f"LoCoMo postprocess failed, using raw answer. Error: {e}")
                    query_solution.answer = pred_ans
            else:
                query_solution.answer = pred_ans
            queries_solutions.append(query_solution)

        return queries_solutions, all_response_message, all_metadata

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """
        Adds fact edges from given triples to the graph.

        The method processes chunks of triples, computes unique identifiers
        for entities and relations, and updates various internal statistics
        to build and maintain the graph structure. Entities are uniquely
        identified and linked based on their relationships.

        Parameters:
            chunk_ids: List[str]
                A list of unique identifiers for the chunks being processed.
            chunk_triples: List[Tuple]
                A list of tuples representing triples to process. Each triple
                consists of a subject, predicate, and object.

        Raises:
            Does not explicitly raise exceptions within the provided function logic.
        """

        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info(f"Adding OpenIE triples to graph.")

        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            entities_in_chunk = set()

            if chunk_key not in current_graph_nodes:
                for triple in triples:
                    triple = tuple(triple)

                    node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                    node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))

                    self.node_to_node_stats[(node_key, node_2_key)] = self.node_to_node_stats.get(
                        (node_key, node_2_key), 0.0) + 1
                    self.node_to_node_stats[(node_2_key, node_key)] = self.node_to_node_stats.get(
                        (node_2_key, node_key), 0.0) + 1

                    entities_in_chunk.add(node_key)
                    entities_in_chunk.add(node_2_key)

                for node in entities_in_chunk:
                    self.ent_node_to_chunk_ids[node] = self.ent_node_to_chunk_ids.get(node, set()).union(set([chunk_key]))

    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]):
        """
        Adds edges connecting passage nodes to phrase nodes in the graph.

        This method is responsible for iterating through a list of chunk identifiers
        and their corresponding triple entities. It calculates and adds new edges
        between the passage nodes (defined by the chunk identifiers) and the phrase
        nodes (defined by the computed unique hash IDs of triple entities). The method
        also updates the node-to-node statistics map and keeps count of newly added
        passage nodes.

        Parameters:
            chunk_ids : List[str]
                A list of identifiers representing passage nodes in the graph.
            chunk_triple_entities : List[List[str]]
                A list of lists where each sublist contains entities (strings) associated
                with the corresponding chunk in the chunk_ids list.

        Returns:
            int
                The number of new passage nodes added to the graph.
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.info(f"Connecting passage nodes to phrase nodes.")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):

            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    self.node_to_node_stats[(chunk_key, node_key)] = 1.0

                num_new_chunks += 1

        return num_new_chunks

    def add_synonymy_edges(self):
        """
        Adds synonymy edges between similar nodes in the graph to enhance connectivity by identifying and linking synonym entities.

        This method performs key operations to compute and add synonymy edges. It first retrieves embeddings for all nodes, then conducts
        a nearest neighbor (KNN) search to find similar nodes. These similar nodes are identified based on a score threshold, and edges
        are added to represent the synonym relationship.

        Attributes:
            entity_id_to_row: dict (populated within the function). Maps each entity ID to its corresponding row data, where rows
                              contain `content` of entities used for comparison.
            entity_embedding_store: Manages retrieval of texts and embeddings for all rows related to entities.
            global_config: Configuration object that defines parameters such as `synonymy_edge_topk`, `synonymy_edge_sim_threshold`,
                           `synonymy_edge_query_batch_size`, and `synonymy_edge_key_batch_size`.
            node_to_node_stats: dict. Stores scores for edges between nodes representing their relationship.

        """
        logger.info(f"Expanding graph with synonymy edges")

        self.entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)}).")

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        # Here we build synonymy edges only between newly inserted phrase nodes and all phrase nodes in the storage to reduce cost for incremental graph updates
        query_node_key2knn_node_keys = retrieve_knn(query_ids=entity_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=self.global_config.synonymy_edge_topk,
                                                    query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                                                    key_batch_size=self.global_config.synonymy_edge_key_batch_size)

        num_synonym_triple = 0
        synonym_candidates = []  # [(node key, [(synonym node key, corresponding score), ...]), ...]

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []

            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != '':
                        sim_edge = (node_key, nn)
                        synonyms.append((nn, score))
                        num_synonym_triple += 1

                        self.node_to_node_stats[sim_edge] = score  # Need to seriously discuss on this
                        num_nns += 1

            synonym_candidates.append((node_key, synonyms))

    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """
        Loads existing OpenIE results from the specified file if it exists and combines
        them with new content while standardizing indices. If the file does not exist or
        is configured to be re-initialized from scratch with the flag `force_openie_from_scratch`,
        it prepares new entries for processing.

        Args:
            chunk_keys (List[str]): A list of chunk keys that represent identifiers
                                     for the content to be processed.

        Returns:
            Tuple[List[dict], Set[str]]: A tuple where the first element is the existing OpenIE
                                         information (if any) loaded from the file, and the
                                         second element is a set of chunk keys that still need to
                                         be saved or processed.
        """

        # combine openie_results with contents already in file, if file exists
        chunk_keys_to_save = set()

        if not self.global_config.force_openie_from_scratch and os.path.isfile(self.openie_results_path):
            openie_results = json.load(open(self.openie_results_path))
            all_openie_info = openie_results.get('docs', [])

            #Standardizing indices for OpenIE Files.

            renamed_openie_info = []
            for openie_info in all_openie_info:
                openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
                renamed_openie_info.append(openie_info)

            all_openie_info = renamed_openie_info

            existing_openie_keys = set([info['idx'] for info in all_openie_info])

            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(self,
                             all_openie_info: List[dict],
                             chunks_to_save: Dict[str, dict],
                             ner_results_dict: Dict[str, NerRawOutput],
                             triple_results_dict: Dict[str, TripleRawOutput]) -> List[dict]:
        """
        Merges OpenIE extraction results with corresponding passage and metadata.

        This function integrates the OpenIE extraction results, including named-entity
        recognition (NER) entities and triples, with their respective text passages
        using the provided chunk keys. The resulting merged data is appended to
        the `all_openie_info` list containing dictionaries with combined and organized
        data for further processing or storage.

        Parameters:
            all_openie_info (List[dict]): A list to hold dictionaries of merged OpenIE
                results and metadata for all chunks.
            chunks_to_save (Dict[str, dict]): A dict of chunk identifiers (keys) to process
                and merge OpenIE results to dictionaries with `hash_id` and `content` keys.
            ner_results_dict (Dict[str, NerRawOutput]): A dictionary mapping chunk keys
                to their corresponding NER extraction results.
            triple_results_dict (Dict[str, TripleRawOutput]): A dictionary mapping chunk
                keys to their corresponding OpenIE triple extraction results.

        Returns:
            List[dict]: The `all_openie_info` list containing dictionaries with merged
            OpenIE results, metadata, and the passage content for each chunk.

        """

        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            try:
                chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                 'extracted_entities': ner_results_dict[chunk_key].unique_entities,
                                 'extracted_triples': triple_results_dict[chunk_key].triples}
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_key}: {e}")
                chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                 'extracted_entities': [],
                                 'extracted_triples': []}
            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """
        Computes statistics on extracted entities from OpenIE results and saves the aggregated data in a
        JSON file. The function calculates the average character and word lengths of the extracted entities
        and writes them along with the provided OpenIE information to a file.

        Parameters:
            all_openie_info : List[dict]
                List of dictionaries, where each dictionary represents information from OpenIE, including
                extracted entities.
        """

        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_openie_info])

        if len(all_openie_info) > 0:
            # Avoid division by zero if there are no phrases
            if num_phrases > 0:
                avg_ent_chars = round(sum_phrase_chars / num_phrases, 4)
                avg_ent_words = round(sum_phrase_words / num_phrases, 4)
            else:
                avg_ent_chars = 0
                avg_ent_words = 0
                
            openie_dict = {
                'docs': all_openie_info,
                'avg_ent_chars': avg_ent_chars,
                'avg_ent_words': avg_ent_words
            }
            
            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")

    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")
        print(self.get_graph_info())

    def add_new_nodes(self):
        """
        Adds new nodes to the graph from entity and passage embedding stores based on their attributes.

        This method identifies and adds new nodes to the graph by comparing existing nodes
        in the graph and nodes retrieved from the entity embedding store and the passage
        embedding store. The method checks attributes and ensures no duplicates are added.
        New nodes are prepared and added in bulk to optimize graph updates.
        """

        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_to_row = self.entity_embedding_store.get_all_id_to_rows()
        passage_to_row = self.chunk_embedding_store.get_all_id_to_rows()

        node_to_rows = entity_to_row
        node_to_rows.update(passage_to_row)

        new_nodes = {}
        for node_id, node in node_to_rows.items():
            node['name'] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def add_new_edges(self):
        """
        Processes edges from `node_to_node_stats` to add them into a graph object while
        managing adjacency lists, validating edges, and logging invalid edge cases.
        """

        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]: continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({
                "weight": weight
            })

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
        self.graph.add_edges(
            valid_edges,
            attributes=valid_weights
        )

    def save_igraph(self):
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        self.graph.write_pickle(self._graph_pickle_filename)
        logger.info(f"Saving graph completed!")

    def get_graph_info(self) -> Dict:
        """
        Obtains detailed information about the graph such as the number of nodes,
        triples, and their classifications.

        This method calculates various statistics about the graph based on the
        stores and node-to-node relationships, including counts of phrase and
        passage nodes, total nodes, extracted triples, triples involving passage
        nodes, synonymy triples, and total triples.

        Returns:
            Dict
                A dictionary containing the following keys and their respective values:
                - num_phrase_nodes: The number of unique phrase nodes.
                - num_passage_nodes: The number of unique passage nodes.
                - num_total_nodes: The total number of nodes (sum of phrase and passage nodes).
                - num_extracted_triples: The number of unique extracted triples.
                - num_triples_with_passage_node: The number of triples involving at least one
                  passage node.
                - num_synonymy_triples: The number of synonymy triples (distinct from extracted
                  triples and those with passage nodes).
                - num_total_triples: The total number of triples.
        """
        graph_info = {}

        # get # of phrase nodes
        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        # get # of passage nodes
        passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))

        # get # of total nodes
        graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"]

        # get # of extracted triples
        graph_info["num_extracted_triples"] = len(self.fact_embedding_store.get_all_ids())

        num_triples_with_passage_node = 0
        passage_nodes_set = set(passage_nodes_keys)
        num_triples_with_passage_node = sum(
            1 for node_pair in self.node_to_node_stats
            if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
        )
        graph_info['num_triples_with_passage_node'] = num_triples_with_passage_node

        graph_info['num_synonymy_triples'] = len(self.node_to_node_stats) - graph_info[
            "num_extracted_triples"] - num_triples_with_passage_node

        # get # of total triples
        graph_info["num_total_triples"] = len(self.node_to_node_stats)

        return graph_info

    def prepare_retrieval_objects(self):
        """
        Prepares various in-memory objects and attributes necessary for fast retrieval processes, such as embedding data and graph relationships, ensuring consistency
        and alignment with the underlying graph structure.
        """

        logger.info("Preparing for fast retrieval.")

        logger.info("Loading keys.")
        self.query_to_embedding: Dict = {'triple': {}, 'passage': {}}

        self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids()) # a list of phrase node keys
        self.passage_node_keys: List = list(self.chunk_embedding_store.get_all_ids()) # a list of passage node keys
        self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids())

        # Check if the graph has the expected number of nodes
        expected_node_count = len(self.entity_node_keys) + len(self.passage_node_keys)
        actual_node_count = self.graph.vcount()
        
        if expected_node_count != actual_node_count:
            logger.warning(f"Graph node count mismatch: expected {expected_node_count}, got {actual_node_count}")
            # If the graph is empty but we have nodes, we need to add them
            if actual_node_count == 0 and expected_node_count > 0:
                logger.info(f"Initializing graph with {expected_node_count} nodes")
                self.add_new_nodes()
                self.save_igraph()

        # Create mapping from node name to vertex index
        try:
            igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)} # from node key to the index in the backbone graph
            self.node_name_to_vertex_idx = igraph_name_to_idx
            
            # Check if all entity and passage nodes are in the graph
            missing_entity_nodes = [node_key for node_key in self.entity_node_keys if node_key not in igraph_name_to_idx]
            missing_passage_nodes = [node_key for node_key in self.passage_node_keys if node_key not in igraph_name_to_idx]
            
            if missing_entity_nodes or missing_passage_nodes:
                logger.warning(f"Missing nodes in graph: {len(missing_entity_nodes)} entity nodes, {len(missing_passage_nodes)} passage nodes")
                # If nodes are missing, rebuild the graph
                self.add_new_nodes()
                self.save_igraph()
                # Update the mapping
                igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
                self.node_name_to_vertex_idx = igraph_name_to_idx
            
            self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys] # a list of backbone graph node index
            self.passage_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.passage_node_keys] # a list of backbone passage node index
        except Exception as e:
            logger.error(f"Error creating node index mapping: {str(e)}")
            # Initialize with empty lists if mapping fails
            self.node_name_to_vertex_idx = {}
            self.entity_node_idxs = []
            self.passage_node_idxs = []

        logger.info("Loading embeddings.")
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))

        self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))

        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])

        self.proc_triples_to_docs = {}

        for doc in all_openie_info:
            triples = flatten_facts([doc['extracted_triples']])
            for triple in triples:
                if len(triple) == 3:
                    proc_triple = tuple(text_processing(list(triple)))
                    self.proc_triples_to_docs[str(proc_triple)] = self.proc_triples_to_docs.get(str(proc_triple), set()).union(set([doc['idx']]))

        if self.ent_node_to_chunk_ids is None:
            ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

            # Check if the lengths match
            if not (len(self.passage_node_keys) == len(ner_results_dict) == len(triple_results_dict)):
                logger.warning(f"Length mismatch: passage_node_keys={len(self.passage_node_keys)}, ner_results_dict={len(ner_results_dict)}, triple_results_dict={len(triple_results_dict)}")
                
                # If there are missing keys, create empty entries for them
                for chunk_id in self.passage_node_keys:
                    if chunk_id not in ner_results_dict:
                        ner_results_dict[chunk_id] = NerRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            unique_entities=[]
                        )
                    if chunk_id not in triple_results_dict:
                        triple_results_dict[chunk_id] = TripleRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            triples=[]
                        )

            # prepare data_store
            chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in self.passage_node_keys]

            self.node_to_node_stats = {}
            self.ent_node_to_chunk_ids = {}
            self.add_fact_edges(self.passage_node_keys, chunk_triples)

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping. The method determines whether each query
        is already present in the `self.query_to_embedding` dictionary under the keys 'triple' and 'passage'. If a query is not present in
        either, it is encoded into embeddings using the embedding model and stored.

        Args:
            queries List[str] | List[QuerySolution]: A list of query strings or QuerySolution objects. Each query is checked for
            its presence in the query-to-embedding mappings.
        """

        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['triple'] or query.question not in
                    self.query_to_embedding['passage']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            # get all query embeddings
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            query_embeddings_for_triple = self.embedding_model.batch_encode(all_query_strings,
                                                                            instruction=get_query_instruction('query_to_fact'),
                                                                            norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                             instruction=get_query_instruction('query_to_passage'),
                                                                             norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def get_fact_scores(self, query: str) -> np.ndarray:
        """
        Retrieves and computes normalized similarity scores between the given query and pre-stored fact embeddings.

        Parameters:
        query : str
            The input query text for which similarity scores with fact embeddings
            need to be computed.

        Returns:
        numpy.ndarray
            A normalized array of similarity scores between the query and fact
            embeddings. The shape of the array is determined by the number of
            facts.

        Raises:
        KeyError
            If no embedding is found for the provided query in the stored query
            embeddings dictionary.
        """
        query_embedding = self.query_to_embedding['triple'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_fact'),
                                                                norm=True)

        # Check if there are any facts
        if len(self.fact_embeddings) == 0:
            logger.warning("No facts available for scoring. Returning empty array.")
            return np.array([])
            
        try:
            query_fact_scores = np.dot(self.fact_embeddings, query_embedding.T) # shape: (#facts, )
            query_fact_scores = np.squeeze(query_fact_scores) if query_fact_scores.ndim == 2 else query_fact_scores
            query_fact_scores = min_max_normalize(query_fact_scores)
            return query_fact_scores
        except Exception as e:
            logger.error(f"Error computing fact scores: {str(e)}")
            return np.array([])

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.

        This function processes a given query using a pre-trained embedding model
        to generate query embeddings. The similarity scores between the query
        embedding and passage embeddings are computed using dot product, followed
        by score normalization. Finally, the function ranks the documents based
        on their similarity scores and returns the ranked document identifiers
        and their scores.

        Parameters
        ----------
        query : str
            The input query for which relevant passages should be retrieved.

        Returns
        -------
        tuple : Tuple[np.ndarray, np.ndarray]
            A tuple containing two elements:
            - A list of sorted document identifiers based on their relevance scores.
            - A numpy array of the normalized similarity scores for the corresponding
              documents.
        """
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_passage'),
                                                                norm=True)
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores


    def get_top_k_weights(self,
                          link_top_k: int,
                          all_phrase_weights: np.ndarray,
                          linking_score_map: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        This function filters the all_phrase_weights to retain only the weights for the
        top-ranked phrases in terms of the linking_score_map. It also filters linking scores
        to retain only the top `link_top_k` ranked nodes. Non-selected phrases in phrase
        weights are reset to a weight of 0.0.

        Args:
            link_top_k (int): Number of top-ranked nodes to retain in the linking score map.
            all_phrase_weights (np.ndarray): An array representing the phrase weights, indexed
                by phrase ID.
            linking_score_map (Dict[str, float]): A mapping of phrase content to its linking
                score, sorted in descending order of scores.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: A tuple containing the filtered array
            of all_phrase_weights with unselected weights set to 0.0, and the filtered
            linking_score_map containing only the top `link_top_k` phrases.
        """
        # choose top ranked nodes in linking_score_map
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # only keep the top_k phrases in all_phrase_weights
        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = set(
            [compute_mdhash_id(content=top_k_phrase, prefix="entity-") for top_k_phrase in top_k_phrases])

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        assert np.count_nonzero(all_phrase_weights) == len(linking_score_map.keys())
        return all_phrase_weights, linking_score_map

    def graph_search_with_fact_entities(self, query: str,
                                        link_top_k: int,
                                        query_fact_scores: np.ndarray,
                                        top_k_facts: List[Tuple],
                                        top_k_fact_indices: List[str],
                                        passage_node_weight: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes document scores based on fact-based similarity and relevance using personalized
        PageRank (PPR) and dense retrieval models. This function combines the signal from the relevant
        facts identified with passage similarity and graph-based search for enhanced result ranking.

        Parameters:
            query (str): The input query string for which similarity and relevance computations
                need to be performed.
            link_top_k (int): The number of top phrases to include from the linking score map for
                downstream processing.
            query_fact_scores (np.ndarray): An array of scores representing fact-query similarity
                for each of the provided facts.
            top_k_facts (List[Tuple]): A list of top-ranked facts, where each fact is represented
                as a tuple of its subject, predicate, and object.
            top_k_fact_indices (List[str]): Corresponding indices or identifiers for the top-ranked
                facts in the query_fact_scores array.
            passage_node_weight (float): Default weight to scale passage scores in the graph.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - The first array corresponds to document IDs sorted based on their scores.
                - The second array consists of the PPR scores associated with the sorted document IDs.
        """

        #Assigning phrase weights based on selected facts from previous steps.
        linking_score_map = {}  # from phrase to the average scores of the facts that contain the phrase
        phrase_scores = {}  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))
        number_of_occurs = np.zeros(len(self.graph.vs['name']))

        phrases_and_ids = set()

        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            fact_score = query_fact_scores[
                top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores

            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(
                    content=phrase,
                    prefix="entity-"
                )
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    weighted_fact_score = fact_score

                    if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        weighted_fact_score /= len(self.ent_node_to_chunk_ids[phrase_key])

                    phrase_weights[phrase_id] += weighted_fact_score
                    number_of_occurs[phrase_id] += 1

                phrases_and_ids.add((phrase, phrase_id))

        phrase_weights /= number_of_occurs

        for phrase, phrase_id in phrases_and_ids:
            if phrase not in phrase_scores:
                phrase_scores[phrase] = []

            phrase_scores[phrase].append(phrase_weights[phrase_id])

        # calculate average fact score for each phrase
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k,
                                                                           phrase_weights,
                                                                           linking_score_map)  # at this stage, the length of linking_scope_map is determined by link_top_k

        #Get passage scores according to chosen dense retrieval model
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)["content"]
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

        #Combining phrase and passage scores into one array for PPR
        node_weights = phrase_weights + passage_weights

        #Recording top 30 facts in linking_score_map
        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        assert sum(node_weights) > 0, f'No phrases found in the graph for the given facts: {top_k_facts}'

        #Running PPR algorithm based on the passage and phrase weights previously assigned
        ppr_start = time.time()
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=self.global_config.damping)
        ppr_end = time.time()

        self.ppr_time += (ppr_end - ppr_start)

        assert len(ppr_sorted_doc_ids) == len(
            self.passage_node_idxs), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores


    def rerank_facts(self, query: str, query_fact_scores: np.ndarray) -> Tuple[List[int], List[Tuple], dict]:
        """

        Args:

        Returns:
            top_k_fact_indicies:
            top_k_facts:
            rerank_log (dict): {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
                - candidate_facts (list): list of link_top_k facts (each fact is a relation triple in tuple data type).
                - top_k_facts:


        """
        # load args
        link_top_k: int = self.global_config.linking_top_k
        
        # Check if there are any facts to rerank
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}
            
        try:
            # Get the top k facts by score
            if len(query_fact_scores) <= link_top_k:
                # If we have fewer facts than requested, use all of them
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                # Otherwise get the top k
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
                
            # Get the actual fact IDs
            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            candidate_facts = [eval(fact_row_dict[id]['content']) for id in real_candidate_fact_ids]
            
            top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(query,
                                                                                candidate_facts,
                                                                                candidate_fact_indices,
                                                                                len_after_rerank=link_top_k)
            
            # 直接使用按分数排序的candidate_fact_indices和candidate_facts
            # top_k_fact_indices = candidate_fact_indices
            # top_k_facts = candidate_facts
            
            rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
            
            return top_k_fact_indices, top_k_facts, rerank_log
            
        except Exception as e:
            logger.error(f"Error in rerank_facts: {str(e)}")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': [], 'error': str(e)}
    
    def run_ppr(self,
                reset_prob: np.ndarray,
                damping: float =0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Personalized PageRank (PPR) on a graph and computes relevance scores for
        nodes corresponding to document passages. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        Parameters:
            reset_prob (np.ndarray): A 1-dimensional array specifying the reset
                probability distribution for each node. The array must have a size
                equal to the number of nodes in the graph. NaNs or negative values
                within the array are replaced with zeros.
            damping (float): A scalar specifying the damping factor for the
                computation. Defaults to 0.5 if not provided or set to `None`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The
                first array represents the sorted node IDs of document passages based
                on their relevance scores in descending order. The second array
                contains the corresponding relevance scores of each document passage
                in the same order.
        """

        if damping is None: damping = 0.5 # for potential compatibility
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores