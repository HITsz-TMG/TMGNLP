import os
from typing import List
import json
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.misc_utils import string_to_bool
from src.hipporag.utils.config_utils import BaseConfig

import argparse

# os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '7'
os.environ["OPENAI_API_KEY"] = "sk-xx" 

import logging

def get_gold_docs(samples: List, dataset_name: str = None) -> List:
    gold_docs = []
    for sample in samples:
        if 'supporting_facts' in sample:  # hotpotqa, 2wikimultihopqa
            gold_title = set([item[0] for item in sample['supporting_facts']])
            gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
            if dataset_name.startswith('hotpotqa'):
                gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
        elif 'contexts' in sample:
            gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
        else:
            assert 'paragraphs' in sample, "`paragraphs` should be in sample, or consider the setting not to evaluate retrieval"
            gold_paragraphs = []
            for item in sample['paragraphs']:
                if 'is_supporting' in item and item['is_supporting'] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]

        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)
    return gold_docs


def get_gold_answers(samples):
    gold_answers = []
    for sample_idx in range(len(samples)):
        gold_ans = None
        sample = samples[sample_idx]

        if 'answer' in sample or 'gold_ans' in sample:
            gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        elif 'obj' in sample:
            gold_ans = set(
                [sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']])
            gold_ans = list(gold_ans)
        assert gold_ans is not None
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        assert isinstance(gold_ans, list)
        gold_ans = set(gold_ans)
        if 'answer_aliases' in sample:
            gold_ans.update(sample['answer_aliases'])

        gold_answers.append(gold_ans)

    return gold_answers

def main():
    parser = argparse.ArgumentParser(description="HippoRAG retrieval and QA with Episodic Memory + Reranker support")
    parser.add_argument('--dataset', type=str, default='musique', help='Dataset name')
    parser.add_argument('--llm_base_url', type=str, default='http://localhost:9000/v1', help='LLM base URL')
    parser.add_argument('--llm_name', type=str, default='pangu_embedded_7b', help='LLM name')
    parser.add_argument('--embedding_name', type=str, default='bge-m3', help='embedding model name')
    parser.add_argument('--force_index_from_scratch', type=str, default='false',
                        help='If set to True, will ignore all existing storage files and graph data and will rebuild from scratch.')
    parser.add_argument('--force_openie_from_scratch', type=str, default='false', help='If set to False, will try to first reuse openie results for the corpus if they exist.')
    parser.add_argument('--openie_mode', choices=['online', 'offline'], default='online',
                        help="OpenIE mode, offline denotes using VLLM offline batch mode for indexing, while online denotes")
    parser.add_argument('--save_dir', type=str, default='outputs_episodic', help='Save directory')

    # Reranker specific arguments
    parser.add_argument('--rerank_llm_base_url', type=str, default=None,
                        help='Base URL for the rerank LLM service (OpenAI-compatible /v1). If set, enables reranker.')
    parser.add_argument('--rerank_model_name', type=str, default='pangu_embedded_7b',
                        help='Rerank model name served by the rerank LLM service')

    # Episodic Memory specific arguments
    parser.add_argument('--generate_episodic_memory', type=str, default='true',
                        help='Whether to generate episodic memory for chunks during indexing (default: true)')
    parser.add_argument('--episodic_memory_llm_name', type=str, default=None,
                        help='LLM model name for episodic memory generation. If None, uses the main LLM.')
    parser.add_argument('--episodic_memory_batch_size', type=int, default=10,
                        help='Batch size for episodic memory generation (default: 10)')
    parser.add_argument('--episodic_memory_embedding_batch_size', type=int, default=32,
                        help='Batch size for computing episodic memory embeddings (default: 32)')
    parser.add_argument('--related_chunks_top_k', type=int, default=5,
                        help='Number of candidate chunks to pre-select using KNN for related chunk identification (default: 5)')
    parser.add_argument('--related_chunks_llm_filter', type=str, default='true',
                        help='Whether to use LLM filtering after KNN pre-selection for related chunk identification (default: true)')
    parser.add_argument('--related_chunks_strategy', type=str, default='embedding_llm',
                        choices=['embedding_llm', 'entity', 'episodic', 'hybrid'],
                        help='Strategy for finding related chunks (default: embedding_llm)')
    
    # Multi-chunk memory (多对一模式) specific arguments
    parser.add_argument('--enable_multi_chunk_memory', type=str, default='true',
                        help='Whether to enable multi-to-one episodic memory mode (N chunks -> 1 memory) (default: true)')
    parser.add_argument('--memory_integration_similarity_threshold', type=float, default=0.7,
                        help='Similarity threshold for memory integration (>= this threshold will be considered for integration) (default: 0.7)')
    parser.add_argument('--memory_integration_llm_filter', type=str, default='true',
                        help='Whether to use LLM filtering for integration judgment (default: true)')
    parser.add_argument('--memory_relation_similarity_min', type=float, default=0.6,
                        help='Minimum similarity threshold for memory relationships (default: 0.6)')
    parser.add_argument('--memory_relation_similarity_max', type=float, default=0.7,
                        help='Maximum similarity threshold for memory relationships (should be lower than integration threshold) (default: 0.7)')
    parser.add_argument('--memory_relation_max_per_memory', type=int, default=5,
                        help='Maximum number of related memories per memory (default: 5)')
    parser.add_argument('--memory_relation_llm_filter', type=str, default='true',
                        help='Whether to use LLM filtering for relationship judgment (default: true)')
    
    # OLD: 以下参数是为旧的 episodic_elements 结构化格式设计的，新格式使用自然语言事件列表，不再需要这些参数
    # parser.add_argument('--extract_when', type=str, default='true',
    #                     help='Whether to extract time information (when) in episodic elements (default: true)')
    # parser.add_argument('--extract_where', type=str, default='true',
    #                     help='Whether to extract location information (where) in episodic elements (default: true)')
    # parser.add_argument('--extract_why', type=str, default='true',
    #                     help='Whether to extract reason/motivation (why) in episodic elements (default: true)')
    # parser.add_argument('--extract_how', type=str, default='true',
    #                     help='Whether to extract method/manner (how) in episodic elements (default: true)')
    # parser.add_argument('--max_who_entities', type=int, default=10,
    #                     help='Maximum number of entities to extract in the "who" field of episodic elements (default: 10)')
    # parser.add_argument('--max_what_concepts', type=int, default=10,
    #                     help='Maximum number of concepts to extract in the "what" field of episodic elements (default: 10)')
    # NOTE: 新格式使用自然语言事件列表，LLM会自动在事件描述中包含可用的5W1H元素，不需要控制是否提取特定元素或限制数量
    
    parser.add_argument('--use_episodic_memory', type=str, default='true',
                        help='Whether to use episodic memory during retrieval and QA (default: true)')
    parser.add_argument('--qa_top_k', type=int, default=5, help='Number of top passages to use for QA')

    args = parser.parse_args()

    dataset_name = args.dataset
    save_dir = args.save_dir
    llm_base_url = args.llm_base_url
    llm_name = args.llm_name
    # if save_dir == 'outputs':
    #     save_dir = save_dir + '/' + dataset_name
    # else:
    #     save_dir = save_dir + '_' + dataset_name

    # 处理 longmemeval_s 数据集的特殊路径（文件在 longmemeval_s 子目录下）
    if dataset_name.startswith("longmemeval_s_"):
        corpus_path = f"longmemeval_s/{dataset_name}_corpus.json"
        dataset_path = f"longmemeval_s/{dataset_name}.json"
    elif dataset_name.startswith("locomo"):
        corpus_path = f"locomo/{dataset_name}_corpus.json"
        dataset_path = f"locomo/{dataset_name}.json"
    else:
        corpus_path = f"reproduce/dataset/{dataset_name}_corpus.json"
        dataset_path = f"reproduce/dataset/{dataset_name}.json"
    
    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

    force_index_from_scratch = string_to_bool(args.force_index_from_scratch)
    force_openie_from_scratch = string_to_bool(args.force_openie_from_scratch)

    # Prepare datasets and evaluation
    samples = json.load(open(dataset_path, "r"))
    all_queries = [s['question'] for s in samples]

    gold_answers = get_gold_answers(samples)
    try:
        gold_docs = get_gold_docs(samples, dataset_name)
        assert len(all_queries) == len(gold_docs) == len(gold_answers), "Length of queries, gold_docs, and gold_answers should be the same."
    except:
        gold_docs = None
    
    # LoCoMo 适配：当样本中包含 category 或 adversarial_answer 元数据时，传入 qa_items 以启用 LoCoMo 特性
    qa_items = samples if any(('category' in s) or ('adversarial_answer' in s) for s in samples) else None


    # Convert episodic memory related string arguments to boolean
    generate_episodic_memory = string_to_bool(args.generate_episodic_memory)
    use_episodic_memory = string_to_bool(args.use_episodic_memory)
    related_chunks_llm_filter = string_to_bool(args.related_chunks_llm_filter)
    enable_multi_chunk_memory = string_to_bool(args.enable_multi_chunk_memory)
    memory_integration_llm_filter = string_to_bool(args.memory_integration_llm_filter)
    memory_relation_llm_filter = string_to_bool(args.memory_relation_llm_filter)
    # OLD: 以下参数已废弃，新格式使用自然语言事件列表，不再需要这些控制参数
    # extract_when = string_to_bool(args.extract_when)
    # extract_where = string_to_bool(args.extract_where)
    # extract_why = string_to_bool(args.extract_why)
    # extract_how = string_to_bool(args.extract_how)

    config = BaseConfig(
        save_dir=save_dir,
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        dataset=dataset_name,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=force_index_from_scratch,
        force_openie_from_scratch=force_openie_from_scratch,
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=args.qa_top_k,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=8,
        max_new_tokens=None,
        corpus_len=len(corpus),
        openie_mode=args.openie_mode,
        # Episodic Memory configurations
        generate_episodic_memory=generate_episodic_memory,
        use_episodic_memory=use_episodic_memory,
        episodic_memory_llm_name=args.episodic_memory_llm_name,
        episodic_memory_batch_size=args.episodic_memory_batch_size,
        episodic_memory_embedding_batch_size=args.episodic_memory_embedding_batch_size,
        related_chunks_top_k=args.related_chunks_top_k,
        related_chunks_llm_filter=related_chunks_llm_filter,
        related_chunks_strategy=args.related_chunks_strategy,
        # Multi-chunk memory (多对一模式) configurations
        enable_multi_chunk_memory=enable_multi_chunk_memory,
        memory_integration_similarity_threshold=args.memory_integration_similarity_threshold,
        memory_integration_llm_filter=memory_integration_llm_filter,
        memory_relation_similarity_min=args.memory_relation_similarity_min,
        memory_relation_similarity_max=args.memory_relation_similarity_max,
        memory_relation_max_per_memory=args.memory_relation_max_per_memory,
        memory_relation_llm_filter=memory_relation_llm_filter,
        # OLD: 以下参数已废弃，新格式使用自然语言事件列表，不再需要这些控制参数
        # extract_when=extract_when,
        # extract_where=extract_where,
        # extract_why=extract_why,
        # extract_how=extract_how,
        # max_who_entities=args.max_who_entities,
        # max_what_concepts=args.max_what_concepts,
        # Reranker configurations
        rerank_llm_base_url=args.rerank_llm_base_url,
        rerank_model_name=args.rerank_model_name
    )

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"使用的模型名称: {llm_name}")
    logger.info(f"对应的调用URL地址: {llm_base_url}")
    if generate_episodic_memory:
        logger.info("=" * 80)
        logger.info("Episodic Memory is ENABLED")
        logger.info(f"  - Batch size: {args.episodic_memory_batch_size}")
        logger.info(f"  - Embedding batch size: {args.episodic_memory_embedding_batch_size}")
        logger.info(f"  - Related chunks top-k: {args.related_chunks_top_k}")
        logger.info(f"  - LLM filter: {related_chunks_llm_filter}")
        logger.info(f"  - Strategy: {args.related_chunks_strategy}")
        logger.info("")
        logger.info("Multi-chunk Memory (多对一模式) Configuration:")
        logger.info(f"  - Enable multi-chunk memory: {enable_multi_chunk_memory}")
        if enable_multi_chunk_memory:
            logger.info(f"  - Integration similarity threshold: {args.memory_integration_similarity_threshold}")
            logger.info(f"  - Integration LLM filter: {memory_integration_llm_filter}")
            logger.info(f"  - Relation similarity range: [{args.memory_relation_similarity_min}, {args.memory_relation_similarity_max})")
            logger.info(f"  - Relation max per memory: {args.memory_relation_max_per_memory}")
            logger.info(f"  - Relation LLM filter: {memory_relation_llm_filter}")
        logger.info("=" * 80)
    else:
        logger.info("Episodic Memory is DISABLED")

    if args.rerank_llm_base_url:
        logger.info("Reranker is ENABLED")
        logger.info(f"  - rerank_llm_base_url: {args.rerank_llm_base_url}")
        logger.info(f"  - rerank_model_name: {args.rerank_model_name}")
    else:
        logger.info("Reranker is DISABLED (using DSPy filter)")

    hipporag = HippoRAG(global_config=config)

    hipporag.index(docs)

    # Retrieval and QA
    solutions = hipporag.rag_qa(queries=all_queries, gold_docs=gold_docs, gold_answers=gold_answers, qa_items=qa_items)[0]
    for idx, q in enumerate(solutions):
        q.gold_answers = list(gold_answers[idx])
    result_list = []
    for idx, (q, solution) in enumerate(zip(all_queries, solutions)):
        result_list.append({
            "idx": idx,
            "question": q,
            "golden_answers": solution.gold_answers,
            "output": solution.answer
        })

    folder_path = save_dir
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, "results.json"), "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()


