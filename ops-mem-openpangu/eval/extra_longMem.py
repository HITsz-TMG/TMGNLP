import json
import os
import argparse
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description="合并原始数据集与模型输出结果")
    
    # 1. 原始数据集目录 (包含 longmemeval_s_i.json)
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default='longmemeval_s', 
        help='原始数据集存放目录 (默认: longmemeval_s)'
    )
    
    # 2. 模型输出目录 (包含 longmemeval_s_i/results.json)
    parser.add_argument(
        '--result_dir', 
        type=str, 
        default='outputs_longmemeval_s', 
        help='模型输出结果存放目录 (默认: outputs_longmemeval_s)'
    )
    
    # 3. 最终保存路径
    parser.add_argument(
        '--save_path', 
        type=str, 
        default='results/merged_longmemeval_s.json', 
        help='合并后文件的保存路径 (默认: results/merged_longmemeval_s.json)'
    )
    
    # 4. 数据索引范围 - 开始
    parser.add_argument(
        '--start', 
        type=int, 
        default=0, 
        help='数据处理起始索引 (默认: 0)'
    )
    
    # 5. 数据索引范围 - 结束
    parser.add_argument(
        '--end', 
        type=int, 
        default=500, 
        help='数据处理结束索引 (不包含, 默认: 500)'
    )
    
    # 6. 文件名前缀 (可选，以防文件名变更)
    parser.add_argument(
        '--prefix', 
        type=str, 
        default='longmemeval_s', 
        help='文件名前缀 (默认: longmemeval_s)'
    )

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # 提取参数
    src_data_dir = args.data_dir
    src_output_dir = args.result_dir
    target_path = args.save_path
    start_idx = args.start
    end_idx = args.end
    file_prefix = args.prefix
    
    # 确保输出目录存在
    target_dir = os.path.dirname(target_path)
    if target_dir and not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建目录: {target_dir}")

    final_results = []
    success_count = 0
    missing_count = 0
        
    for i in range(start_idx, end_idx):
        # 构造动态文件名
        # 原始文件: {data_dir}/{prefix}_{i}.json
        filename_src = f"{file_prefix}_{i}.json"
        file_path_src = os.path.join(src_data_dir, filename_src)
        
        # 结果文件: {result_dir}/{prefix}_{i}/results.json
        # 注意：这里假设结果文件夹名也是 {prefix}_{i}
        folder_name_res = f"{file_prefix}_{i}"
        file_path_out = os.path.join(src_output_dir, folder_name_res, "results.json")
        
        # 检查文件是否存在
        if not os.path.exists(file_path_src):
            missing_count += 1
            continue
            
        if not os.path.exists(file_path_out):
            missing_count += 1
            continue

        try:
            # 读取原始数据
            with open(file_path_src, 'r', encoding='utf-8') as f:
                data_src_list = json.load(f)
                if not data_src_list: continue
                item_src = data_src_list[0]

            # 读取结果数据
            with open(file_path_out, 'r', encoding='utf-8') as f:
                data_out_list = json.load(f)
                if not data_out_list: continue
                item_out = data_out_list[0]

            # 合并逻辑
            merged_entry = {
                "idx": item_out.get("idx"),
                "question": item_out.get("question"),
                "golden_answers": item_out.get("golden_answers"),
                "output": item_out.get("output"),
                # 映射 category 取 question_type
                "category": item_src.get("question_type"),
                # 映射 id 取 question_id
                "id": item_src.get("question_id")
            }

            final_results.append(merged_entry)
            success_count += 1

        except Exception as e:
            print(f"索引 {i} 处理出错: {e}")

    # 保存结果
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    print("-" * 30)
    print(f"处理结束。")
    print(f"成功合并: {success_count} 条")
    print(f"结果已保存至: {target_path}")

if __name__ == "__main__":
    main()