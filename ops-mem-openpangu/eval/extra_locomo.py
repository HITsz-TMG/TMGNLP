import json
import os
import argparse

def merge_locomo_data(original_file_path, output_file_path, result_folder_prefix="locomo_"):
    try:
        with open(original_file_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        print(f"成功读取原始文件，共包含 {len(original_data)} 个对话样本。")
    except FileNotFoundError:
        print(f"错误：找不到文件 {original_file_path}")
        return

    final_results = []

    for i, original_sample in enumerate(original_data):
        
        result_file_path = os.path.join(f"{result_folder_prefix}{i}", "results.json")
        
        if not os.path.exists(result_file_path):
            print(f"警告：找不到文件 {result_file_path}，跳过该样本。")
            continue
            
        # 读取 result.json
        with open(result_file_path, 'r', encoding='utf-8') as f:
            result_list = json.load(f)
            
        sample_id = original_sample.get('sample_id', f"unknown-{i}")
        original_qa_list = original_sample.get('qa', [])

        for res_item in result_list:
            idx = res_item.get('idx')
            
            # 边界检查
            if idx is not None and 0 <= idx < len(original_qa_list):
                original_qa_item = original_qa_list[idx]
                
                # --- 关键步骤：验证问题是否一致 ---
                res_question = res_item.get('question', '').strip()
                orig_question = original_qa_item.get('question', '').strip()

                if res_question != orig_question:
                    print(f"[警告] ID {sample_id} - index {idx} 问题不匹配!")
                    print(f"  Result:   {res_question}")
                    print(f"  Original: {orig_question}")

                
                category = original_qa_item.get('category')
                
                unique_id = f"{sample_id}-{idx}"

                new_item = res_item.copy() 
                new_item['category'] = category
                new_item['id'] = unique_id
                
                final_results.append(new_item)
                
            else:
                print(f"错误：{result_file_path} 中的 idx {idx} 超出了原始 QA 列表的范围。")

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    
    print(f"处理完成！已保存 {len(final_results)} 条数据到 {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_dir', default='../data/locomo10.json')
    parser.add_argument('--result_dir', default='')
    parser.add_argument('--output', default='./result/locomo_extra.json')
    args = parser.parse_args()
    merge_locomo_data(args.ori_dir, args.output, args.result_dir)