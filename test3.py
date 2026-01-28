import os
import glob


def replace_all_lines(file_path):
    # 定义替换规则
    replacement_map = {
        'handler1': '0',
        'handler2': '1',
        'handler3': '2',
        'handler4': '3',
        'handler5': '4',
        'handler6': '5',
        'handler7': '6'
    }

    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # 如果文件不为空
        if lines:
            modified = False
            new_lines = []

            # 处理每一行
            for line in lines:
                original_line = line
                stripped_line = line.strip()

                # 检查每一行是否以某个handler开头
                replaced = False
                for handler, replacement in replacement_map.items():
                    if stripped_line.startswith(handler):
                        # 替换行开头，保持其他内容不变
                        new_line = replacement + stripped_line[len(handler):] + '\n'
                        new_lines.append(new_line)
                        replaced = True
                        modified = True
                        break

                # 如果没有匹配的handler，保持原行不变
                if not replaced:
                    new_lines.append(line)

            # 如果进行了修改，则写回文件
            if modified:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.writelines(new_lines)
                print(f"已处理文件: {file_path}")
            else:
                print(f"无需处理: {file_path} - 没有找到匹配的handler")
        else:
            print(f"跳过空文件: {file_path}")

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")


def process_folder(folder_path):
    # 获取文件夹中所有的文本文件（.txt文件）
    text_files = glob.glob(os.path.join(folder_path, "*.txt"))

    if not text_files:
        print(f"在文件夹 {folder_path} 中未找到.txt文件")
        return

    print(f"找到 {len(text_files)} 个文本文件，开始处理...")

    # 处理每个文件
    for file_path in text_files:
        replace_all_lines(file_path)

    print("处理完成！")


# 使用示例
if __name__ == "__main__":
    folder_path = "C:/Users/Administrator/Desktop/ultralytics-main/ultralytics-main/datasets/hander/labels/train"

    # 检查文件夹是否存在
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        process_folder(folder_path)
    else:
        print("指定的文件夹路径不存在或不是文件夹")