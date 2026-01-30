import random

def split_val_file(input_file='val.txt', output_val='val.txt', output_test='test.txt'):
    """
    将val.txt中的文件名随机分成1:1比例的test.txt和val.txt
    
    参数:
        input_file: 输入文件名 (默认: 'val.txt')
        output_val: 输出验证集文件名 (默认: 'val.txt')
        output_test: 输出测试集文件名 (默认: 'test.txt')
    """
    # 1. 读取原始val.txt文件
    with open(input_file, 'r') as f:
        filenames = f.read().splitlines()
    
    # 2. 随机打乱文件名列表
    random.shuffle(filenames)
    
    # 3. 计算分割点（1:1比例）
    split_point = len(filenames) // 2
    
    # 4. 分割成两部分
    val_filenames = filenames[:split_point]
    test_filenames = filenames[split_point:]
    
    # 5. 写入新的val.txt和test.txt文件
    with open(output_val, 'w') as f:
        f.write('\n'.join(val_filenames))
    
    with open(output_test, 'w') as f:
        f.write('\n'.join(test_filenames))
    
    # 返回统计信息
    return {
        'total_samples': len(filenames),
        'val_samples': len(val_filenames),
        'test_samples': len(test_filenames)
    }

# 使用示例
if __name__ == "__main__":
    result = split_val_file()
    print(f"分割完成！总共 {result['total_samples']} 个样本")
    print(f"新验证集: {result['val_samples']} 个样本 (保存到 val.txt)")
    print(f"测试集: {result['test_samples']} 个样本 (保存到 test.txt)")