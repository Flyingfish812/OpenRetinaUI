from torch.utils.data import DataLoader
        
def print_dataloader_info(dataloaders: dict[str, dict[str, DataLoader]]) -> None:
    """
    打印每个DataLoader中包含的样本数量和每个样本的尺寸信息
    
    Args:
        dataloaders: 嵌套字典结构，外层键为分割类型(train/validation/test)，
                    内层键为session_id，值为DataLoader对象
    """
    for split_name, session_loaders in dataloaders.items():
        print(f"Split: {split_name}")
        print("-" * 40)
        
        for session_id, dataloader in session_loaders.items():
            dataset = dataloader.dataset
            sample_count = len(dataset)
            
            # 获取第一个样本来确定数据形状
            first_sample = next(iter(dataset))
            
            # 处理DataPoint namedtuple结构
            if isinstance(first_sample, tuple) and hasattr(first_sample, '_fields'):
                inputs_shape = first_sample.inputs.shape
                targets_shape = first_sample.targets.shape
                
                print(f"  Session: {session_id}")
                print(f"    Samples: {sample_count}")
                print(f"    Input shape: {inputs_shape}")
                print(f"    Target shape: {targets_shape}")
                print()
            else:
                print(f"  Session: {session_id} - Unexpected data format")
                print()
        print()

