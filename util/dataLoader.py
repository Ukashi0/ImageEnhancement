
class BaseDataLoader():
    def __init__(self):
        pass

    def initialize(self, opt):
        self.opt = opt
        pass
    # def load_data():
    #     return None


class dataLoader(BaseDataLoader):
    def name(self):
        return 'dataLoder'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        # 主要是对数据进行batch的划分
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,   # torch TensorDataset format
            batch_size=opt.batchSize,   # mini batch size
            shuffle=not opt.serial_batches,   # 要不要打乱数据
            num_workers=int(opt.nThreads))  # # 多线程来读数据



    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


def CreateDataLoader(opt):

    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
