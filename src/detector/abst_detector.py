from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import ttest_ind


class AbstDetector:
    def __init__(self, clean_loader, test_loader,  device):

        self.clean_loader = clean_loader
        self.test_loader = test_loader

        self.device = device

    def compute_score(self, model, bd_trigger):
        pass


    @staticmethod
    def resample_data_loader(dataloader, percent):
        ori_data_num = len(dataloader.dataset)
        new_num = int(ori_data_num * percent)
        batch_size = dataloader.batch_size
        new_num = (new_num // batch_size) * batch_size
        if new_num == 0:
            new_num = batch_size
        assert new_num % batch_size == 0
        new_dataset = dataloader.dataset.shuffle(seed=66).select(range(new_num))
        new_loader = DataLoader(new_dataset, batch_size=dataloader.batch_size, shuffle=False)
        return new_loader

    @staticmethod
    def get_class_num(model):
        NC = model.class_num
        return NC
    @staticmethod
    def conduct_p_test(list1, list2):
        """
        Conducts a t-test to determine if two lists of arrays are statistically different.

        Args:
            list1 (list of float): First list of arrays.
            list2 (list of float): Second list of arrays.

        Returns:
            dict: A dictionary containing the p-value and test statistic.
        """
        # Flatten the lists of arrays
        data1 = np.array(list1)
        data2 = np.array(list2)

        # Conduct the t-test
        stat, p_value = ttest_ind(data1, data2, equal_var=False)  # Welch's t-test

        return {"statistic": stat, "p_value": p_value}
