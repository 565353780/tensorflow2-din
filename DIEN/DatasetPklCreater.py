import os
import random
import pickle
from tqdm import tqdm

class DatasetPklCreater:
    def __init__(self):
        random.seed(1234)

        self.reviews_df = None
        self.cate_list = None
        self.user_count = None
        self.item_count = None
        self.cate_count = None
        self.example_count = None
        return

    def load_remap_pkl(self):
        with open('../datasets/raw_data/remap.pkl', 'rb') as f:
            self.reviews_df = pickle.load(f)
            self.cate_list = pickle.load(f)
            self.user_count, self.item_count, self.cate_count, self.example_count = pickle.load(f)
        return True

    def get_dataset_name(self, pos_list_len_max, use_din_source_method):
        save_dataset_name = 'dataset_new_method.pkl'
        if use_din_source_method:
            save_dataset_name = 'dataset_source_method.pkl'
        if pos_list_len_max > 1:
            save_dataset_name = 'dataset_' + str(pos_list_len_max) + '_new_method.pkl'
            if use_din_source_method:
                save_dataset_name = 'dataset_' + str(pos_list_len_max) + '_source_method.pkl'
        return save_dataset_name;

    def is_dataset_exists(self, pos_list_len_max, use_din_source_method):
        save_dataset_name = self.get_dataset_name(pos_list_len_max, use_din_source_method)
        if os.path.exists("../datasets/" + save_dataset_name):
            return True
        return False

    def create_dataset_pkl(self, pos_list_len_max, use_din_source_method):
        if self.is_dataset_exists(pos_list_len_max, use_din_source_method):
            return True

        train_set = []
        test_set = []

        for reviewerID, hist in tqdm(self.reviews_df.groupby('reviewerID')):
            pos_list = hist['asin'].tolist()
            def gen_neg():
                neg = pos_list[0]
                while neg in pos_list:
                    neg = random.randint(0, self.item_count-1)
                return neg
            neg_list = [gen_neg() for i in range(len(pos_list))]

            valid_pos_list_len = len(pos_list)
            if pos_list_len_max > 1:
                valid_pos_list_len = min(valid_pos_list_len, pos_list_len_max)

            for i in range(1, valid_pos_list_len):
                hist = pos_list[:i]
                if i != valid_pos_list_len - 1:
                    train_set.append((reviewerID, hist, pos_list[i], 1))
                    train_set.append((reviewerID, hist, neg_list[i], 0))
                else:
                    if use_din_source_method:
                        # DIN source method
                        label = (pos_list[i], neg_list[i])
                        test_set.append((reviewerID, hist, label))
                    else:
                        # din-tf2 new method
                        test_set.append((reviewerID, hist, pos_list[i], 1))
                        test_set.append((reviewerID, hist, neg_list[i], 0))

        random.shuffle(train_set)
        random.shuffle(test_set)

        # assert len(test_set) == user_count
        # assert(len(test_set) + len(train_set) // 2 == reviews_df.shape[0])

        save_dataset_name = self.get_dataset_name(pos_list_len_max, use_din_source_method)
        with open("../datasets/" + save_dataset_name, 'wb') as f:
            pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.cate_list, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump((self.user_count, self.item_count, self.cate_count), f, pickle.HIGHEST_PROTOCOL)
        return True

if __name__ == "__main__":
    dataset_pkl_creater = DatasetPklCreater()
    dataset_pkl_creater.load_remap_pkl()
    dataset_pkl_creater.create_dataset_pkl(100, True)

