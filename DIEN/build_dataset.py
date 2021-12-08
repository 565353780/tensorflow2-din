import random
import pickle
from tqdm import tqdm

'''
Settings
'''
pos_list_len_max = 100 # need to >= 2; if set to 1, will use all data
use_DIN_source_method = False

random.seed(1234)

with open('../datasets/raw_data/remap.pkl', 'rb') as f:
    reviews_df = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count, example_count = pickle.load(f)

train_set = []
test_set = []

for reviewerID, hist in tqdm(reviews_df.groupby('reviewerID')):
    pos_list = hist['asin'].tolist()
    def gen_neg():
        neg = pos_list[0]
        while neg in pos_list:
            neg = random.randint(0, item_count-1)
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
            if use_DIN_source_method:
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

save_dataset_name = 'dataset.pkl'
if pos_list_len_max > 1:
    save_dataset_name = 'dataset-' + str(pos_list_len_max) + '.pkl'
with open("../datasets/" + save_dataset_name, 'wb') as f:
    pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)

