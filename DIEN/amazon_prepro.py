import numpy as np
import pandas as pd
import random
import pickle

root = './data/private/Ad/amazon/'

with open(root+'Electronics_5.json') as fin:
    df = {}
    for i, line in enumerate(fin):
        df[i] = eval(line)
    reviews_df = pd.DataFrame.from_dict(df, orient='index')


with open(root+'np_prepro/reviews.pkl', 'wb') as f:
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)


with open(root+'meta_Electronics.json') as fin:
    df = {}
    for i, line in enumerate(fin):
        df[i] = eval(line)
    meta_df = pd.DataFrame.from_dict(df, orient='index')

meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
meta_df = meta_df.reset_index(drop=True)
with open(root+'np_prepro/meta.pkl', 'wb') as f:
    pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)


reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
meta_df = meta_df[['asin', 'categories']]
# only one category...
meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])


def build_map(df, col_name):
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key


asin_map, asin_key = build_map(meta_df, 'asin')
cate_map, cate_key = build_map(meta_df, 'categories')
revi_map, revi_key = build_map(reviews_df, 'reviewerID')


user_count, item_count, cate_count, example_count =\
    len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]
print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
      (user_count, item_count, cate_count, example_count))

meta_df = meta_df.sort_values('asin')
meta_df = meta_df.reset_index(drop=True)

reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
reviews_df = reviews_df.reset_index(drop=True)

cate_list = [meta_df['categories'][i] for i in range(len(asin_map))]
cate_list = np.array(cate_list, dtype=np.int32)

with open(root+'np_prepro/remap.pkl', 'wb') as f:
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL) # uid, iid
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL) # cid of iid line
    pickle.dump((user_count, item_count, cate_count, example_count),
              f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)

random.seed(1234)

train_set = []
test_set = []
for reviewerID, hist in reviews_df.groupby('reviewerID'):
    pos_list = hist['asin'].tolist()
    neg_list = []
    for _ in range(len(pos_list)):
        neg = pos_list[0]
        while neg in pos_list + neg_list:
            neg = random.randint(0, item_count - 1)
        neg_list.append(neg)

    for i in range(1, len(pos_list) - 1):
        hist = pos_list[:i]
        train_set.append((reviewerID, hist, pos_list[i], 1))
        train_set.append((reviewerID, hist, neg_list[i], 0))
    label = (pos_list[-1], neg_list[-1])
    test_set.append((reviewerID, hist, label))

random.shuffle(train_set)
random.shuffle(test_set)

assert len(test_set) == user_count

with open(root+'np_prepro/dataset.pkl', 'wb') as f:
    pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)


