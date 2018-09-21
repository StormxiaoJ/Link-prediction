import math
import random
import csv
import sys
import gc
from tqdm import tqdm
# other users followed by particular user. key:userID,value: set of users
user_follow = {}
# other users who follow this user. key: userID,value: set of users.
follow_user = {}
# how many people this user follow
user_follow_num = {}
# how many people follow this user
follow_user_num = {}

# input train set
with open('train.txt', "r") as f:
    for line in f:
        piece = line.strip().split("\t")
        this_user = piece[0]
        who_follow_this_user = set()
        for i in range(1, len(piece)):
            who_follow_this_user.add(piece[i])
            follow_user.setdefault(piece[i], set(this_user)).add(this_user)
            follow_user_num[piece[i]] = follow_user_num.get(piece[i], 0) + 1
        if len(who_follow_this_user)==0:
            print("user {} has an empty list".format(this_user))
        else:
            user_follow[this_user] = who_follow_this_user
            user_follow_num[this_user] = len(who_follow_this_user)
print("train.txt is inputed")
# print("source list has {} samples. target list has {} samples. Intersection has {} samples.".
#       format(
#     len(user_follow),
#     len(follow_user),
#     len(set(user_follow.keys())&set(follow_user.keys()))))

# input test_set
test_set = {}
test_result=[]
with open('test-public.txt', "r") as f:
    # read header and pass
    line = f.readline()
    # read data
    line = f.readline().strip()
    while line:
        piece = line.split("\t")
        test_set[piece[1]] = piece[2]
        test_result.append((piece[1],piece[2]))
        line = f.readline().strip()
print("test-public.txt is inputed.")
# print("source in 20W is {}, source in 500W is {}, target in 20W is {}, target in 500W is {}".format(source_in_source_list,source_in_target_list,target_in_source_list,target_in_target_list))

"""
metric list:
CN:Common Neighbors
JC:Jaccard Coefficient
SI:Sørensen Index
SC:Salton Cosine Similarity
HP:Hub Promoted
HD:Hub Depressed
LHN:Leicht-Holme-Nerman
AA:Adamic-Adar Coefficient
PA:Preferential Attachment
"""
metrics = ['CN','JC','RA']
"""
mode list:
feature explanation: reference from https://arxiv.org/pdf/1411.5118.pdf
follow means source user follow others
follow by means source user followed by others
intersection: source user follow and target user follow by
reverse_intersection:  source user follow by and target user follow
"""
# modes = ['follow', 'follow_by', 'intersection', 'reverse_intersection']
modes = ['follow_by', 'intersection']
"""
feature list:
          'follow_CN','follow_by_CN','intersection_CN','reverse_intersection_CN'
          'follow_JC','follow_by_JC','intersection_JC','reverse_intersection_JC'
          'follow_SI', 'follow_by_SI','intersection_SI','reverse_intersection_SI'
          'follow_SC', 'follow_by_SC','intersection_SC','reverse_intersection_SC'
          'follow_HP', 'follow_by_HP','intersection_HP','reverse_intersection_HP'
          'follow_HD', 'follow_by_HD','intersection_HD','reverse_intersection_HD'
          'follow_LHN', 'follow_by_LHN','intersection_LHN','reverse_intersection_LHN'
          'follow_AA', 'follow_by_AA','intersection_AA','reverse_intersection_AA'
          'follow_PA', 'follow_by_PA','intersection_PA','reverse_intersection_PA'
"""
# note the sequence is not consistent with above comment
# creat header for features
features = ['source', 'target', 'exist',
            'user_follow_num_source', 'follow_user_num_source',
            'follow_user_num_target']
for mode in modes:
    for metric in metrics:
        features.append(mode + '_' + metric)
print("feature is created with dictionary: /n {}".format(features))


# calculate each feature for a sample pair.
def calculate_metric(mode, source=None, target=None):
    # if mode is 'follow':
    #     first_list = user_follow.get(source, set()) # safe
    #     second_list = user_follow.get(target, set())
    # elif mode is 'follow_by':
    #     first_list = follow_user.get(source, set()) # safe
    #     second_list = follow_user.get(target, set()) # safe
    # elif mode is 'intersection':
    #     first_list = user_follow.get(source, set()) # safe
    #     second_list = follow_user.get(target, set()) # safe
    # else:
    #     first_list = follow_user.get(source, set()) # safe
    #     second_list = user_follow.get(target, set())

    if mode is 'follow_by':
        first_list = follow_user.get(source, set())  # safe
        second_list = follow_user.get(target, set())  # safe
    elif mode is 'intersection':
        first_list = user_follow.get(source, set())  # safe
        second_list = follow_user.get(target, set())  # safe
    else:
        print('Error, unknown mode {}'.format(mode))
        sys.exit(0)
    intersection = first_list & second_list
    union = first_list | second_list
    PA = len(first_list) * len(second_list)
    CN = float(len(intersection))
    JC = CN / len(union)
    SI = CN / (len(first_list) + len(second_list))
    SC = CN / math.sqrt(len(first_list) * len(second_list))
    HP = CN / min(len(first_list), len(second_list))
    HD = CN / max(len(first_list), len(second_list))
    LHN = CN / PA
    CN = int(CN)
    RA = 0
    # calculate RA
    if mode is "follow_by":
        for i in intersection:
            if user_follow_num.get(i):
                RA += float(1) / user_follow_num.get(i)
    elif mode is "intersection":
        # note that source follow more user, less important this intermediary is
        #           more user follow i, less important this intermediary is
        #           i follow more user , less important this intermediary is
        #           more user follow target, less important this intermediary is
        for i in intersection:
            res = user_follow_num.get(source, 0) * \
                  follow_user_num.get(i, 0) * \
                  user_follow_num.get(i, 0) * \
                  follow_user_num.get(target, 0)
            if res != 0:
                RA += float(1) / res
    else:
        for i in intersection:
            # note that target follow more user, less important this intermediary is
            #           more user follow i, less important this intermediary is
            #           i follow more user , less important this intermediary is
            #           more user follow source, less important this intermediary is
            res = user_follow_num.get(target, 0) * \
                  follow_user_num.get(i, 0) * \
                  user_follow_num.get(i, 0) * \
                  follow_user_num.get(source, 0)
            if res != 0:
                RA += float(1) / res
    return [CN, JC, SI, SC, HP, HD, LHN, RA, PA]


# random generate train sample, no duplicate with test set. and generate features.
def generate_train_set():
    user_list = list(user_follow.keys())
    follow_list = list(follow_user.keys())
    with open("train_set.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=features)
        writer.writeheader()
        for i in tqdm(range(50000)):
            try:
                source = random.choice(user_list)
                target = random.choice(list(user_follow.get(source)))
                feature_for_this = {'source': source,
                                    'target': target,
                                    'exist': 1,
                                    'user_follow_num_source': user_follow_num.get(source),
                                    'follow_user_num_source': follow_user_num.get(source),
                                    'follow_user_num_target': follow_user_num.get(target)}
                for mode in modes:
                    for one_metrics in list(zip(calculate_metric(mode=mode, source=source, target=target), metrics)):
                        feature_for_this[mode + '_' + one_metrics[1]] = one_metrics[0]
                writer.writerow(feature_for_this)
            except IndexError:
                pass
        print('positive sample generated')
        gc.collect()
        for i in tqdm(range(50000)):
            try:
                source = random.choice(user_list)
                while (True):
                    target = random.choice(follow_list)
                    if target in (user_follow.get(source)) or test_set.get(source) == target:
                        continue
                    else:
                        break
                feature_for_this = {'source': source,
                                    'target': target,
                                    'exist': 0,
                                    'user_follow_num_source': user_follow_num.get(source),
                                    'follow_user_num_source': follow_user_num.get(source),
                                    'follow_user_num_target': follow_user_num.get(target)}
                for mode in modes:
                    for one_metrics in list(zip(calculate_metric(mode=mode, source=source, target=target), metrics)):
                        feature_for_this[mode + '_' + one_metrics[1]] = one_metrics[0]
                writer.writerow(feature_for_this)
            except IndexError:
                pass
        print('negative samples generated')
        gc.collect()
generate_train_set()

def generate_test_set():
    with open("test_set.csv", "w", newline="") as f:
        features.remove('exist')
        writer = csv.DictWriter(f, fieldnames=features)
        writer.writeheader()
        for test_sample in tqdm(test_result):
            feature_for_this = {'source': test_sample[0],
                                'target': test_sample[1],
                                'user_follow_num_source': user_follow_num.get(test_sample[0]),
                                'follow_user_num_source': follow_user_num.get(test_sample[0]),
                                'follow_user_num_target': follow_user_num.get(test_sample[1])}
            for mode in modes:
                for one_metrics in list(zip(calculate_metric(mode=mode, source=test_sample[0], target=test_sample[1]), metrics)):
                    feature_for_this[mode + '_' + one_metrics[1]] = one_metrics[0]
            writer.writerow(feature_for_this)
        print('testcsv is generated')
generate_test_set()
