import codecs
import copy
import os
import random


def read_dataset(file_name):
    data = {}
    with open(file_name,"r",encoding="utf-8") as fr:
        for line in fr:
            user, item, score = line.strip().split("\t")
            if user not in data.keys():
                data[user] = [item]
            else :
                data[user].append(item)
    return data

def create_user_dict(user, data):
    user = copy.deepcopy(user)
    for u in data.keys():
        if u not in user.keys():
            user[u] = len(user)
    return user

def create_item_dict(item, data):
    for u in data.keys():
        for i in data[u]:
            if i not in item.keys():
                item[i] = len(item)
    return item


def ensure_dir(d, verbose=True):
    if not os.path.exists(d):
        if verbose:
            print("Directory {} do not exist; creating...".format(d))
        os.makedirs(d)

def generate_train_valid_test(file, total, common, user, item, choose):
    train_file = file + "train.txt"
    valid_file = file + "valid.txt"
    test_file = file + "test.txt"


    itemx = 0

    L = int(len(common) * 0.1)
    LL = L * 9
    LLL = L * 8

    train_number = 0
    valid_number = 0
    test_number = 0
    with codecs.open(train_file,"w",encoding="utf-8") as fw:
        with codecs.open(test_file, "w", encoding="utf-8") as fw2:
            with codecs.open(valid_file, "w", encoding="utf-8") as fw3:
                for da in total:
                    if choose == 0 and user[da] >= LL and user[da] < len(common):
                        for i in total[da]:
                            itemx = max(itemx, int(item[i]))
                            if random.randint(0,1):
                                test_number+=1
                                fw2.write(str(user[da]) + "\t" + str(item[i]) + "\n") # cold-start user for test
                            else :
                                valid_number+=1
                                fw3.write(str(user[da]) + "\t" + str(item[i]) + "\n")  # cold-start user for valid
                    elif choose == 1 and user[da] >= LLL and user[da] < LL:
                        for i in total[da]:
                            itemx = max(itemx, int(item[i]))
                            if random.randint(0, 1):
                                test_number += 1
                                fw2.write(str(user[da]) + "\t" + str(item[i]) + "\n")  # cold-start user for test
                            else:
                                valid_number += 1
                                fw3.write(str(user[da]) + "\t" + str(item[i]) + "\n")  # cold-start user for valid
                    else :
                        for i in total[da]:
                            train_number += 1
                            itemx = max(itemx, int(item[i]))
                            fw.write(str(user[da])+"\t"+str(item[i])+"\n")

    print("Train {}, valid {}, test {}".format(train_number, valid_number, test_number))
    print("sss", itemx)

if __name__ == '__main__':
    random.seed(42)
    # cloth sport
    # cell electronic
    # game video
    # cd movie
    # music instrument
    source = "cloth"
    target = "sport"
    f1 = source + "_" + target + "/"
    f2 = target + "_" + source + "/"

    source_common_data = read_dataset(f1 + "common_new_reindex.txt") # same users data
    target_common_data = read_dataset(f2 + "common_new_reindex.txt")

    user_dict = {} # re-index
    source_item = {}
    target_item = {}
    if len(source_common_data) == len(target_common_data):
        user_dict = create_user_dict(user_dict, source_common_data)
    else:
        print("error!!!!!!")
        exit(0)

    source_total_data = read_dataset(f1 + "new_reindex.txt")
    target_total_data = read_dataset(f2 + "new_reindex.txt")

    source_user = create_user_dict(user_dict, source_total_data) # re-index
    target_user = create_user_dict(user_dict, target_total_data)

    source_item = create_item_dict(source_item, source_total_data)
    target_item = create_item_dict(target_item, target_total_data)

    print(len(source_user))
    print(len(target_user))
    print(len(source_item))
    print(len(target_item))

    generate_train_valid_test(f1, source_total_data, source_common_data, source_user, source_item, 0)
    generate_train_valid_test(f2, target_total_data, target_common_data, target_user, target_item, 1)

