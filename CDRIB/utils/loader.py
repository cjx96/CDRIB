"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np
import codecs

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, evaluation):
        self.batch_size = batch_size
        self.opt = opt
        self.eval = evaluation

        # ************* source data *****************
        source_train_data = "../dataset/" + filename + "/train.txt"
        source_valid_data = "../dataset/" + filename + "/valid.txt"
        source_test_data = "../dataset/" + filename + "/test.txt"

        self.source_ma_set, self.source_ma_list, self.source_train_data, self.source_user_set, self.source_item_set = self.read_train_data(source_train_data)
        if evaluation == -1:
            opt["source_user_num"] = max(self.source_user_set) + 1
            opt["source_item_num"] = max(self.source_item_set) + 1

        # ************* target data *****************
        filename = filename.split("_")
        filename = filename[1] + "_" + filename[0]
        target_train_data = "../dataset/" + filename + "/train.txt"
        target_valid_data = "../dataset/" + filename + "/valid.txt"
        target_test_data = "../dataset/" + filename + "/test.txt"
        self.target_ma_set, self.target_ma_list, self.target_train_data, self.target_user_set, self.target_item_set = self.read_train_data(
            target_train_data)
        if evaluation == -1:
            opt["target_user_num"] = max(self.target_user_set) + 1
            opt["target_item_num"] = max(self.target_item_set) + 1

        if evaluation == 1:
            self.test_data = self.read_test_data(source_test_data, self.source_item_set)
        elif evaluation == 2:
            self.test_data = self.read_test_data(target_test_data, self.target_item_set)


        if evaluation == 3:
            self.test_data = self.read_test_data(source_valid_data, self.source_item_set)
        elif evaluation == 4:
            self.test_data = self.read_test_data(target_valid_data, self.target_item_set)


        # assert opt["source_user_num"] == opt["target_user_num"]
        if evaluation < 0:
            data = self.preprocess()
        else :
            data = self.preprocess_for_predict()
        # shuffle for training
        if evaluation == -1:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
            if batch_size > len(data):
                batch_size = len(data)
                self.batch_size = batch_size
            if len(data)%batch_size != 0:
                data += data[:batch_size]
            data = data[: (len(data)//batch_size) * batch_size]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def read_train_data(self, train_file):
        with codecs.open(train_file, "r", encoding="utf-8") as infile:
            train_data = []
            user_set = set()
            item_set = set()
            ma = {}
            ma_list = {}
            for line in infile:
                line=line.strip().split("\t")
                user = int(line[0])
                item = int(line[1])
                train_data.append([user, item])
                if user not in ma.keys():
                    ma[user] = set()
                    ma_list[user] = []
                ma[user].add(item)
                ma_list[user].append(item)
                user_set.add(user)
                item_set.add(item)
        return ma, ma_list, train_data, user_set, item_set

    def read_test_data(self, test_file, item_set):
        user_item_set = {}
        self.MIN_USER = 10000000
        self.MAX_USER = 0
        with codecs.open(test_file, "r", encoding="utf-8") as infile:
            for line in infile:
                line=line.strip().split("\t")
                user = int(line[0])
                item = int(line[1])
                if user not in user_item_set:
                    user_item_set[user] = set()
                user_item_set[user].add(item)
                self.MIN_USER = min(self.MIN_USER, user)
                self.MAX_USER = max(self.MAX_USER, user)

        with codecs.open(test_file, "r", encoding="utf-8") as infile:
            test_data = []
            item_list = sorted(list(item_set))
            cnt = 0
            for line in infile:
                line=line.strip().split("\t")
                user = int(line[0])
                item = int(line[1])
                if item in item_set:
                    ret = [item]
                    for i in range(self.opt["test_sample_number"]):
                        while True:
                            rand = item_list[random.randint(0, len(item_set) - 1)]
                            if self.eval == 1:
                                if rand in user_item_set[user]:
                                    continue
                            ret.append(rand)
                            break
                    test_data.append([user, ret])
                else :
                    cnt += 1
            print("un test:", cnt)
            print("test length:", len(test_data))
        return test_data

    def preprocess_for_predict(self):
        processed=[]
        for d in self.test_data:
            processed.append([d[0],d[1]])
        return processed
    def preprocess(self):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in self.source_train_data:
            d = [d[1], d[0]]
            processed.append(d + [-1]) # i u -1
        for d in self.target_train_data:
            processed.append([-1] + d) # -1 u i
        return processed

    def find_pos(self,ma_list, user):
        rand = random.randint(0, 1000000)
        rand %= len(ma_list[user])
        return ma_list[user][rand]

    def find_neg(self, ma_set, user, type):
        n = 5
        while n:
            n -= 1
            rand = random.randint(0, self.opt[type] - 1)
            if rand not in ma_set[user]:
                return rand
        return rand

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        if self.eval!=-1:
            batch = list(zip(*batch))
            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]))

        else :
            source_user = []
            target_user = []
            source_neg_tmp = []
            target_neg_tmp = []
            source_pos_tmp = []
            target_pos_tmp = []
            for b in batch:
                if b[0] == -1: # -1 u i
                    target_user.append(b[1])
                    target_pos_tmp.append(b[2])
                    target_neg_tmp.append(self.find_neg(self.target_ma_set, b[1], "target_item_num"))
                else: # i u -1
                    source_user.append(b[1])
                    source_pos_tmp.append(b[0])
                    source_neg_tmp.append(self.find_neg(self.source_ma_set, b[1], "source_item_num"))
            return (torch.LongTensor(source_user), torch.LongTensor(source_pos_tmp), torch.LongTensor(source_neg_tmp), torch.LongTensor(target_user), torch.LongTensor(target_pos_tmp), torch.LongTensor(target_neg_tmp))
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)