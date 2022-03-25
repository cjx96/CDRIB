import pandas as pd
import os


def pprint(str_, f):
    print(str_)
    print(str_, end='\n', file=f)


def filter_data(filePath):
    data = []
    ratings = pd.read_csv(filePath, delimiter=",", encoding="latin1")
    ratings.columns = ['userId', 'itemId', 'Rating', 'timesteamp']

    rate_size_dic_i = ratings.groupby('itemId').size()
    # choosed_index_del_i = rate_size_dic_i.index[rate_size_dic_i < 10]
    choosed_index_del_i = rate_size_dic_i.index[rate_size_dic_i < 10]
    ratings = ratings[~ratings['itemId'].isin(list(choosed_index_del_i))] # item freq more than 10

    user_unique = list(ratings['userId'].unique())
    movie_unique = list(ratings['itemId'].unique())

    u = len(user_unique)
    i = len(movie_unique)
    rating_num = len(ratings)
    return u, i, rating_num, user_unique, ratings


def get_min_group_size(ratings):
    rate_size_dic_u = ratings.groupby('userId').size()
    return min(rate_size_dic_u)


def my_reindex_data(ratings1, dic_u=None):
    data = []
    if dic_u is None:
        user_unique = list(ratings1['userId'].unique())
        user_index = list(range(0, len(user_unique)))
        dic_u = dict(zip(user_unique, user_index))
    movie_unique1 = list(ratings1['itemId'].unique())
    movie_index1 = list(range(0, len(movie_unique1)))
    dic_m1 = dict(zip(movie_unique1, movie_index1))
    for element in ratings1.values:
        data.append((dic_u[element[0]], dic_m1[element[1]], 1))
    data = sorted(data, key=lambda x: x[0])
    return data, dic_u

def reindex_data(ratings1, dic_u=None,dic_v=None):
    data = []
    if dic_u is None:
        user_unique = list(ratings1['userId'].unique())
        user_index = list(range(0, len(user_unique)))
        dic_u = dict(zip(user_unique, user_index))
    if dic_v is None:
        movie_unique1 = list(ratings1['itemId'].unique())
        movie_index1 = list(range(0, len(movie_unique1)))
        dic_v = dict(zip(movie_unique1, movie_index1))
    for element in ratings1.values:
        data.append((dic_u[element[0]], dic_v[element[1]], 1))
    data = sorted(data, key=lambda x: x[0])
    return data, dic_u


def get_common_data(data1, data2, common_user):
    rating_new_1 = data1[data1['userId'].isin(common_user)]
    rating_new_2 = data2[data2['userId'].isin(common_user)]
    return rating_new_1, rating_new_2


def get_unique_lenth(ratings):
    r_n = len(ratings)
    user_unique = list(ratings['userId'].unique())
    movie_unique = list(ratings['itemId'].unique())
    u = len(user_unique)
    i = len(movie_unique)
    return u, i, r_n


def filter_user(ratings1, ratings2):
    rate_size_dic_u1 = ratings1.groupby('userId').size()
    rate_size_dic_u2 = ratings2.groupby('userId').size()
    choosed_index_del_u1 = rate_size_dic_u1.index[rate_size_dic_u1 < 5]
    choosed_index_del_u2 = rate_size_dic_u2.index[rate_size_dic_u2 < 5]
    ratings1 = ratings1[~ratings1['userId'].isin(list(choosed_index_del_u1) + list(choosed_index_del_u2))]
    ratings2 = ratings2[~ratings2['userId'].isin(list(choosed_index_del_u1) + list(choosed_index_del_u2))]
    return ratings1, ratings2

def filter_item(ratings1, ratings2):
    rate_size_dic_u1 = ratings1.groupby('itemId').size()
    rate_size_dic_u2 = ratings2.groupby('itemId').size()
    choosed_index_del_u1 = rate_size_dic_u1.index[rate_size_dic_u1 < 5]
    choosed_index_del_u2 = rate_size_dic_u2.index[rate_size_dic_u2 < 5]
    ratings1 = ratings1[~ratings1['itemId'].isin(list(choosed_index_del_u1) + list(choosed_index_del_u2))]
    ratings2 = ratings2[~ratings2['itemId'].isin(list(choosed_index_del_u1) + list(choosed_index_del_u2))]
    return ratings1, ratings2


def write_to_txt(data, file):
    f = open(file, 'w+')
    for i in data:
        line = '\t'.join([str(x) for x in i]) + '\n'
        f.write(line)
    f.close


def get_common_user(data1, data2):
    common_user = list(set(data1).intersection(set(data2)))
    return len(common_user), common_user

def get_total_user(data1, data2):
    total_user = list(set(data1).union(set(data2)))
    return len(total_user), total_user


# cloth sport
# cell electronic
# game video
# cd movie
# music instrument


# datapath = './Amazon/'
datapath = './rating_dataset/'
data_name_s = 'music'
data_name_t = 'instrument'
save_path = 'generate_dataset/'

save_path_s = save_path + data_name_s + '_' + data_name_t + '/'
save_path_t = save_path + data_name_t + '_' + data_name_s + '/'
if not os.path.exists(save_path_s):
    os.makedirs(save_path_s)
if not os.path.exists(save_path_t):
    os.makedirs(save_path_t)
data_dic = {'sport': 'ratings_Sports_and_Outdoors', 'electronic': 'ratings_Electronics',
            'cloth': 'ratings_Clothing_Shoes_and_Jewelry', 'cell': 'ratings_Cell_Phones_and_Accessories',
            'instrument': 'ratings_Musical_Instruments', 'game': 'ratings_Toys_and_Games', 'video': 'ratings_Video_Games',
            'book':'ratings_Books', 'movie':'ratings_Movies_and_TV', 'cd': 'ratings_CDs_and_Vinyl', 'music':'ratings_Digital_Music'}


filepath1 = datapath + data_dic[data_name_s] + '.csv'
filepath2 = datapath + data_dic[data_name_t] + '.csv'
save_file1 = save_path_s + 'common_new_reindex.txt'
save_file2 = save_path_t + 'common_new_reindex.txt'
save_file3 = save_path_s + 'new_reindex.txt'
save_file4 = save_path_t + 'new_reindex.txt'
f_path = save_path_t + '%s_%s_data_info.txt' % (data_name_s, data_name_t)
f = open(f_path, 'w+')
u_num, i_num, r_num, user_unique, data = filter_data(filepath1) # item<10
u_num2, i_num2, r_num2, user_unique2, data2 = filter_data(filepath2) # item<10
# nn_data_1 = filter_item(new_data_1)
# nn_data_2 = filter_item(new_data_2)
# u,i ,r= get_unique_lenth(nn_data_1)
# u2,i2 ,r2= get_unique_lenth(nn_data_2)


data, data2 = filter_user(data, data2) # del overlap user < 5
data, data2 = filter_item(data, data2) # del overlap user < 5
user_unique = list(data['userId'].unique())
user_unique2 = list(data2['userId'].unique())
item_unique = list(data['itemId'].unique())
item_unique2 = list(data2['itemId'].unique())



c_n, common_user = get_common_user(user_unique, user_unique2)
t_n, total_user = get_total_user(user_unique, user_unique2)
pprint('raw_data1 info : %d %d %d' % (u_num, i_num, r_num), f)
pprint('raw_data2 info : %d %d %d' % (u_num2, i_num2, r_num2), f)
pprint('common user num %d' % c_n, f)
pprint('total user num %d' % t_n, f)


new_data_1, new_data_2 = get_common_data(data, data2, common_user) # overlap
# new_data_1, new_data_2 = filter_user(new_data_1, new_data_2)


u, i, r = get_unique_lenth(data)
u2, i2, r2 = get_unique_lenth(data2)
pprint('after filter_data1 info : %d %d %d %.6f' % (u, i, r, r / (u * i)), f)
pprint('after filter_data2 info : %d %d %d %.6f' % (u2, i2, r2, r2 / (u2 * i2)), f)

u, i, r = get_unique_lenth(new_data_1)
u2, i2, r2 = get_unique_lenth(new_data_2)
pprint('after common_data1 info : %d %d %d %.6f' % (u, i, r, r / (u * i)), f)
pprint('after common_data2 info : %d %d %d %.6f' % (u2, i2, r2, r2 / (u2 * i2)), f)


user_index = list(range(0, len(total_user)))
dic_u = dict(zip(total_user, user_index))

item_index1 = list(range(0, len(item_unique)))
dic_v1 = dict(zip(item_unique, item_index1))

item_index2 = list(range(0, len(item_unique2)))
dic_v2 = dict(zip(item_unique2, item_index2))

common_data1, dic_u = reindex_data(new_data_1,dic_u,dic_v1)
common_data2, dic_u2 = reindex_data(new_data_2, dic_u,dic_v2)
write_to_txt(common_data1, save_file1)
write_to_txt(common_data2, save_file2)

data1, dic_u = reindex_data(data, dic_u, dic_v1)
data2, dic_u2 = reindex_data(data2, dic_u, dic_v2)
write_to_txt(data1, save_file3)
write_to_txt(data2, save_file4)

# min1 = get_min_group_size(new_data_1)
# min2 = get_min_group_size(new_data_2)
# assert dic_u == dic_u2, 'user_dic not same'
# pprint('min user group size is %d %d' % (min1, min2), f)
# pprint('filter way: user>%d,item>%d' % (5, 10), f)
# print('after common_data+filter item info : %d %d %d'%(u,i,r))
# print('after common_data2+filter item info : %d %d %d'%(u2,i2,r2))
# write_to_txt(data1, save_file1)
# write_to_txt(data2, save_file2)
pprint('write data finished!', f)
