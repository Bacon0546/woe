#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Bacon

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from collections import OrderedDict
from sklearn.utils import multiclass

class WoeSingleNumberic(object):
    def __init__(self, min_sample_rate=0.05, woe_min=-20, woe_max=20):
        """
        初始化函数
        :param bins: 最大分箱数目，int
        :param min_sample_rate: 每箱最少比例，float
        :param woe_min: woe最小值
        :param woe_max: woe最大值
        """
        # self.__bins = bins
        self.__min_sample_rate = min_sample_rate
        self.__min_sample = 1
        self.__woe_min = woe_min
        self.__woe_max = woe_max
        self.__split_list = [] # 分割点list
        self.__map_woe = OrderedDict() # woe映射字典，左闭右开区间
        self.__split_group = []
        self.__iv = 0

    def __check_label_binary(self, label):
        """
        检查标签是否是binary
        :param label: label array
        :return: None
        """
        label_type = multiclass.type_of_target(label)
        if label_type != 'binary':
            raise ValueError('WoW!WoW!：Label type must be binary')

    def __cal_bad_good_rate(self, label):
        """
        计算好坏比例
        :param label: label list or array
        :return: 好坏比例
        """
        bad_nums = sum(label)
        good_nums = len(label) - bad_nums
        return np.divide(bad_nums, good_nums)

    def __filter_isnan(self, col, y):
        """
        找到col中为nan的
        :param y:
        :return: nan的col及对应y
        """
        return [x for x in col if np.isnan(x)], [y[i] for i in range(len(col)) if np.isnan(col[i])]

    def __filter_notnan(self, col, y):
        """
        找到col中不为nan的
        :param y:
        :return: 不为nan的col及对应y
        """
        return [x for x in col if ~np.isnan(x)], [y[i] for i in range(len(col)) if ~np.isnan(col[i])]

    def __get_split_points(self, col):
        """
        计算分割点，使用相邻点的中值
        col: 分割的列
        begin: 设定的初始点
        end:设定的结束点
        return: 分割点的list
        """
        var_lst = np.unique(col)
        var_lst.sort()
        var_split_lst = []
        for i in range(len(var_lst) - 1):
            var_split = (var_lst[i] + var_lst[i + 1]) / 2
            var_split_lst.append(var_split)
        return var_split_lst

    def __cal_gini(self, y):
        """
        计算gini，此gini非彼gini
        :return: 长度及gini
        """
        y1 = np.mean(y)
        y0 = 1 - y1
        return len(y), 1 - np.square(y1 - y0)

    def __split_left_right(self, col, label, split_point, lr_flag):
        """
        根据切分点分割左右数据集，采取左开右闭
        :param col: 需要划分的列
        :param label: 需要划分的标签
        :param split_point: 切分点
        :param lr_flag: 左右标志
        :return: 切分的数据集
        """
        if lr_flag == 'left':
            return [x for x in col if x <= split_point], [label[i] for i in range(len(col)) if col[i] < split_point]
        return [x for x in col if x > split_point], [label[i] for i in range(len(col)) if col[i] >= split_point]

    def __get_best_split_point(self, col, y, min_sample):
        """
        得到最佳分割点
        :param col: 需要计算分割点的数据
        :param y: 对应的label
        :param min_sample: 最小分箱数目
        :return: 最佳分割点
        """
        var_split_lst = self.__get_split_points(col)
        cnt, gini = self.__cal_gini(y)
        best_gini = 0.0
        best_split_point = None
        for p in var_split_lst:
            left_col, left_y = self.__split_left_right(col, y, p, lr_flag='left')
            right_col, right_y = self.__split_left_right(col, y, p, lr_flag='right')
            if len(right_col) <= min_sample or len(left_col) <= min_sample:
                continue
            cnt_left, gini_left = self.__cal_gini(left_y)
            cnt_right, gini_right = self.__cal_gini(right_y)
            gini_tmp = gini - (cnt_left * gini_left + cnt_right * gini_right) / cnt
            if gini_tmp > best_gini:
                best_gini = gini_tmp
                best_split_point = p
        gini = gini - best_gini
        return best_split_point

    def __get_best_split_point_lst(self, col, y, min_sample, split_list):
        """
        递归计算得到最佳分割点list
        :param col:
        :param y:
        :param min_sample:
        :param bins:
        :param split_list:
        :return:
        """
        # if len(split_list) < bins - 1:
        split = self.__get_best_split_point(col, y, min_sample)
        if split is not None:
            split_list.append(split)
            left_col, left_y = self.__split_left_right(col, y, split, lr_flag='left')
            right_col, right_y = self.__split_left_right(col, y, split, lr_flag='right')
            if len(left_col) >= min_sample * 2: # and len(split_list) < bins - 1
                self.__get_best_split_point_lst(left_col, left_y, min_sample, split_list)
            if len(right_col) >= min_sample * 2: #  and len(split_list) < bins - 1
                self.__get_best_split_point_lst(right_col, right_y, min_sample, split_list)

    def __get_best_split_list(self, col, label):
        """
        计算最佳分割点
        :param col:
        :param label:
        :param bins:
        :param min_sample:
        :return:
        """
        #bins = min(len(self.__get_split_points(col)) + 1, self.__bins)
        self.__get_best_split_point_lst(col, label, self.__min_sample, self.__split_list)
        # print(self.__split_list)
        self.__split_list.sort(reverse=False)
        # print(self.__split_list)

    def __cal_woe(self, label, bad_good_rate_all):
        """
        计算woe
        :param label: 标签
        :param bad_good_rate_all: 整体样本的好坏比
        :return:
        """
        bad_good_rate = self.__cal_bad_good_rate(label)
        if np.isinf(bad_good_rate):
            return 20
        if bad_good_rate == 0:
            return -20
        return np.log(bad_good_rate / bad_good_rate_all)

    def __woe(self, col, label):
        """
        计算woe和iv
        :param col: 需要计算的列，array
        :param label: 标签， array
        :return:
        """
        split_list_tmp = self.__split_list.copy()
        bad_good_rate_all = self.__cal_bad_good_rate(label)
        nan_col, nan_label = self.__filter_isnan(col, label)
        y_cnt = sum(label)
        n_cnt = len(label) - y_cnt
        notnan_col, notnan_label = self.__filter_notnan(col, label)
        for k in zip([-np.inf] + split_list_tmp, split_list_tmp + [np.inf]):
            self.__split_group.append(k)
            label_tmp = [notnan_label[i] for i in range(len(notnan_col)) if notnan_col[i] >= k[0] and notnan_col[i] < k[1]]
            y_cnt_tmp = sum(label_tmp)
            n_cnt_tmp = len(label_tmp) - y_cnt_tmp
            self.__map_woe[k] = self.__cal_woe(label_tmp, bad_good_rate_all)
            self.__iv += self.__map_woe[k] * (y_cnt_tmp / y_cnt - n_cnt_tmp / n_cnt)
        if len(nan_col) > 0:
            self.__map_woe['default'] = self.__cal_woe(nan_label, bad_good_rate_all)
            self.__iv += self.__map_woe['default'] * (sum(nan_label) / y_cnt - (len(nan_label) - sum(nan_label)) / n_cnt)
        else:
            self.__map_woe['default'] = self.__map_woe[self.__split_group[0]]


    def fit(self, col, label, split_list=None):
        """
        fit func
        :param col: 需要得到woe映射关系的列，array
        :param label: 对应标签
        :param split_list: 如果给定，则按照这个计算，否则，自动划分
        :return:
        """
        col = np.array(col)
        label = np.array(label)
        self.__check_label_binary(label)
        if split_list is None or ( not isinstance(split_list, list) ) or len(split_list) == 0:
            self.__min_sample = int(len(col) * self.__min_sample_rate)
            self.__get_best_split_list(col, label)
        else:
            self.__split_list = split_list
            self.__split_list.sort(reverse=False)
        self.__woe(col, label)

    def transform(self, col):
        """
        trandform func
        :param col:
        :return: woe_val
        """
        res_col = []
        for v in col:
            for k in self.__map_woe.keys():
                if k == 'default':
                    res_col.append(self.__map_woe[k])
                    break
                elif v >= k[0] and v < k[1]:
                    res_col.append(self.__map_woe[k])
                    break
        return np.array(res_col)
    @property
    def woe_map(self):
        return self.__map_woe
    @property
    def iv(self):
        return self.__iv
    @property
    def split_list(self):
        return self.__split_list
    @property
    def min_sample(self):
        return self.__min_sample

class WoeSingleObject(object):
    def __init__(self, min_sample_rate=0.05, woe_min=-20, woe_max=20):
        """

        :param min_sample_rate:
        :param woe_min:
        :param woe_max:
        """
        self.__min_sample_rate = min_sample_rate
        self.__min_sample = 1
        self.__woe_min = woe_min
        self.__woe_max = woe_max
        self.__split_list = []
        self.__var_lst_detail = []
        self.__map_woe = OrderedDict()  # woe映射字典，左闭右开区间
        self.__iv = 0

    def __check_label_binary(self, label):
        """
        检查标签是否是binary
        :param label: label array
        :return: None
        """
        label_type = multiclass.type_of_target(label)
        if label_type != 'binary':
            raise ValueError('WoW!WoW!：Label type must be binary')

    def __cal_bad_good_rate(self, label):
        """
        计算好坏比例
        :param label: label list or array
        :return: 好坏比例
        """
        bad_nums = sum(label)
        good_nums = len(label) - bad_nums
        return np.divide(bad_nums, good_nums)

    def __filter_isnan(self, col, y):
        """
        找到col中为nan的
        :param y:
        :return: nan的col及对应y
        """
        return [x for x in col if len(str(x)) == 0 or str(x) == 'nan'], [y[i] for i in range(len(col)) if len(str(col[i])) == 0 or str(col[i]) == 'nan']

    def __filter_notnan(self, col, y):
        """
        找到col中不为nan的
        :param y:
        :return: 不为nan的col及对应y
        """
        return [x for x in col if (len(str(x)) > 0) and (str(x) != 'nan')], [y[i] for i in range(len(col)) if (len(str(col[i])) > 0) and (str(col[i]) != 'nan')]

    def __get_split_list(self, col, label):
        """
        对非nan的变量及对应的y做处理，首先计算每个取值对应的数量，如果满足最小分箱数，则单独一组，否则跟下一组合并，
        继续判断是否满足，直到满足为止，如果最后一个箱子，不满足单独条件，则与第一组合并
        :param col:
        :param label:
        :return:
        """
        var_lst = np.unique(col)
        var_lst_detail_tmp = []
        for v in var_lst:
            var_detail_tmp = []
            var_detail_tmp.append([v])
            col_tmp = [x for x in col if x == v]
            var_detail_tmp.append(len(col_tmp))
            var_lst_detail_tmp.append(var_detail_tmp)
        var_lst_detail_tmp.sort(key=lambda x: x[1])
        tmp = [[], 0]
        for var_detail in var_lst_detail_tmp:
            if var_detail[1] >= self.__min_sample:
                self.__var_lst_detail.append(var_detail)
            else:
                tmp[0].append(var_detail[0][0])
                tmp[1] += var_detail[1]
                if tmp[1] >= self.__min_sample:
                    self.__var_lst_detail.append(tmp)
                    tmp = [[], 0]
        if tmp[1] > 0 and tmp[1] < self.__min_sample:
            self.__var_lst_detail[0][0].append(tmp[0][0])
            self.__var_lst_detail[0][1] += tmp[1]
        self.__split_list = [tuple(x[0]) for x in self.__var_lst_detail]

    def __cal_woe(self, label, bad_good_rate_all):
        bad_good_rate = self.__cal_bad_good_rate(label)
        if np.isinf(bad_good_rate):
            return 20
        if bad_good_rate == 0:
            return -20
        return np.log(bad_good_rate / bad_good_rate_all)

    def __woe(self, col, label):
        bad_good_rate_all = self.__cal_bad_good_rate(label)
        nan_col, nan_label = self.__filter_isnan(col, label)
        y_cnt = sum(label)
        n_cnt = len(label) - y_cnt
        notnan_col, notnan_label = self.__filter_notnan(col, label)
        for k in self.__split_list:
            label_tmp = [notnan_label[i] for i in range(len(notnan_col)) if notnan_col[i] in k]
            y_cnt_tmp = sum(label_tmp)
            n_cnt_tmp = len(label_tmp) - y_cnt_tmp
            self.__map_woe[k] = self.__cal_woe(label_tmp, bad_good_rate_all)
            self.__iv += self.__map_woe[k] * (y_cnt_tmp / y_cnt - n_cnt_tmp / n_cnt)
        if len(nan_col) > 0:
            self.__map_woe['default'] = self.__cal_woe(nan_label, bad_good_rate_all)
            self.__iv += self.__map_woe['default'] * (sum(nan_label) / y_cnt - (len(nan_label) - sum(nan_label)) / n_cnt)
        else:
            self.__map_woe['default'] = self.__map_woe[self.__split_list[0]]

    def fit(self, col, label, split_list = None):
        """
        fit funv
        :param col:
        :param label:
        :param split_list:
        :return:
        """
        col = np.array(col)
        label = np.array(label)
        self.__check_label_binary(label)
        if split_list is None or ( not isinstance(split_list, list) ) or len(split_list) == 0:
            self.__min_sample = int(len(col) * self.__min_sample_rate)
            notnan_col, notnan_label = self.__filter_notnan(col, label)
            self.__get_split_list(notnan_col, notnan_label)
        else:
            self.__split_list = [tuple(x) for x in split_list]
        self.__woe(col, label)

    def transform(self, col):
        """
        trandform func
        :param col:
        :return: woe_val
        """
        res_col = []
        col = np.array(col)
        for v in col:
            for k in self.__map_woe.keys():
                if k == 'default':
                    res_col.append(self.__map_woe[k])
                    break
                elif v in k:
                    res_col.append(self.__map_woe[k])
                    break
        return np.array(res_col)

    @property
    def woe_map(self):
        return self.__map_woe
    @property
    def iv(self):
        return self.__iv
    @property
    def split_list(self):
        return self.__split_list
    @property
    def min_sample(self):
        return self.__min_sample

class Woe(object):
    def __init__(self, min_sample_rate=0.05, woe_min=-20, woe_max=20):
        """
        初始化函数
        :param min_sample_rate: 每箱数量占总样本最大比例
        :param woe_min:
        :param woe_max:
        """
        self.__min_sample_rate = min_sample_rate
        self.__min_sample = 1
        self.__woe_min = woe_min
        self.__woe_max = woe_max
        self.__split_list = {}
        self.__map_woe = {}  # woe映射字典，左闭右开区间
        self.__iv = {}
        self.__woe = {}
        self.__column_type = {}

    def __check_label_binary(self, label):
        """
        检查标签是否是binary
        :param label: label array
        :return: None
        """
        label_type = multiclass.type_of_target(label)
        if label_type != 'binary':
            raise ValueError('WoW!WoW!：Label type must be binary')

    def __fit_single_woe(self, df, col, label, woe, **split_dict):
        """
        fit单变量的函数
        :param df: 需要处理的数据集，DataFrame
        :param col: 处理的列名
        :param label: 标签列
        :param woe: 传入的woe对象
        :param split_dict: 分割点的dict
        :return:
        """
        if col in split_dict.keys():
            self.__split_list[col] = split_dict[col]
        else:
            self.__split_list[col] = None
        woe.fit(df[col], label, self.__split_list[col])
        self.__split_list[col] = woe.split_list
        self.__map_woe[col] = woe.woe_map
        self.__iv[col] = woe.iv
        self.__woe[col] = woe

    def fit(self, df, label, **split_dict):
        """
        fit func
        :param df: 数据集
        :param label: 标签
        :param split_dict: 手动确定分割点，dict类型，如{''col1': [0,1,2,3], 'col2': [['A'], ['B'], ['c', 'd', 'e']}
        :return: None
        """
        label = np.array(label)
        self.__check_label_binary(label)
        self.__min_sample = int(len(df) * self.__min_sample_rate)
        for col in df.columns.tolist():
            print(col, ' start fitting...')
            if is_numeric_dtype(df[col]):
                self.__column_type[col] = 'number'
                woe = WoeSingleNumberic(self.__min_sample_rate, self.__woe_min, self.__woe_max)
            else:
                self.__column_type[col] = 'object'
                woe = WoeSingleObject(self.__min_sample_rate, self.__woe_min, self.__woe_max)
            self.__fit_single_woe(df, col, label, woe, **split_dict)
            print(col, ' fitted...')

    def transform(self, df):
        """
        transform func
        :param df: 需要做woe变换的数据集
        :return:
        """
        if df.shape[-1] != len(self.__map_woe):
            raise "input DataFrame must have the same columns with origion Dataframe"
        df_tmp = df.copy()
        for col in df_tmp.columns.tolist():
            if col not in self.__map_woe.keys():
                raise "{} not exists in the origin Dataframe...".format(col)
            df_tmp[col] = self.__woe[col].transform(df_tmp[col])
        return df_tmp
    @property
    def woe_map_dict(self):
        return self.__map_woe
    @property
    def woe_map_df(self):
        res = []
        for k, v in self.__map_woe.items():
            group  = [str(x) for x in v.keys()]
            if self.__column_type[k] == 'number':
                group = ['[' + x[1:] for x in group if x != 'default'] + ['default']
            woe  = [x for x in v.values()]
            try:
                tmp = pd.DataFrame({'var': [k] * len(v), 'group': group, 'woe': woe})
            except:
                print(group, woe)
            tmp.set_index(['var', 'group'], inplace=True)
            res.append(tmp)
        return pd.concat(res)
    @property
    def iv(self):
        return self.__iv
    @property
    def iv_df(self):
        return pd.DataFrame({'vars': list(self.__iv.keys()), 'iv': list(self.__iv.values())})
    @property
    def split_list(self):
        return self.__split_list
    @property
    def min_sample(self):
        return self.__min_sample
