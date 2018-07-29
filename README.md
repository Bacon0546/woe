# woe
woe和IV值计算，py3编写
<br/>
可以自动分组，也可以手动进行分组，其中数值型变量的自动分组使用CART进行分组
<br/>
包含支持单变量做woe进行替换的类，支持对缺失值的计算
## 项目依赖
numpy，pandas，sklearn
## 使用方法
```
from woe import Woe
df = pd.read_csv('./dataset.csv', sep='\t')
data = df[['col1', 'col2', 'col3', 'col4']].copy()
label = df.y
woe = Woe(min_sample_rate=0.03)
split_dict = {
        'col1': [0.038, 0.073, 0.158, 0.3],
        'col2': [5, 10, 25],
        'col4': [['重庆市'], ['上海市'], ['山东省', '广东省']]
    }
woe.fit(data, label, **split_dict)
#woe.fit(data, label) #自动寻找最优分割点
data_woe = woe.transform(data)
print(woe.woe_map_dict)
print("=====")
print(woe.woe_map_df)
print("=====")
woe.woe_map_df.to_excel('./map_woe.xlsx', sheet_name='Sheet1')
print("iv:", woe.iv)
print("=========")
print(woe.iv_df)
```
