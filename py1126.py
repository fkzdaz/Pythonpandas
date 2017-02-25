import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 一、创建对象
# 1、可以通过传递一个list对象来创建一个Series，pandas会默认创建整型索引：
s=pd.Series([1,3,5,6,np,6,9,8])
print(s)
# 2、通过传递一个numpy array，时间索引以及列标签来创建一个DataFrame：
dates=pd.date_range('20170224',periods=6)
print(dates)
df=pd.DataFrame(np.random.rand(6,4),index=dates,columns=list('ABCD'))
print(df)
# 3、通过传递一个能够被转换成类似序列结构的字典对象来创建一个DataFrame：
df2=pd.DataFrame({
    'A':1.,
    'B':pd.Timestamp('20170226'),
    'C':pd.Series(1,index=(range(4)),dtype="float32"),
    'D':np.array([4]*4,dtype='int32'),
    'E':pd.Categorical(["test","train","test","train"]),
    'F':'foo'
})
print(df2)
# 3、查看不同列的数据类型：
print("  ")
print(df2.dtypes)

# 二、查看数据
#1、 查看frame中头部和尾部的行：
print(df2.head())
print(df.head())
print(df.head(1))
print(df.tail(1))

#2、 显示索引、列和底层的numpy数据：
print('  ')
print(df.index)
print("  ")
print(df.columns)
print("  ")
print(df.values)

# 3、describe()函数对于数据的快速统计汇总：
print(" ")
print(df.describe())
print(" ")
print(df2.describe())
# 4、对数据的转置：
print(" ")
print(df.T)

# 4、按轴进行排序
print("1 ")
print(df.sort_index(axis=1,ascending=False))
print("2 ")
print(df.sort_index(axis=0,ascending=False))

# 5、按值进行排序
print("  ")
print(df.sort(columns='B'))
print(" ")
print(df.sort(columns='A'))

# 三、选择

# 1、获取
# 1、1 选择一个单独的列，这将会返回一个Series，等同于df.A：
print(" ")
print(df['A'])
# 1.2 通过[]进行选择，这将会对行进行切片
print(" ")
print(df[0:3])
print(df[1:3])
print(df['20170224':'20170226'])

# 2 通过标签选择
# 2.1 使用标签来获取一个交叉的区域
print(" ")
print(df.loc[dates[0]])
# 2.2 通过标签来在多个轴上进行选择
print(" ")
print(df.loc[:,['A','B']])
# 2.3 标签切片
print(" ")
print(df.loc['20170224':'20170228',['A','B']])
# 2.4 对于返回的对象进行维度缩减
print(" ")
print(df.loc['20170226',['A','B']])
# 2.5  快速访问一个标量
print( " ")
print(df.at[dates[0],'A'])
# 2.6 获取一个标量
print(" ")
print(df.loc[dates[0],'A'])

# 3 通过位置选择
# 3.1  通过传递数值进行位置选择（选择的是行）
print(" ")
print(df.iloc[3])
print(" ")
print(df.iloc[2])
print(df.loc['20170224':'20170301'].T)
# 3.2 通过数值进行切片，与numpy/python中的情况类似
print(" ")
print(df.loc['20170224':'20170301'])
print(df.iloc[3:5,2:4])
# 3.3通过指定一个位置的列表，与numpy/python中的情况类似
print(" ")
print(df.iloc[[1,2,4],[0,2]])
# 3.4 对行进行切片
print(" ")
print(df.iloc[1:3:])
# 3.5 对列进行切片
print(" ")
print(df.iloc[:,1:3])
# 3.6 对指定的行列进行切片
print(" ")
print(df.iloc[1:3,1:3])
# 3.7 获取特定的值
print(" ")
print(df.iloc[1,1])
print(" ")
print(df.iloc[3,3])
print(" ")
print(df.iat[1,1])

# 4 布尔索引
# 4.1  使用一个单独列的值来选择数据：
print(" ")
print(df[df.B>0.9])
# 4.2 使用where操作来选择数据：
print(" ")
print(df[df>0.5])
# 4.3 使用isin()方法来过滤：
print(" ")
df2=df.copy()
df2['E']=['one','one','two','three','four','three']
print(df2)
print( " ")
print(df2[df2['E'].isin(['two','four'])])

# 5 设置
# 5.1 设置一个新的列：
print(" ")
s1=pd.Series([1,2,3,4,5,6],index=pd.date_range("20170302",periods=6))
print(s1)
# 5.2 通过标签设置新的值：
df.at[dates[0],'A']=0
print(df)
# 5.3  通过位置设置新的值：
print(" ")
df.iat[0,1]=0
print(df)
# 5.4 通过一个numpy数组设置一组新值：?
print(" ")
df.loc[:,'D']=np.array([5]*len(df))
print(df)
# 5.5  通过where操作来设置新的值：
df2=df.copy()
print(" ")
df2[df2>0]=-df2
print(df2)

# 四 缺失值处理
# 在pandas中，使用np.nan来代替缺失值，这些值将默认不会包含在计算中，详情请参阅：Missing Data Section。
# 1.1 reindex()方法可以对指定轴上的索引进行改变/增加/删除操作，这将返回原始数据的一个拷贝：
print(" ")
df1=df.reindex(index=dates[0:4],columns=list(df.columns)+['E'])
df1.loc[dates[0]:dates[1],'E']=1
print(df1)
# 1.2 去掉包含缺失值的行：
print(" ")
print(df1.dropna(how="any"))
# 1.3 对缺失值进行填充：
print(" ")
print(df1.fillna(value=5))
# 1.4 对数据进行布尔填充：
print(" ")
print(pd.isnull(df1))

# 五、相关操作
# 1  统计（相关操作通常情况下不包括缺失值）
# 1.1 执行描述性统计：？
print(" ")
print(df.mean())
# 1.2 在其他轴上进行相同的操作：
print(" ")
print(df.mean(1))
# 1.3 对于拥有不同维度，需要对齐的对象进行操作。Pandas会自动的沿着指定的维度进行广播：?
print(" ")
s=pd.Series([1,3,5,np.nan,6,8],index=dates).shift(2)
print(s)
print(" ")
print(df.sub(s,axis='index'))
# 2 Apply对数据应用函数：
print(" ")
print(df)
print(" ")
print(df.apply(np.cumsum))
print(" ")
print(df.apply(lambda x:x.max()-x.min()))
# 3 直方图
print(" ")
s=pd.Series(np.random.randint(0,10,size=10))
print(s)
# 4 字符串方法
# Series对象在其str属性中配备了一组字符串处理方法，可以很容易的应用到数组中的每个元素，如下段代码所示。
print(" ")
s=pd.Series(['A','B','C','AaBa','BaCa',np.nan,'CABA','dog','cat']
    )
print(s.str.lower())

# 六、合并
# Pandas提供了大量的方法能够轻松的对Series，DataFrame和Panel对象进行各种符合各种逻辑关系的合并操作。
# 1 Concat ?
print(" ")
df=pd.DataFrame(np.random.randn(10,4))
print(df)
print(" ")
pieces=[df[:3],df[3:7],df[7:]]
print(pd.concat(pieces))
# 2 Join 类似于SQL类型的合并
print(" ")
left=pd.DataFrame({
    'key':['fkz','fkz'],'rval':[1,2]
})
right=pd.DataFrame({
    'key':['fkz','fkz'],'rval':[4,5]
})
print(left)
print(right)
print("")
print(pd.merge(left,right,on='key'))
# 3 Append 将一行连接到一个DataFrame上 ?
print("")
df=pd.DataFrame(np.random.randn(8,4),columns=['A','B','C','D'])
print(df)
print("")
s=df.iloc[3]
print(df.append(s,ignore_index=True))
s1=df.iloc[3]
print(df.append(s1,ignore_index=True))

# 七、分组
# 对于”group by”操作，我们通常是指以下一个或多个操作步骤：
# l  （Splitting）按照一些规则将数据分为不同的组；
# l  （Applying）对于每组数据分别执行一个函数；
# l  （Combining）将结果组合到一个数据结构中；
df=pd.DataFrame({
    'A':['fkz','bar','fkz','fdz','fkz','bar','fkz','fdz'],
    'B':['one','two','three','four','one','two','three','four'],
    'C':np.random.randn(8),
    'D':np.random.randn(8)
})
print("")
print(df)
# 1.分组并对每个分组执行sum函数：
print("")
print(df.groupby('A').sum())
# 2.通过多个列进行分组形成一个层次索引，然后执行函数：
print("")
print(df.groupby(['A','B']).sum())

# 八、Reshaping
# 1. Stack
tuples=list(zip(*[['bar','bar','fkz','fkz','faz','fdz','que','que'],['one','two','one','two','one','two','one','two']]))
index=pd.MultiIndex.from_tuples(tuples,names=['frist','second'])
df=pd.DataFrame(np.random.randn(8,2),index=index,columns=['A','B'])
df2=df[:4]
print("")
print(df2)
print("")
stacked=df2.stack()
print(stacked)
print("")
print(stacked.unstack())
print("")
print(stacked.unstack(0))
print("")
print(stacked.unstack(1))
# 2.数据透视表
df=pd.DataFrame({
    'A':['one','one','two','three']*3,
    'B':['A','B','C']*4,
    'C':['foo','foo','foo','bar','bar','bar']*2,
    'D':np.random.randn(12),
    'E':np.random.randn(12)
})
print("")
print(df)
print("")
# 可以从这个数据中轻松的生成数据透视表：
pdf=pd.pivot_table(df,values='D',index=['A','B'],columns=['C'])
print(pdf)
# 九、时间序列
# Pandas在对频率转换进行重新采样时拥有简单、强大且高效的功能（如将按秒采样的数据转换为按5分钟为单位进行采样的数据）。这种操作在金融领域非常常见。具体参考：Time
#  Series section。
# 1.时区表示：
print("")
rng=pd.date_range('26/02/2017 00:00',periods=5,freq='D')
print(rng)
ts=pd.Series(np.random.randn(len(rng)),rng)
print(ts)
print("")
ts_utc=ts.tz_localize('UTC')
print(ts_utc)
# 2.时区转换：
print("")
print(ts_utc.tz_convert('US/Eastern'))
# 3.时间跨度转换：
print("")
rng=pd.date_range('26/02/2017',periods=5,freq='M')
ts=pd.Series(np.random.randn(len(rng)),rng)
print(ts)
print("")
ps=ts.to_period()
print(ps)
print("")
print(ps.to_timestamp())
# 4.时期和时间戳之间的转换使得可以使用一些方便的算术函数。?
prng=pd.period_range('1995Q1','2017Q4',freq='Q-NOV')
ts=pd.Series(np.random.randn(len(prng)),prng)
ts.index=(prng.asfreq('M','e')+1).asfreq('H','s')+9
print("")
print(ts.head())

# 十、Categorical?
# 从0.15版本开始，pandas可以在DataFrame中支持Categorical类型的数据
df=pd.DataFrame({
    'id':[1,2,3,4,5,6],
    'raw_grade':['a','b','b','a','a','e']
})
print("")
print(df)
# 1.将原始的grade转换为Categorical数据类型：
print("")
df['grade']=df['raw_grade'].astype('category')
print(df['grade'])
# 2.将Categorical类型数据重命名为更有意义的名称：
print("")
df['grade'].cat.categories=['very good','good','very bad']
# 3.对类别进行重新排序，增加缺失的类别：
df['grade']=df['grade'].cat.set_categories(['very good','good','very bad'])
print([df['grade']])
# # 4.排序是按照Categorical的顺序进行的而不是按照字典顺序进行：?
# print("")
# print(df.sort('grade'))
# 5.对Categorical列进行排序时存在空的类别：
print("")
print(df.groupby('grade').size())
# 十一、画图'!!!!!
print("")
ts=pd.Series(np.random.randn(1000),index=pd.date_range('26/02/2017',periods=1000))
ts=ts.cumsum()
print(ts.plot())
# 对于DataFrame来说，plot是一种将所有列及其标签进行绘制的简便方法：
df=pd.DataFrame(np.random.randn(1000,4),index=ts.index,columns=['A','B','C','D'])
df=df.cumsum()
plt.figure() ; df.plot() ;plt.legend(loc='best')
# 十二、导入和保存数据
# 1.写入csv文件：
print("CSV")
df.to_csv('fkz.csv')
# 2.从csv文件中读取：
print(pd.read_csv('fkz.csv'))
# 1.写入HDF5存储：
print("HDF5")
df.to_hdf('fkz.h5','df')
# 2.从HDF5存储中读取：
print(pd.read_hdf('fkz.h5','df'))
# 1.写入excel文件：
print("Excel")
df.to_excel('fkz.xlsx',sheet_name='Sheet1')
# 2.从excel文件中读取：
print(pd.read_excel('fkz.xlsx','Sheet1',index_col=None,na_values=['NA']))




