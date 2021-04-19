# -*- coding: UTF-8 -*-
import pymysql

db = pymysql.connect("localhost","zzh","123456","weather_disasters")
cursor = db.cursor(cursor=pymysql.cursors.DictCursor)

last_zq_id = -1
last_zq_news_id = -1

"""
由于各表主键都设置为 auto_increment, 因此在插入数据时并不指定主键值
"""

def weather_disaster_news_insert(zq_title,zq_time,info_source,info_publish,zq_content):
    """
    插入一条灾情新闻数据，各个字段都不能为None
    :param zq_title:
    :param zq_time:
    :param info_source:
    :param info_publish:
    :param zq_content:
    :return:
    """
    news_dict = {"zq_title":zq_title,"zq_time":zq_time,"info_source":info_source,"info_publish":info_publish,"zq_content":zq_content}
    keys = []
    s = []
    values = []
    for key in news_dict.keys():
        if (news_dict[key] != None):
            keys.append(key)
            s.append('%s')
            values.append(news_dict[key])

    sql1 = '(' + ','.join(keys) + ')'
    sql2 = '(' + ','.join([ss for ss in s]) + ')'

    sql = """insert into weather_disaster_news """ + sql1 + " values" + sql2
    print(sql)
    try:
        cursor.execute(sql, values)
        db.commit()
    except Exception as e:
        db.rollback()
        e.with_traceback()

    cursor.execute("select LAST_INSERT_ID() from weather_disaster_news;")
    data = cursor.fetchone()
    global last_zq_news_id
    last_zq_news_id = data["LAST_INSERT_ID()"]



def weather_disaster_info_insert(zq_type,zq_title,zq_news_id = None,
                                 zq_startdate = None,zq_enddate = None,process_starttime = None,process_endtime = None,
                                 info_source = None,economic_loss = None,
                                 pop_disaster = None,pop_death = None,pop_miss = None,pop_illness = None,pop_transfer = None,
                                 house_down = None,crop_disaster = None,crop_grow = None,crop_harvest = None,
                                 wt_depth = None,sw_depth = None):
    """
    用于插入一条主表数据，在该函数中，zq_type,zq_title不能为空，其他字段可以为空（待修改）
    :param zq_type: string
    :param zq_title: string
    :param zq_news_id: 指向对应的新闻文本，若设置为-1，则不指向任何新闻文本，若不设置，则会指向最近插入的灾情新闻文本
    :param zq_startdate: 数据可为 yy-mm-dd 格式的字符串
    :param zq_enddate: 数据可为 yy-mm-dd 格式的字符串
    :param process_starttime: 数据可为 hh:mm:ss 格式的字符串
    :param process_endtime: 数据可为 hh:mm:ss 格式的字符串
    :param info_source: string
    :param economic_loss: float
    :param pop_disaster: int
    :param pop_death: int
    :param pop_miss: int
    :param pop_illness: int
    :param pop_transfer: int
    :param house_down: int
    :param crop_disaster: float
    :param crop_grow: float
    :param crop_harvest: float
    :param wt_depth: float
    :param sw_depth: float
    :return:
    """
    if zq_news_id == None:#在使用函数不对其进行设置
        if last_zq_news_id == -1:#若在之前未插入过新闻文本信息
            raise Exception("请指定插入的信息所属的灾情信息的zq_id,或者在进行一次灾情新闻文本插入后再进行灾情信息的插入")
        else:
            zq_news_id = last_zq_news_id#若插入过新闻文本信息，则会指向最近插入的新闻文本
    elif zq_news_id == -1:
        zq_news_id = None #如将其设置为-1，则该灾情信息不会与任何新闻文本相关


    zq_info_dict = {"zq_type":zq_type,"zq_title":zq_title,"zq_news_id":zq_news_id,
               "zq_startdate":zq_startdate,"zq_enddate":zq_enddate,"process_starttime":process_starttime,"process_endtime":process_endtime,
                "info_source":info_source,"economic_loss":economic_loss,
                "pop_disaster":pop_disaster,"pop_death":pop_death,"pop_miss":pop_miss,"pop_illness":pop_illness,"pop_transfer":pop_transfer,
                "house_down":house_down,"crop_disaster":crop_disaster,"crop_grow":crop_grow,"crop_harvest":crop_harvest,
                "wt_depth":wt_depth,"sw_depth":sw_depth}

    keys = []
    s = []
    values = []
    for key in zq_info_dict.keys():
        if(zq_info_dict[key] != None):
            keys.append(key)
            s.append('%s')
            values.append(zq_info_dict[key])

    if len(keys) == 0:
        raise Exception("数据不能全为空")

    sql1 = '('+','.join(keys)+')'
    sql2 = '('+','.join([ss for ss in s])+')'

    sql = """insert into weather_disasters_infos """+sql1+" values"+sql2
    print(sql)
    try:
        cursor.execute(sql,values)
        db.commit()
    except Exception as e:
        db.rollback()
        e.with_traceback()

    cursor.execute("select LAST_INSERT_ID() from weather_disasters_infos;")
    data = cursor.fetchone()
    global last_zq_id
    last_zq_id = data["LAST_INSERT_ID()"]

def disaster_locations_info_insert(zq_id = None,province_id = None,city_id = None,county_id = None,country_id = None,location = None):
    """
    用于插入一条灾害发生位置信息，外键 zq_id 默认为最近一条插入的灾情受灾信息的主键（只有在脚本先插入一条主表数据后才能在zq_id为默认值的情况下进行此函数，否则会报错）， 也可自己设置。
    :param zq_id: 若要自己指定，则应为int类型
    :param province_id: string
    :param city_id: string
    :param county_id: string
    :param country_id: string
    :param location: string
    :return:
    """
    if zq_id == None:
        if last_zq_id == -1:
            raise Exception("请指定插入的信息所属的灾情信息的zq_id,或者在进行一次灾情信息插入后再进行灾情地址信息的插入")
        zq_id = last_zq_id

    locs_dict = {"zq_id":zq_id,"province_id":province_id,"city_id":city_id,"county_id":county_id,"country_id":country_id,"location":location}
    keys = []
    s = []
    values = []
    for key in locs_dict.keys():
        if (locs_dict[key] != None):
            keys.append(key)
            s.append('%s')
            values.append(locs_dict[key])

    if len(keys) == 1 and keys[0] == 'zq_id':
        raise Exception("数据不能全为空")

    sql1 = '(' + ','.join(keys) + ')'
    sql2 = '(' + ','.join([ss for ss in s]) + ')'

    sql = """insert into disaster_locations_infos """ + sql1 + " values" + sql2
    print(sql)
    try:
        cursor.execute(sql, values)
        db.commit()
    except Exception as e:
        db.rollback()
        e.with_traceback()

def disaster_victims_info_insert(zq_id = None,victim_first = None,victim_second = None,victim_third = None):
    """
    用于插入一条灾害受灾体信息，外键 zq_id 默认为最近一条插入的灾情受灾信息的主键（只有在脚本先插入一条主表数据后才能在zq_id为默认值的情况下进行此函数，否则会报错）， 也可自己设置。
    :param zq_id: 若要自己指定，则应为int类型
    :param victim_first: string
    :param victim_second: string
    :param victim_third: string
    :return:
    """
    if zq_id == None:
        if last_zq_id == -1:
            raise Exception("请指定插入的信息所属的灾情信息的zq_id,或者在进行一次灾情信息插入后再进行灾情受灾体信息的插入")
        zq_id = last_zq_id


    victims_dict = {"zq_id":zq_id,"victim_first":victim_first,"victim_second":victim_second,"victim_third":victim_third}
    keys = []
    s = []
    values = []
    for key in victims_dict.keys():
        if (victims_dict[key] != None):
            keys.append(key)
            s.append('%s')
            values.append(victims_dict[key])

    if len(keys) == 1 and keys[0] == 'zq_id':
        raise Exception("数据不能全为空")

    sql1 = '(' + ','.join(keys) + ')'
    sql2 = '(' + ','.join([ss for ss in s]) + ')'

    sql = """insert into disaster_victims_infos """ + sql1 + " values" + sql2
    print(sql)
    try:
        cursor.execute(sql, values)
        db.commit()
    except Exception as e:
        db.rollback()
        e.with_traceback()


if __name__ == '__main__':

    title = "四川降下特大冰雹，25人死亡"
    time = "2020-1-1"
    source = "http://www.baidu.com"
    pub = "中新网"
    zw = """今年7月前，通渭县因60年不遇旱灾损失逾亿元。进入7月以来，接踵而来的冰雹更让通渭人民雪上加霜。7月26日下午，一场罕见的特大冰雹再次突袭通渭：鸡蛋大小的冰雹铺天盖地而来，所到之处农作物损失惨重，房屋瓦片被砸烂，个别村民被砸得头破血流
特大冰雹突袭通渭
7月26日下午3时10分至5时30分，通渭县大多数乡镇先后不同程度地遭受了鸡蛋大的冰雹的袭击，灾害持续时间最长的地方达50分钟，冰雹的直径约2至5厘米，落地冰雹的厚度达10厘米。据通渭县有关部门统计，此次冰雹灾涉及该县鸡川、新景、陇山、陇川、寺子、陇阳、碧玉、平襄、襄南、什川等10个乡镇的51个村271个村民小组，受灾人口4万余人。
砸烂房瓦砸伤村民
7月27日上午8时开始，冰雹之后的一场大雨再次光临通渭。下午2时许，记者冒雨驱车来到了此次受灾最严重的碧玉、鸡川两乡镇采访。
在碧玉乡玉关村坡头社，记者见到了正在雨中安排抢险救灾工作的通渭县民政局局长赵鹏等人。赵鹏告诉记者，这是自1920年以来通渭县遭遇的最严重的一次冰雹灾难，其持续时间之长、分布范围之广、危害程度之大，十分罕见。
在坡头社社长陈中虎的指引下，记者走进了他家的院子。只见满地都是裂碎的瓦片。“鸡蛋大的冰雹，砸在瓦上咯咯地响，像石头一样！”陈中虎心有余悸地对记者说。据赵鹏介绍，当日的雹灾中，村民的房瓦至少有三成被砸烂。村民胥小燕撸起自己的衣袖，露出青肿的胳膊对记者说：“我的身上都是这样，冰雹来得快，跑都跑不及。”一位叫陈守珍的老人因来不及躲避，当场被一颗冰雹砸得头破血流。
农作物损失相当惨重
据通渭县民政局介绍，此次冰雹袭击共造成该县6902公顷农作物受灾，绝收面积874公顷。造成直接经济损失1879.14万元，其中农业直接经济损失1807.9万元。记者在采访中了解到，冰雹袭击后，个别无劳力的特困家庭将面临断粮的危机。目前，通渭县有关方面已对部分特别困难的群众启动救济方案。
通渭县副县长柳生坠告诉记者，对于通渭来说，今年是个多灾之年。7月之前，通渭全县遭遇了60年不遇的特大旱灾，直接经济损失逾亿元。进入7月以后，冰雹又接踵而来。7月13日、14日、24日、25日，通渭县部分乡镇先后4次遭遇冰雹袭击，损失1500多万元。7月26日，罕见的特大冰雹第5次袭击通渭10个乡镇，造成直接经济损失1800余万元，5次冰雹共造成直接经济损失3400多万元。另据了解，目前，定西市已就通渭县的灾情向省上有关部门进行了报告，通渭县的灾后自救工作已全面展开。
"""

    # weather_disaster_news_insert(zq_title=title,zq_time=time,info_source=source,info_publish=pub,zq_content=zw)
    #
    # weather_disaster_info_insert(zq_type="冰雹灾害",zq_title="四川降下特大冰雹，25人死亡",zq_startdate="20-12-20",zq_enddate="20-12-20",
    #                              process_starttime="12:20",process_endtime="1:00",info_source="http://www.baidu.com",economic_loss=20.2,
    #                              pop_disaster=100,pop_death=20,pop_miss=15,pop_illness=30,pop_transfer=60,
    #                              house_down=20,crop_disaster=12.5455445,crop_grow=12,crop_harvest=0.5454545,wt_depth=20.55,sw_depth=10)
    #
    # disaster_locations_info_insert(province_id="安徽",city_id="宿州",county_id="萧县",country_id="龙城镇",location="梅村中学")
    #
    # disaster_victims_info_insert(victim_first="人",victim_second="人类",victim_third="学生")
    #
    #连续执行以上四条命令时，第一条命令插入一条新闻文本信息，第二条命令插入一条灾情信息（默认外键指向第一条所插入的新闻文本信息），第三，四条命令分别插入位置信息和受载体信息（默认外键指向灾情信息）。



