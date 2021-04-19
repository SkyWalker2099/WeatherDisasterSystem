import re

time_keys = {
    "下午":14,
    "上午":6,
    "中午":12,
    "凌晨":0,
    "晚":20,
    "傍晚":18,
    "夜间":20,
    "晚上":20,
    "白天":10,
    "早":6,
}

day_keys = {
    "目前":0,
    "今日":0,
    "今天":0,
    "今":0,
    "前日":-2,
    "前天":-2,
    "昨日":-1,
    "昨天":-1,
    "昨":-1
}


time_keys_pattern = re.compile("(下午|上午|凌晨|晚|傍晚|夜间|晚上|白天|早)")

date_keys_pattern = re.compile("(目前|前日|前天|今天|今|昨日|昨天|昨|当日)")

date_pattern_y = re.compile("([一二三四五六七八九十千百万零点两]+|[\d.]+)(年)")
date_pattern_m = re.compile("([一二三四五六七八九十千百万零点两]+|[\d.]+)(月)")
date_pattern_d = re.compile("(([一二三四五六七八九十千百万零点两]+|[\d.]+)(日|号))|(中旬|上旬|下旬|初|底)")
date_patterns = [date_pattern_y,date_pattern_m,date_pattern_d]

time_pattern_h = re.compile("([一二三四五六七八九十千百万零点两]+|[\d.]+)(时|点)")
time_pattern_m = re.compile("(([一二三四五六七八九十千百万零点两]+|[\d.]+)(分))|(半)")
time_patterns = [time_pattern_h,time_pattern_m]

def correct(date):
    days = [0,31,27,31,30,31,30,31,31,30,31,30,31]
    if date["dd"] != None and date["dd"] < 1:
        date["mm"] = date["mm"] -1
        if date["mm"] < 1:
            date["yyyy"] = date["yyyy"] - 1
            date["mm"] = 12 + date["mm"]
        date["dd"] = days[date["mm"]] + date["dd"]

    if (date["dd"] != None and date["mm"] != None):
        date["dd"] = min(days[date["mm"]], date["dd"])
    return date

def str2num(str):
    if str == "半":
        return 30
    elif str == "上旬":
        return 5
    elif str == "中旬":
        return 15
    elif str == "下旬":
        return 25
    elif str == "底":
        return 30
    elif str == "初":
        return 1

    str = re.search("([一二三四五六七八九十千百万零点两]+|[\d.]+)",str).group()
    if re.match("[\d.]+",str):
        return int(str)
    elif re.match("[一二三四五六七八九十千百万零点两]+",str):
        zhong={'零':0,'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
        danwei={'十':10,'百':100,'千':1000,'万':10000}
        num=0
        if len(str)==0:
            return 0
        if len(str)==1:
            if str == '十':
                return 10
            num=zhong[str]
            return num
        temp=0
        if str[0] == '十':
            num=10
        for i in str:
            if i == '零':
                temp=zhong[i]
            elif i == '一':
                temp=zhong[i]
            elif i == '二':
                temp=zhong[i]
            elif i == '三':
                temp=zhong[i]
            elif i == '四':
                temp=zhong[i]
            elif i == '五':
                temp=zhong[i]
            elif i == '六':
                temp=zhong[i]
            elif i == '七':
                temp=zhong[i]
            elif i == '八':
                temp=zhong[i]
            elif i == '九':
                temp=zhong[i]
            if i == '十':
                temp=temp*danwei[i]
                num+=temp
            elif i == '百':
                temp=temp*danwei[i]
                num+=temp
            elif i == '千':
                temp=temp*danwei[i]
                num+=temp
            elif i == '万':
                temp=temp*danwei[i]
                num+=temp
        if str[len(str)-1] != '十'and str[len(str)-1] != '百'and str[len(str)-1] != '千'and str[len(str)-1] != '万':
            num+=temp
        return num


def str2time(time,standard):
    t = {"hh": None,
        "mm": None}

    if time.find(":") != -1 or time.find("：") != -1:
        ss = re.split("[:：]",time)
        h = ss[0]
        m = ss[1]

        h = str2num(h)
        m = str2num(m)

        t["hh"] = h
        t["mm"] = m


    else:

        h = re.search(time_pattern_h,time)
        m = re.search(time_pattern_m,time)

        if h != None:
            h = h.group()
            h = str2num(h)
            t["hh"] = h

        if m != None:
            m = m.group()
            m = str2num(m)
            t["mm"] = m

    if re.search("(下午|上午|凌晨|晚|傍晚|夜间|晚上|白天|早)", time) != None:
        time_key = re.search("(下午|上午|凌晨|晚|傍晚|夜间|晚上|白天|早)", time).group()
        if t["hh"] != None and t["hh"] <= 12:
            if time_keys[time_key] > 12:
                t["hh"] = t["hh"] + 12
        elif t["hh"] == None:
            t["hh"] = time_keys[time_key]
            t["mm"] = 0

    if t["mm"] == None:
        t["mm"] = 0

    return t


def str2date(date,standard_date,last_date):
    date = date + "日"
    da = {
        "yyyy":None,
        "mm":None,
        "dd":None
    }
    if date_keys_pattern.search(date) != None:
        date_key = date_keys_pattern.search(date).group()
        if date_key == "当日":
            if last_date["dd"] != None:
                return last_date
            else:
                return None
        else:
            da["yyyy"] = standard_date["yyyy"]
            da["mm"] = standard_date["mm"]
            da["dd"] = standard_date["dd"] + day_keys[date_key]
            da = correct(da)
            return da

    if date.find("-") != -1 and len(date.split("-")) == 3:
        ss = date.split("-")
        y = ss[0]
        m = ss[1]
        d = ss[2]
    else:
        y = re.search(date_pattern_y,date)
        m = re.search(date_pattern_m,date)
        d = re.search(date_pattern_d,date)

    if y != None:
        y = y.group()
        y = str2num(y)
        da["yyyy"] = y

    if m != None:
        m = m.group()
        m = str2num(m)
        da["mm"] = m

    if d != None:
        d = d.group()
        d = str2num(d)
        da["dd"] = d

    if da["dd"] != None:
        if da["mm"] == None:
            if last_date["mm"] != None:
                da["mm"] = last_date["mm"]
            else:
                da["mm"] = standard_date["mm"]
        if da["yyyy"] == None:
            if last_date["yyyy"] != None:
                da["yyyy"] = last_date["yyyy"]
            else:
                da["yyyy"] = standard_date["yyyy"]

    if da["yyyy"] == None:
        if last_date["yyyy"] != None:
            da["yyyy"] = last_date["yyyy"]
        else:
            da["yyyy"] = standard_date["yyyy"]

    return da


# def dict2full(dt,standard_time):
#
#     d = dt["D"]
#     t = dt["T"]
#
#     if d["yyyy"] == None and d["mm"] == None and d["dd"] == None:
#         return None
#
#     if d["yyyy"] == None:
#         d["yyyy"] = standard_time["D"]["yyyy"]
#     if d["mm"] == None:
#         d["mm"] = standard_time["D"]["mm"]
#
#     def p(s):
#         if s == None:
#             return 0
#         else:
#             return s
#
#     res = "%04d年%02d月%02d日%02d时%02d分"%(p(d["yyyy"]),p(d["mm"]),p(d["dd"]),p(t["hh"]),p(t["mm"]))
#     return res
#

def pd(ddts,stime):
    # print(ddts)
    # print(stime)
    if ddts["yyyy"] != stime["yyyy"]:
        return ddts["yyyy"] < stime["yyyy"]
    elif ddts["mm"] != stime["mm"]:
        return ddts["mm"] < stime["mm"]
    elif ddts["dd"] != stime["dd"]:
        return ddts["dd"] < stime["dd"]
    return None

def pt(tdts,stime):
    if tdts["hh"] != stime["hh"]:
        return tdts["hh"] < stime["hh"]
    elif tdts["mm"] != stime["mm"]:
        return tdts["mm"] < stime["mm"]
    return None

def isituseful(datetime):
    return re.search("([一二三四五六七八九十千百万零点两]+|[\d.]+)(年|月|日|号|时|分|秒|点)",datetime) != None or \
           re.search("(下午|上午|凌晨|晚|傍晚|夜间|晚上|白天|早)",datetime) != None or \
           re.search("(目前|前日|前天|今天|今日|今|昨日|昨天|昨|当日)",datetime) != None


def KeyTimeDecide(datetimes,standard_time):

    dts = [datetime for datetime in datetimes if isituseful(datetime[1])]

    has_s_d = False
    has_s_t = False
    stime = {
        "D": {
            "yyyy": None,
            "mm": None,
            "dd": None
        },
        "T": {
            "hh": None,
            "mm": None
        }
    }


    has_o_d = False
    has_o_t = False
    otime = {
        "D": {
            "yyyy": None,
            "mm": None,
            "dd": None
        },
        "T": {
            "hh": None,
            "mm": None
        }
    }

    last_day = {
        "yyyy":None,
        "mm":None,
        "dd":None
    }

    has_d = False

    for dt in dts:
        # print(dt)
        if (dt[0] == 'DS'):
            ddts = str2date(dt[1], standard_time["D"],last_day)
            if ddts == None:
                continue
            if has_s_d == False or pd(ddts, stime["D"]) == True:
                stime["D"]["yyyy"] = ddts["yyyy"]
                stime["D"]["mm"] = ddts["mm"]
                stime["D"]["dd"] = ddts["dd"]
                stime["T"]["hh"] = None
                stime["T"]["mm"] = None
                has_d = True
                last_day = stime["D"]
                has_s_d = True
                has_s_t = False

        elif (dt[0] == 'TS' and has_d):
            tdts = str2time(dt[1], standard_time["T"])
            if (has_s_t == False) or pd(last_day,stime["D"]) == True or (pt(tdts, stime["T"]) == True and pd(last_day,stime["D"]) == None):
                stime["T"]["hh"] = tdts["hh"]
                stime["T"]["mm"] = tdts["mm"]
                stime["D"]["yyyy"] = last_day["yyyy"]
                stime["D"]["mm"] = last_day["mm"]
                stime["D"]["dd"] = last_day["dd"]
                has_s_t = True

        elif (dt[0] == 'DO'):
            ddto = str2date(dt[1], standard_time["D"],last_day)
            if ddto == None:
                continue
            if has_o_d == False or pd(ddto, otime["D"]) == False:
                otime["D"]["yyyy"] = ddto["yyyy"]
                otime["D"]["mm"] = ddto["mm"]
                otime["D"]["dd"] = ddto["dd"]
                otime['T']["hh"] = None
                otime['T']["mm"] = None
                has_d = True
                last_day = otime["D"]
                has_o_d = True
                has_o_t = False

        elif (dt[0] == 'TO' and has_d):
            tdto = str2time(dt[1], standard_time["T"])
            if (has_o_t == False) or pd(last_day,otime["D"]) == False or (pt(tdto, otime["T"]) == False and pd(last_day,otime["D"]) == None):
                otime["T"]["hh"] = tdto["hh"]
                otime["T"]["mm"] = tdto["mm"]
                otime["D"]["yyyy"] = last_day["yyyy"]
                otime["D"]["mm"] = last_day["mm"]
                otime["D"]["dd"] = last_day["dd"]
                has_o_t = True

    stime["D"] = correct(stime["D"])
    otime["D"] = correct(otime["D"])
    return stime,otime




def DrawStandardTime(datetime):
    dts = re.findall("([一二三四五六七八九十千百万零两]+|[\d]+)",datetime)
    if len(dts) < 3:
        return None
    try:
        if len(dts[0]) < 4:
            idx = 0
            while idx < len(dts):
                if len(dts[idx]) >= 3:
                    break
                idx+=1

            if idx == len(dts):
                y = str2num(dts[0])
                m = str2num(dts[1])
                d = str2num(dts[2])
            else:
                d = str2num(dts[idx][:2])
                m = str2num(dts[idx-1])
                y = str2num(dts[idx-2])
            if y <= 20:
                y+=2000
            else:
                y+=1900


        elif len(dts[0]) == 4:
            y = str2num(dts[0])
            m = str2num(dts[1])
            d = str2num(dts[2])

        return {
            "D":{
                "yyyy":y,
                "mm":m,
                "dd":d
            },
            "T":{
                "hh":None,
                "mm":None
            }
        }

    except Exception as e:
        return None



if __name__ == '__main__':
     pass
