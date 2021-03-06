# -*- coding:utf-8 -*-
# editor: zzh
# date: 2020/11/14

from re_extract import *

# k_v = {"AIMP": "受灾人口", "AMP": "失踪人口",
#                            "AINP": "受伤人口", "ATP": "转移人口", "ADP": "死亡人口", "AIAC": "受灾面积", "AIAC2": "受灾面积（类似）",
#                            "ASAC": "成灾面积",
#                            "ATAC": "绝收面积", "AHC": "倒塌房屋", "AHC2": "损坏房屋", "AE": "经济损失", "AWD": "积水深度", "ASD": "积雪深度"}
# v_k = {}
# for k in k_v.keys():
#     v_k[k_v[k]] = k
#
# print(v_k)

miss_points = [('受灾面积（类似）', '500亩', '据村里估计,这次遭受冻害的葡萄应该在500亩以上,”宋女士无奈地说 '),
('受灾面积（类似）', '65.4万亩', '截止3月底,全市大春烂秧死苗总面积65.4万亩,占该市春播总面积的3.4%'),
('倒塌房屋', '15户53间', '此次灾害还造成广西75.13千公顷农作物受灾,居民住房15户53间倒塌,损坏房屋6337间'),
('受灾面积（类似）', '420万亩', '据农业部门统计,江西全省果树受冻面积420万亩,预计减产40万吨'),
('受灾面积（类似）', '十五万亩', '冻害使南丰蜜桔树遭受严重损失,桔树受冻面积达十五万亩,占总面积的百分之三十以上'),
('受灾面积（类似）', '260万亩', '陕西省1000万亩果园受旱,4月份严重的倒春寒进一步导致260万亩苹果受冻,坐果率降低'),
('受灾面积（类似）', '7万亩', '受持续低温阴雨的影响,该区7万亩小麦成熟期比去年推迟约20天,18万亩马铃薯发生晚疫病'),
('受灾面积（类似）', '20万多亩', '该市大部分茶园春茶受冻,面积达20万多亩,预计经济损失1.5亿元人民币'),
('受灾面积（类似）', '15万亩', '该市20.5万亩茶园中,不同程度受冻达15万亩,预计春茶减产2000吨'),
('受灾面积（类似）', '8000多亩', '10300亩大棚西瓜,8000多亩棉花被砸,据悉'),
('倒塌房屋', '65户78间', '这次冰雹灾害共造成32016人需紧急生活救助,倒塌房屋65户78间,损坏房屋14户48间'),
('倒塌房屋', '62户73间', '毁坏耕地面积1102公顷,倒塌居民房屋62户73间,造成直接经济损失约4801万元'),
('经济损失', '224万元', '致使15个行政村的2914户12640人受灾,直接经济损失达224万元,据进一步核实'),
('经济损失', '一亿三千多万元', '农作物受灾面积近4500公顷,造成直接经济损失一亿三千多万元,风雹灾害还造成部分乡镇供电'),
('受灾人口', '25559人', '受灾人口7899户,25559人,农作物受灾面积24686亩'),
('经济损失', '100万', '黄土矿镇石溪村稻田受灾面积达600多亩,估计受损金额有100万左右,由于受损的稻谷还未完全成熟'),
('受灾人口', '156880人', '秦安县4县区20个乡镇,246个村34530户156880人受灾,农作物受灾面积19004.021公顷'),
('损坏房屋', '1700余间', '100余间房屋倒塌,1700余间损坏,农作物受灾面积27.6千公顷'),
('损坏房屋', '600间', '4人溺水死亡,近600间房屋损坏,农作物受灾面积27.9千公顷'),
('经济损失', '七', '从来没见过这样的冰雹,经济损失约七,八十万元'),
('经济损失', '八十万元', '经济损失约七,八十万元,'),
('受灾人口', '44万', '灌南等10个县区受灾,受灾人口44万多,因灾紧急转移安置60697人'),
('损坏房屋', '200余间', '200余间房屋倒塌,200余间不同程度损坏,农作物受灾面积6.3千公顷'),
('受灾面积（类似）', '7.5万亩', '全部受灾,7.5万亩农田被淹绝产,冲毁'),
('受灾面积（类似）', '30万亩', '预计全县94万亩耕地中有35万亩绝产,30万亩减产5成以上,全冲毁房屋1158间'),
('绝收面积', '35万亩', '淹没果菜棚室3.2万个,预计全县94万亩耕地中有35万亩绝产,30万亩减产5成以上'),
('倒塌房屋', '1158间', '30万亩减产5成以上,全冲毁房屋1158间,受灾人数达40万人'),
('损坏房屋', '3000余间', '500余间房屋倒塌,3000余间不同程度损坏,直接经济损失42.2亿元'),
('损坏房屋', '2900余间', '500余间房屋倒塌,2900余间不同程度损坏,农作物受灾面积189.6千公顷'),
('受灾面积（类似）', '169公顷', '高雄市损失4万元,农作物被害面积169公顷,损害程度23%'),
('损坏房屋', '3000间', '近300间房屋倒塌,近3000间不同程度损坏,农作物受灾面积46.5千公顷'),
('经济损失', '75956.68万元人民币', '据初步统计,今年第6号台风“米克拉”袭击致漳州全市直接经济总损失75956.68万元人民币,\n'),
('受伤人口', '72伤', '根据台当局灾害应变中心数据显示,至今已造成2死72伤,并有1836件路树'),
('死亡人口', '2死', '根据台当局灾害应变中心数据显示,至今已造成2死72伤,并有1836件路树'),
('死亡人口', '1名', '其中,花莲县7日1名男子在七星潭落水身亡,同日在连江县则有1名军人在海岸失踪'),
('死亡人口', '1名', '其中,花莲县7日1名男子在七星潭落水身亡,同日在连江县则有1名军人在海岸失踪'),
('损坏房屋', '400多间', '台风“卡努”还造成上述5省区近100间房屋倒塌,400多间不同程度损坏,农作物受灾面积54.1千公顷'),
('损坏房屋', '200余间', '4900余人紧急转移,台风“苏力”还至江西省200余间房屋倒塌,200余间不同程度损坏'),
('损坏房屋', '200余间', '台风“苏力”还至江西省200余间房屋倒塌,200余间不同程度损坏,农作物受灾面积6.3千公顷'),
('受灾面积（类似）', '7319公顷', '其中农产品损失金额3.6964亿元,农作物被害面积7319公顷,损害程度22%'),
('受灾面积（类似）', '14000多公顷', '“农委会”统计,这次农作物被害面积达14000多公顷,梨子的最新农损金额飙到1.6亿元'),
('经济损失', '8.55亿元', '截至8月12日下午6点,初步统计受损8.55亿元,无人员伤亡'),
('受灾面积（类似）', '9963公顷', '“尼伯特”台风造成台湾农产损失估计为8亿3821万元,农作物被害面积9963公顷,损害程度28%'),
('死亡人口', '3人', '至昨晚发稿时止,梅州市有3人在水灾中死亡,2人失踪'),
('受灾人口', '20多万人', '目前灾害已造成6人受伤5人死亡,20多万人受灾,受灾地政府已紧急转移安置10656人'),
('受伤人口', '22名', '将全力搜救决不放弃,在医院进行救治的22名受伤人员体征平稳,受灾群众除投亲靠友安置外'),
('转移人口', '250多人', '将长途旅客送到巴东交界处转运,目前已转运旅客250多人,剩余50多人均被妥善安置'),
('倒塌房屋', '两栋', '益阳市赫山区龙光桥镇马头村李家坪组斗笠仑山发生山体滑坡自然灾害,两栋共计600多平方米房屋被滑坡泥石冲跨,5名村民被埋'),
('死亡人口', '3人', '宜宾市屏山县新安镇等地突发泥石流,事故已造成3人身亡1人失踪,初步估计本次灾害造成直接经济损失过千万'),
('倒塌房屋', '两栋', '一名老人失踪,两栋民房被冲毁,307省道新安镇境内多处中断'),
('死亡人口', '4具', '经过救援人员搜寻,已找到4具遇难者遗体,另有10人失踪'),
('损坏房屋', '6户', '造成800余人受灾,农户住房受损约6户,其中全部被掩埋3户'),
('死亡人口', '5人', '据云南省临沧市镇康县政府新闻办20日通报,19日凌晨该县木场乡因大面积山体滑坡失联的5人,经全力搜救已全部找到'),
('死亡人口', '5人', '1人自救成功,5人失联,另有13幢民房不同程度受损'),
('受灾人口', '46人', '云南省镇雄县果珠乡高坡村发生山体滑坡,14户46人被掩埋,其中男27人'),
('受伤人口', '2名', '镇雄山体滑坡共导致42人遇难,2名伤员已经脱离生命危险,2名伤员不再被埋人员之列'),
('受伤人口', '2名', '镇雄山体滑坡共导致42人遇难,2名伤员已经脱离生命危险,2名伤员不再被埋人员之列'),
('受伤人口', '2名', '2名伤员已经脱离生命危险,2名伤员不再被埋人员之列,目前'),
('受伤人口', '2名', '2名伤员已经脱离生命危险,2名伤员不再被埋人员之列,目前'),
('死亡人口', '6名', '截至30日10时55分,建德山体滑坡6名失联人员全部找到,经医院确认均已无生命体征'),
('倒塌房屋', '3户', '该市新安江街道丰产村横路自然村发生山体滑坡,导致3户房屋倒塌,6人失联'),
('失踪人口', '27名', '截至10月13日21时,27名失联人员中已搜救出21人,确认均无生命迹象'),
('失踪人口', '6人', '确认均无生命迹象,仍有6人失联,搜救工作仍在全力推进'),
('死亡人口', '23人', '截至10月16日16时,遂昌苏村山体滑坡现场共搜救出23人,确认均无生命迹象'),
('死亡人口', '21人', '截至10月13日21时,27名失联人员中已搜救出21人,确认均无生命迹象'),
('死亡人口', '6位', '目前,6位遇难者遗体已全部找到,今年第17号台风“鲇鱼”于9月28日4时40分前后在福建省泉州市惠安县沿海登陆'),
('倒塌房屋', '6间', '9月28日文成县双桂乡宝丰村发生山洪灾害引发山体滑坡,导致6间居民房被冲毁,6人失踪'),
('积水深度', '半米', '聊城市区前天下午3个小时的降水达到58毫米,部分道路积水有半米,\n'),
('失踪人口', '1人', '武侯区倒塌房屋12间,新都区1人不慎落水失踪,\xa0\xa0\xa0\xa07月18日20时到7月19日8时'),
('失踪人口', '1人', '27万多人受灾,其中1人死亡,\xa0\xa0\xa0\xa07月6日晚至7日'),
('受灾人口', '一千多名', '更是被洪水围困成“孤岛”,一千多名村民无法出行,7名村民失联'),
('受灾人口', '600多名', '同时记者也了解到,今天凌晨在邢台开发区东华路和七里河交叉口北岸的王快村等几个村庄有600多名村民被大水围困,邢台武警支队抽调68名官兵在今天凌晨5点左右已经到达区域附近'),
('积水深度', '齐腰', '19号,邯郸市区部分地段水深齐腰,积水最深地段甚至达到约1.6米'),
('积水深度', '1.6米', '邯郸市区部分地段水深齐腰,积水最深地段甚至达到约1.6米,而磁县南王庄村'),
('积水深度', '齐腰', '但是像内涝比较严重的邢台,邯郸两地不少地段积水已经齐腰身,公交车已经泡在水中'),
('积水深度', '齐腰身', '但是像内涝比较严重的邢台,邯郸两地不少地段积水已经齐腰身,公交车已经泡在水中'),
('受灾人口', '一千多名', '更是被洪水围困成“孤岛”,一千多名村民无法出行,7名村民失联'),
('受灾人口', '600多名', '\xad同时记者也了解到,今天凌晨在邢台开发区东华路和七里河交叉口北岸的王快村等几个村庄有600多名村民被大水围困,邢台武警支队抽调68名官兵在今天凌晨5点左右已经到达区域附近'),
('积水深度', '齐腰', '19号,邯郸市区部分地段水深齐腰,积水最深地段甚至达到约1.6米'),
('积水深度', '1.6米', '邯郸市区部分地段水深齐腰,积水最深地段甚至达到约1.6米,而磁县南王庄村'),
('损坏房屋', '800余栋', '导致7个乡镇一度停电,城市内涝导致桑植县桥自弯镇800余栋房屋受到不同程度的损坏,860余亩农作物受损'),
('积水深度', '20多厘米', '位于菏泽城区人民南路大屯立交桥两侧积水严重,最深处有20多厘米,导致市民出行受阻'),
('积水深度', '20多厘米', '积水路段百余米,最深处有20多厘米,\n'),
('积水深度', '半米', '多条街面被淹,水深达半米,当日上午'),
('积水深度', '半米', '多条街面被淹,水深达半米,当地多条积水严重的路段随处闪现着消防抢险救援官兵的身影'),
('受灾人口', '150人', '参战官兵380余人,救助遇险群众150人,疏散转移群众320人'),
('积水深度', '1.4米', '28名驾校学员被困其中,消防官兵在深达1.4米的水中前行数百米,历经1个半小时成功解救所有被困人员'),
('受伤人口', '32人', '24日当地发生的一起客车交通事故致4人死亡,32人入院检查治疗,据警方通报'),
('受伤人口', '32人', '该起事故共造成4人死亡,32人先后到院治疗和检查,入院者均无生命危险'),
('死亡人口', '2死', '据称,该起事故目前已造成2死30余人受伤,事故现场车辆拥堵2公里左右'),
('受灾人口', '40余名', '另据消防部门介绍,消防官兵先后救出疏散40余名被困群众,其中包括7名儿童和5名老人'),
('受灾面积（类似）', '20万亩', '这场突如其来的大风给华北最大蔬菜基地永年县90%的蔬菜生产造成重大影响,全县20万亩蔬菜大棚的塑料薄膜,竹竿被损毁'),
('损坏房屋', '79间', '该镇镇区及8条行政村受到严重影响,损毁房屋瓦面累计79间,农作物受灾8120亩'),
('受灾人口', '10万余人', '潮水沿河道上溯50公里,受灾人口10万余人,27个村庄大面积进水'),
('受灾人口', '100名', '在通路工程距岸9公里,12.5公里和16公里处分别有100名,40名'),
('受灾人口', '40名', '12.5公里和16公里处分别有100名,40名,300名民工被困'),
('受灾人口', '300名', '40名,300名民工被困,唐山市委'),
('受灾面积（类似）', '40余万亩', '淹没盐场50座,淹没农田40余万亩,8万多棵树被刮倒'),
('损坏房屋', '168处', '据不完全统计,大约有168处房屋及部分厂区受损,'),
('受灾面积（类似）', '4.8万亩', '损毁厂房8幢,损坏大棚面积4.8万亩,射阳县房屋受损'),
('损坏房屋', '8004户28104间', '医院收治伤员846人,阜宁县倒塌损坏房屋8004户28104间,2所小学房屋受损'),
('损坏房屋', '615户', '射阳县房屋受损,倒塌615户,记者今天从当地民政部门获悉'),
('受灾人口', '1500多位', '供电部分中断,1500多位村民在风灾中遭受经济损失,塘湖村村民郭卖龙心有余悸地回忆龙卷风来袭的场景：“昨天傍晚'),
('受灾面积', '1530.3公顷', '倒塌房屋91间,农作物受灾面积全市合计1530.3公顷,农作物绝收面积253.3公顷'),
('受灾人口', '44户', '据初步统计,当地8月9日发生的龙卷风袭击共造成44户牧民受灾,初核经济损失近1800万元'),
('受灾人口', '44户', '据初步统计,44户牧民房屋或车辆不同程度受损,旅游接待点倒毁或严重损坏的蒙古包182顶'),
('受灾人口', '6万人', '科尔沁左翼中旗和开鲁县等地相继遭受了严重龙卷风灾害,使全市6万人,33万亩农田受灾'),
('受灾面积（类似）', '121公顷', '截至2日下午4时屏东县府初步统计称,农作物损失面积共121公顷,损失金额约新台币2000万元'),
('损坏房屋', '3826间', '成灾面积2150亩,房屋损坏共3826间,此次灾害直接经济总损失1660.6万元'),
('受灾面积（类似）', '59.5万亩', '7800头大牲畜饮水困难,59.5万亩农作物受旱,占在田农作物的50%'),
('受灾人口', '109.3万人', '中新网6月9日电 据民政部网站消息,江苏近期严重干旱灾害已致109.3万人生活困难需政府救助,6月9日9时'),
('受灾人口', '109.3万人', '截至6月7日14时统计,近期严重干旱灾害已造成生活困难需政府救助人口109.3万人,农作物受灾面积617.6千公顷'),
('受灾人口', '3.22万人', '4806户,3.22万人,1.1万头大牲畜饮水困难'),
('受灾人口', '94.5万人', '据省水利厅截至3月20日统计数据显示,旱情造成94.5万人,22.7万头大牲畜饮水困难'),
('受灾人口', '24.47万人', '全市农作物受旱面积达到56.5万亩,24.47万人出现临时饮水困难,根据气象监测统计'),
('受灾人口', '8.4万人', '遭受自1957年有气象记载以来最严重的旱灾,该区28个镇街的8.4万人,3.5万头大牲畜饮水困难'),
('转移人口', '8000多名', '民政兜底等方式,共安置受灾群众8000多名,随着洪水消退'),
('死亡人口', '38人', '暴雨洪涝重灾区河北井陉县33万人中有20.8万人受灾,死亡38人,失踪33人'),
('死亡人口', '7人', '另外,在井陉县境内修高速公路的中铁十一局和中建一局外地施工人员死亡7人,失踪22人'),
('受灾人口', '743.3万人', '辛集市受灾,受灾人口743.3万人,因灾死亡36人'),
('损坏房屋', '808户1951间', '倒塌房屋249户916间,严重损房808户1951间,一般损房2530户5992间'),
('损坏房屋', '2530户5992间', '严重损房808户1951间,一般损房2530户5992间,农作物受灾面积42.5千公顷'),
('损坏房屋', '1800余间', '1200余间房屋倒塌,1800余间不同程度受损,农作物受灾面积48.3千公顷'),
('损坏房屋', '1600余间', '300余间房屋倒塌,1600余间不同程度受损,农作物受灾面积6千公顷'),
('经济损失', '421.2亿元', '损坏房屋56.8万间,因灾直接经济损失421.2亿元,受洪涝灾害影响'),
('受灾面积（类似）', '600多亩', '因遭受狂风暴雨袭击,全乡600多亩早玉米被刮到,由于目前正是早玉米打包结实的季节'),
('倒塌房屋', '1988户', '农作物受灾面积10.32万公顷,农房倒塌1988户,5089间'),
('倒塌房屋', '5089间', '农房倒塌1988户,5089间,农房严重损坏3379户'),
('损坏房屋', '3379户', '5089间,农房严重损坏3379户,9151间'),
('损坏房屋', '9151间', '农房严重损坏3379户,9151间,农房一般损坏23572户'),
('损坏房屋', '23572户', '9151间,农房一般损坏23572户,80719间'),
('损坏房屋', '80719间', '农房一般损坏23572户,80719间,直接经济损失71.92亿元人民币'),
('受灾人口', '4835人', ',共计威胁1186户4835人,威胁财产31552万元'),
('经济损失', '31552万元', '共计威胁1186户4835人,威胁财产31552万元,\n'),
('受灾人口', '70户', '霍城县消防大队与村民编组进行抗洪抢险,据新疆霍城县芦草沟镇政府初步统计：强降雨已造成芦草沟镇180户居民中约70户受灾,暴雨和洪涝灾害造成的损失约27万'),
('倒塌房屋', '五户', '村里共17户人家,有五户人家院墙倒塌,10户人家培育树苗的苗床被淹'),
('积水深度', '2米', '沿途洪水已没过膝盖,最深处近2米深,\n'),
('受灾人口', '2.5万人', '房屋受损250间,灾害影响2.5万人,尚无人员伤亡'),
('受灾面积（类似）', '340公顷', '据调查,大火蔓延面积340公顷,其中有林地面积20公顷'),
('转移人口', '2580人', '目前,当地已疏散群众2580人,安置100多人'),
('倒塌房屋', '1户', '大雪已造成该乡3000多亩竹林受灾,1户村民房屋被压垮,公路和电力已中断'),
('积雪深度', '38厘米', '石家庄遭遇1955年以来最大降雪,积雪深度达到38厘米,暴雪来袭'),
('积雪深度', '34.8厘米', '其中石家庄更是遭遇1955年以来最大降雪,正定积雪深度达34.8厘米,井陉达33厘米'),
('积雪深度', '33厘米', '正定积雪深度达34.8厘米,井陉达33厘米,石家庄市气象局于10日16时发布暴雪红色预警信号和道路结冰黄色预警信号'),
('失踪人口', '10余名', '截至27日下午2时,仍有10余名群众没找到,救援人员还在全力搜救'),
('积雪深度', '7厘米', '雨雪交加,局地雪深超过7厘米,给交通出行带来一定影响'),
('积雪深度', '7.5厘米', '达8.3毫米,雪深7.5厘米,雨雪交加'),
('受灾人口', '460名', '造1名人员死亡,108户460名牧民及2.2万头,只'),
('受灾人口', '300名', '两架救援直升机计划明天飞往巴音郭楞蒙古自治州,救援和静县巩乃斯乡雪崩区域约300名被困农牧民, 今天'),
('受灾人口', '957余名', '终于在10日14时25分开始恢复双向通车,期间被风雪围困的957余名过往旅客也被额敏公路分局的工作人员及公安,消防'),
('受灾人口', '957余名', '历经20个小时的昼夜奋力连续抢险紧急救援,将在风区受尽煎熬的957余名被困旅客全部成功救出,风区内无人员滞留'),
('受灾人口', '一百五十多万人', '    根据新疆民政厅最新的统计,持续的灾情已造成新疆一百五十多万人,次'),
('受灾人口', '9名', '国道218线伊犁地区新源县巩乃斯山区路段发生连续性雪崩,9名回家民工弃车逃生,生命危在眉睫'),
('受灾人口', '9名', '突然接到新源县那拉提镇派出所的电话：巩乃斯山区路段发生连续性雪崩,有9名民工弃车徒步逃生被困,请求公路段立即派人员和机械前往救助'),
('受灾人口', '9名', '直到早晨7时30分终于打通一条通道,救出了饥寒交迫还在雪中挣扎的所有9名民工,在巴音郭楞蒙古自治州巩乃斯林场打工的一名要回家过年的河南藉汉族'),
('积雪深度', '十几米', '因雪崩面积较大,积雪平均厚度达十几米,救援人员靠机械和人力清除积雪需要三至四天'),
('受伤人口', '6伤', '8个村民倒在了田坎上,导致2死6伤,\n'),
('死亡人口', '3人', '经检查证实,3人均是遭雷击而死,\n'),
('死亡人口', '一', '今年3月29日,东方市感城镇尧文一村民朱某遭雷击身亡,该事故引起了气象等部门高度重视'),
('死亡人口', '一', '2013年3月29日上午,东方市感城镇尧文村发生一起雷击事件,一村民兼护林员被雷击死亡'),
('死亡人口', '一', '东方市感城镇尧文村发生一起雷击事件,一村民兼护林员被雷击死亡,另一村民遭雷击重伤'),
('死亡人口', '一', '一村民兼护林员被雷击死亡,另一村民遭雷击重伤,接到雷击事故爆料后'),
('死亡人口', '一', '兼护林员,和李姓村民跑到处于开阔农用地的一座四角亭下的小屋内避雨,雷击发生后'),
('死亡人口', '一', '由于未安装防雷装置,雷击造成亭西面的二个角一个被击掉,一个被击碎'),
('死亡人口', '一', '雷击造成亭西面的二个角一个被击掉,一个被击碎,在调查中发现'),
('死亡人口', '一', '侥幸没有人员伤亡,所以一直没有上报,但3月29日'),
('死亡人口', '一', '虽然雷电发生在某个地方具有偶然性,但一个地方多次被雷电击中,说明其具备适当的雷电发生条件'),
('死亡人口', '一', '以致酿成大祸,不能不说是一次深刻的教训,对此'),
('死亡人口', '一', '气象部门提醒,第一,对于乡村野外空旷场所的建'),
('受伤人口', '一重伤', '郑口镇一村干部冒着大雨在接五保户途中被雷击中,造成一死一重伤,\n'),
('死亡人口', '一死', '郑口镇一村干部冒着大雨在接五保户途中被雷击中,造成一死一重伤,\n'),
('死亡人口', '2名', '吉林省磐石市和舒兰市两地接连发生雷击致死事件,2名农妇在地里劳作时,遭雷击死亡'),
('受伤人口', '一伤', '沧州发生雷灾八起,造成三死一伤,同时造成较大财产损失'),
('受伤人口', '一伤', '沧州发生雷灾八起,造成三死一伤,同时造成较大财产损失'),
('受伤人口', '一伤', '黄骅市办盐场遭雷击,正在拉塑苫的工人一死一伤,7月10日下午'),
('受伤人口', '一伤', '黄骅市办盐场遭雷击,正在拉塑苫的工人一死一伤,7月10日下午'),
('受灾面积', '三死', '沧州发生雷灾八起,造成三死一伤,同时造成较大财产损失'),
('受灾面积', '一死', '黄骅市办盐场遭雷击,正在拉塑苫的工人一死一伤,7月10日下午'),
('受伤人口', '16人', '死22人,伤16人,牲畜死亡200多'),
('受伤人口', '3伤', '造成人员8死,3伤,\n'),
('死亡人口', '22人', '雷电灾害造成的直接经济损失1520万元,死22人,伤16人'),
('死亡人口', '8死', '直接经济损失651万元,造成人员8死,3伤'),
('死亡人口', '一男子', '施秉县牛大场镇发生一起雷击事件,一男子被雷击身亡,记者昨日从黔东南州防雷办获悉'),
('受伤人口', '三伤', '据安义县委宣传部消息,事故已造成一死三伤,其中一位市民被雷击中后当场死亡'),
('死亡人口', '一死', '据安义县委宣传部消息,事故已造成一死三伤,其中一位市民被雷击中后当场死亡'),
('受伤人口', '一伤', '不幸遭到雷击,事故造成一死一伤,据新京报消息'),
('死亡人口', '一死', '不幸遭到雷击,事故造成一死一伤,据新京报消息'),
('死亡人口', '3000—4000人', '官方数据显示,我国每年因雷击造成的伤亡一直持续在3000—4000人,夏季究竟该如何避免雷击伤害'),
('死亡人口', '22人', '去年浙江共发生地闪将近54万次,雷电灾害造成22人死亡,其中21人在农村'),
('死亡人口', '22人', '2010年浙江省共发生雷电灾害2217起,雷击造成人员死亡22人,受伤21人'),
('死亡人口', '22人', '发生在农村人员伤亡事件最为严重,去年因为雷击死亡的22人中有21人都是在农村,防雷专家分析')]

v_k = {'受灾人口': 'AIMP', '失踪人口': 'AMP', '受伤人口': 'AINP', '转移人口': 'ATP', '死亡人口': 'ADP', '受灾面积': 'AIAC', '受灾面积（类似）': 'AIAC2', '成灾面积': 'ASAC', '绝收面积': 'ATAC', '倒塌房屋': 'AHC', '损坏房屋': 'AHC2', '经济损失': 'AE', '积水深度': 'AWD', '积雪深度': 'ASD'}

for mp in miss_points:
    type = mp[0]
    result = mp[1]
    lines = mp[2].split(',')
    pattern = pattern_dict[v_k[type]]
    ans = re.search(pattern, lines[1])
    if ans == None:
        print(type,'\t',result,'\t',lines,'\t',None)
        pass
    else:
        # ans = ans.group()
        # num = extract_num_from_str(ans)
        # print(type, '\t', result, '\t', lines[1], '\t', ans, num)
        pass