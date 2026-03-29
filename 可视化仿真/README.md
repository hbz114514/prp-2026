##
使用方法：先启动pointing_generate.py（本地API）\
再打开test.html，需要下载live server插件，点击go live即可。\
（做数据处理的时候数据直接从api处得到即可，注意参数，数据处理的部分最好也做成api，前端再写一个modal显示预测值。api中误差矩阵纯随机，所以有的时候可能效果不是很好.除此之外，three.js的全局坐标轴定义和j2k有不同，使用的时候要注意角度换算，这一点在py文件和html文件中都有体现）
##
PS:本文件中3d部分由deepseekv3辅助完成，仅用于prp项目可视化模拟。