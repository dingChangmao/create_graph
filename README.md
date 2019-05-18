# create_graph
Generate calculation graph based on tensorflow


开发原因：由于需要将机器学习过程中的流程图实时展示在平台上开始了对tensorboard的研究，研究过程中发现根据tensorflow生成的log文件解析之后前端无法快速
根据数据画图，项目耗时较长，于是要后端直接画图展示给前端，经研究模拟tensorboard得出了两种画graph流程图的方法



方法一：利用linux命令启动tensorboard ，采用python爬虫的优良机制利用selenium，截取图片，保存在hdfs直接把图片供前端使用。
优点：快，准，狠，就是官方图片，效果美美哒
缺点：图片死板，虽然美观，效果ok


方法二：解析tensorflow计算图的数据，根据数据生成利于graphviz（一种画计算图的软件）展示的数据，通过对tf数据的读取，达到实时展示graph的功能。
优点：自己画图，成就感？？？，随意选择图片样式，随意选择展示深度（都是你写的数据想怎么玩就怎么玩）
缺点：需要安装graphviz  ：sudo apt-get install graphviz   如果需要独立部署要解决依赖问题。


方法三：找一个优秀的前端根据你给出的数据画图（tensorboard可以对同样的数据进行画图，利用ts等），同理找一个有耐心研究一下源码的好同事是极其重要的。


补充：本项目多方面引用就不标注了，只是自己的一些集成开发，仅供参考，小白初试，欢迎指点
