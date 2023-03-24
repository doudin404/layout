# layout
layout是我的毕设
用法:打开layout中的Publaynet放入数据集,然后运行layout_blt的中的main.py进行训练.
然后将生成的模型存档命名为layout.pth放入layout_blt/save中,最后运行layout/MainWindow.py就能用了

layout_blt original 是原版BLT的代码 layout_blt_new 是我优化了一些的代码,没有改变主要内容
用法:需要在外面的PubLayNet中放入数据集.然后大概不能直接运行main.py.
先把main.py里的参数--load修改为false.然后就能训练出模型.
然后再把生成放入的模型存档放进save中,然后到main.py里修改--load_name为存档名字,--load修改为true.就能直接进行测试.
