# coding:utf-8



from time import sleep
from PIL import Image
from selenium import webdriver
driver = webdriver.Chrome()

#  需要启动tensorboard （已经生成了events日志文件） tensorboard --logdir logpath

# driver.get(url)    url 为本机localhost  端口可以重新指定   需要掌握使用tensorboard（最起码要会启动一下）
driver.get("http://ding:6006/#graphs")



sleep(3)
driver.save_screenshot('graphs.png')  # 截取当前页面全图
element = driver.find_element_by_id("root")  # 百度一下的按钮
print("获取元素坐标：")
location = element.location
print(location)

print("获取元素大小：")
size = element.size
print(size)

# 计算出元素上、下、左、右 位置
left = element.location['x']
print(left)
top = element.location['y']
print(top)
right = element.location['x'] + element.size['width']
print(right)
bottom = element.location['y'] + element.size['height']
print (bottom)

im = Image.open('graphs.png')
im = im.crop((left, top, right, bottom))
im.save('graphs_1.png')

driver.close()