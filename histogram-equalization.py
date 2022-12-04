import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_input_img(path,img_name):  # 读入图像
      img = Image.open(path)
      img = np.asarray(img)
      np.savetxt('output/'+img_name.replace('.jpg','').lower()+'-input-img-arr.txt',img,fmt='%s')
      #print('Input Image:\n',img,'\n')
      return img

def flatten_img(img,img_name):  # 通过展平图像数组将像素放入一维数组中
      flat = img.flatten()
      np.savetxt('output/'+img_name.replace('.jpg','').lower()+'-flatten.txt',img,fmt='%s')
      #print('Flatten:\n',flat,'\n')
      return flat

def show_input_histogram(flat,img_name):  # 显示输入图像的直方图
      plt.hist(flat,bins=50)
      plt.title(img_name+': Input Image Histogram')
      plt.show()

def show_hist_graph(flat,bins,img_name):   # 直方图函数 bins=256
      '''
      :param flat : 已展平的一维图像数组
      :param bins : 图像的颜色小区间个数。将图像的颜色空间划分为若干小的颜色区间来计算颜色直方图
      :param img_name 
      :return hist_graph : 直方图函数 
      '''
      hist_graph = np.zeros(bins)   # 长度为256的全零数组
      for pixel in flat:           # 遍历图像中的像素并对像素数进行求和，构造直方图
            hist_graph[pixel] += 1  
      plt.plot(hist_graph)
      plt.title(img_name+': Input Image Histogram Graph')
      plt.show()
      return hist_graph

def cumulative_sum(hist_graph,img_name):   # 累积和函数
      '''
      :param hist_graph : 直方图函数
      :param img_name 
      :return csum : 累积和函数 
      '''
      hist_graph = iter(hist_graph)
      csum = [next(hist_graph)] 
      for bin in hist_graph:
            csum.append(csum[-1] + bin)
      csum = np.array(csum)
      plt.plot(csum)  
      plt.title(img_name+': Input Image Cummulative')
      plt.show()
      return csum

def normalize(csum,img_name):   # 将累积和值归一化到 0-255 之间
      '''
      :param csum : 累积和
      :param img_name 
      :return normalize_csum : 重新归一化的累积和
      '''
      max = csum.max()
      min = csum.min()
      nj = (csum - min) * 255    # 分子
      N = max - min              # 分母
      normalize_csum = nj / N    # 重新归一化累积分布函数 CDF(Cumulative Distribution Function)
      normalize_csum = normalize_csum.astype('uint8')  # 由于图像中不能使用浮点数，因此将其值转换回 uint8
      plt.plot(normalize_csum)
      plt.title(img_name+': Output Image Cummulative')
      plt.show()
      return normalize_csum

def get_new_img(img,flat,normalize_csum,img_name):
      '''
      :param img
      :param flat : 展平图像数组
      :param normalize_csum : 重新归一化的累积和
      :param img_name 
      :return new_img
      '''
      # 从展平图像数组 flat 中每个索引的累积和中获取值，并将其设置为 new_img
      new_img = normalize_csum[flat] 
      plt.hist(new_img, bins=50)   # 分布更均匀的直方图
      plt.title(img_name+': Output Image Histogram')
      plt.show()
      new_img = np.reshape(new_img, img.shape)  # 把数组还原成原来图像的形状
      np.savetxt('output/'+img_name.replace('.jpg','').lower()+'-img-new.txt',new_img,fmt='%s')
      #print('New Image:\n',new_img,'\n')
      return new_img

def plot_figure(img,new_img):   # 展示新旧图像
      fig = plt.figure()
      fig.set_figheight(15)
      fig.set_figwidth(15)
      fig.add_subplot(1,2,1)
      plt.imshow(img, cmap='gray')
      fig.add_subplot(1,2,2)  
      plt.imshow(new_img, cmap='gray')
      plt.show(block=True)

def main(path,img_name):
      img = get_input_img(path,img_name)
      flat = flatten_img(img,img_name)
      show_input_histogram(flat,img_name)
      hist_graph = show_hist_graph(flat,bins=256,img_name=img_name)
      csum = cumulative_sum(hist_graph,img_name)       # 累积和函数
      normalize_csum = normalize(csum,img_name)        # 归一化累积和函数
      new_img = get_new_img(img,flat,normalize_csum,img_name)  
      plot_figure(img,new_img)

if __name__ == '__main__':
      '''import cv2
      image = cv2.imread("image/cat.jpg")
      grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      cv2.imwrite("image/cat.jpg", grayimg)'''
      
      main("image/cat.jpg","Cat.jpg")
      main("image/flower.jpg","Flower.jpg")
      main("image/bridge.jpg","Bridge.jpg")
      main("image/movie.jpg","Movie.jpg")

      






