# yolov5_v3.1及其trt部署


## 工作环境
	1、nvidia-jetson 系列，有jetpack
	2、nvidia-gpu 系列需要自己安装 cuda 等



## 使用方法

	1、yolov5-3.1 目录是模型训练的目录，在这里训练 pt 模型，和 gen_wts.py 转换为 wts 文件
	
	2、YoLov5-TensorRT-NMS-master 目录是生成 engine 的工程，方法是：
		a. 将 yolov5s.wts 放在 YoLov5-TensorRT-NMS-master 目录下执行
		b. 执行下面几个命令
			mkdir build & cd build
		    cmake ..
			make -j6
			./yolov5 -s
		c. 在当前目录会得到一个 yolov5s.engine
		
	3、YoLov5-TensorRT-NMS-zh_plugin 目录测试 engine，方法是：
		a. 将 yolov5s.engine 拷贝到 YoLov5-TensorRT-NMS-zh_plugin 目录下
		b. 执行如下命令：
			mv yolov5s.engine coco80.engine
			mkdir build & cd build
			cmake ..
			make -j6
			./tttt
		c. 会对 test.jpg 进行测试，保存为 cv_image.jpg
		
