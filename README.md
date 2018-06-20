![](https://raw.githubusercontent.com/Tencent/ncnn/master/images/256-ncnn.png)
# ncnn

[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://raw.githubusercontent.com/Tencent/ncnn/master/LICENSE.txt) 
[![Build Status](https://travis-ci.org/Tencent/ncnn.svg?branch=master)](https://travis-ci.org/Tencent/ncnn)


ncnn is a high-performance neural network inference computing framework optimized for mobile platforms. ncnn is deeply considerate about deployment and uses on mobile phones from the beginning of design. ncnn does not have third party dependencies, it is cross-platform, and runs faster than all known open source frameworks on mobile phone cpu. Developers can easily deploy deep learning algorithm models to the mobile platform by using efficient ncnn implementation, create intelligent APPs, and bring the artificial intelligence to your fingertips. ncnn is currently being used in many Tencent applications, such as QQ, Qzone, WeChat, Pitu and so on.

ncnn 是一个为手机端极致优化的高性能神经网络前向计算框架。ncnn 从设计之初深刻考虑手机端的部署和使用。无第三方依赖，跨平台，手机端 cpu 的速度快于目前所有已知的开源框架。基于 ncnn，开发者能够将深度学习算法轻松移植到手机端高效执行，开发出人工智能 APP，将 AI 带到你的指尖。ncnn 目前已在腾讯多款应用中使用，如 QQ，Qzone，微信，天天P图等。

---

### HowTo

[how to build ncnn library](https://github.com/Tencent/ncnn/wiki/how-to-build)

[how to use ncnn with alexnet](https://github.com/Tencent/ncnn/wiki/how-to-use-ncnn-with-alexnet)

[ncnn 组件使用指北 alexnet](https://github.com/Tencent/ncnn/wiki/ncnn-%E7%BB%84%E4%BB%B6%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8C%97-alexnet)

[ncnn low-level operation api](https://github.com/Tencent/ncnn/wiki/low-level-operation-api)

[ncnn param and model file spec](https://github.com/Tencent/ncnn/wiki/param-and-model-file-structure)

[ncnn operation param weight table](https://github.com/Tencent/ncnn/wiki/operation-param-weight-table)

[how to implement custom layer step by step](https://github.com/Tencent/ncnn/wiki/how-to-implement-custom-layer-step-by-step)

---

### FAQ

[ncnn throw error](https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-throw-error)

[ncnn produce wrong result](https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-produce-wrong-result)

---

### Features

* Supports convolution neural networks, supports multiple input and multi-branch structure, can calculate part of the branch
* No third-party library dependencies, does not rely on BLAS / NNPACK or any other computing framework
* Pure C ++ implementation, cross-platform, supports android, ios and so on
* ARM NEON assembly level of careful optimization, calculation speed is extremely high
* Sophisticated memory management and data structure design, very low memory footprint
* Supports multi-core parallel computing acceleration, ARM big.LITTLE cpu scheduling optimization
* The overall library size is less than 500K, and can be easily reduced to less than 300K
* Extensible model design, supports 8bit quantization and half-precision floating point storage, can import caffe/pytorch/mxnet/onnx models
* Support direct memory zero copy reference load network model
* Can be registered with custom layer implementation and extended
* Well, it is strong, not afraid of being stuffed with 卷   QvQ

### 功能概述

* 支持卷积神经网络，支持多输入和多分支结构，可计算部分分支

      ncnn 支持卷积神经网络结构，以及多分支多输入的复杂网络结构，如主流的 vgg、googlenet、resnet、squeezenet 等。
      计算时可以依据需求，先计算公共部分和 prob 分支，待 prob 结果超过阈值后，再计算 bbox 分支。
      如果 prob 低于阈值，则可以不计算 bbox 分支，减少计算量。
      
* 无任何第三方库依赖，不依赖 BLAS/NNPACK 等计算框架

      ncnn 不依赖任何第三方库，完全独立实现所有计算过程，不需要 BLAS/NNPACK 等数学计算库。
      
      
* 纯 C++ 实现，跨平台，支持 android ios 等

      ncnn 代码全部使用 C/C++ 实现，以及跨平台的 cmake 编译系统，
      可在已知的绝大多数平台编译运行，如 Linux，Windows，MacOS，Android，iOS 等。

      由于 ncnn 不依赖第三方库，且采用 C++ 03 标准实现，
      只用到了 std::vector 和 std::string 两个 STL 模板，
      可轻松移植到其他系统和设备上。

* ARM NEON 汇编级良心优化，计算速度极快

      ncnn 为手机端 CPU 运行做了深度细致的优化，使用 ARM NEON 指令集实现卷积层，全连接层，池化层等大部分 CNN 关键层。
      对于寄存器压力较大的 armv7 架构，我们手工编写 neon 汇编，内存预对齐，cache 预缓存，
      排列流水线，充分利用一切硬件资源，防止编译器意外负优化。
      测试手机为 Nexus 6p，Android 7.1.2

* 精细的内存管理和数据结构设计，内存占用极低
      
      在 ncnn 设计之初我们已考虑到手机上内存的使用限制，在卷积层、全连接层等计算量较大的层实现中，
      没有采用通常框架中的 im2col + 矩阵乘法，因为这种方式会构造出非常大的矩阵，消耗大量内存。
      因此，ncnn 采用原始的滑动窗口卷积实现，并在此基础上进行优化，大幅节省了内存。
      在前向网络计算过程中，ncnn 可自动释放中间结果所占用的内存，进一步减少内存占用。
      内存占用量使用 top 工具的 RSS 项统计，测试手机为 Nexus 6p，Android 7.1.2。
      
* 支持多核并行计算加速，ARM big.LITTLE cpu 调度优化

      ncnn 提供了基于 openmp 的多核心并行计算加速，在多核心 CPU 上启用后能够获得很高的加速收益。
      ncnn 提供线程数控制接口，可以针对每个运行实例分别调控，满足不同场景的需求。
      针对 ARM big.LITTLE 架构的手机 CPU，ncnn 提供了更精细的调度策略控制功能，
      能够指定使用大核心或者小核心，或者一起使用，获得极限性能和耗电发热之间的平衡。
      例如，只使用1个小核心，或只使用2个小核心，或只使用2个大核心，都尽在掌控之中。

* 整体库体积小于 500K，并可轻松精简到小于 300K

      ncnn 自身没有依赖项，且体积很小，默认编译选项下的库体积小于 500K，能够有效减轻手机 APP 安装包大小负担。
      此外，ncnn 在编译时可自定义是否需要文件加载和字符串输出功能，
      还可自定义去除不需要的层实现，轻松精简到小于 300K。

* 可扩展的模型设计，支持 8bit 量化和半精度浮点存储，可导入 caffe/pytorch/mxnet/onnx 模型

      ncnn 使用自有的模型格式，模型主要存储模型中各层的权重值。
      ncnn 模型中含有扩展字段，用于兼容不同权重值的存储方式，
      如常规的单精度浮点，
      以及占用更小的半精度浮点和 
      8bit 量化数。
      大部分深度模型都可以采用半精度浮点减小一半的模型体积，
      减少 APP 安装包大小和在线下载模型的耗时。
      ncnn 带有 caffe 模型转换器，可以转换为 ncnn 的模型格式，方便研究成果快速落地。
      

* 支持直接内存零拷贝引用加载网络模型

      在某些特定应用场景中，如因平台层 API 只能以内存形式访问模型资源，
      或者希望将模型本身作为静态数据写在代码里，
      ncnn 提供了直接从内存引用方式加载。

* 可注册自定义层实现并扩展

      ncnn 提供了注册自定义层实现的扩展方式，可以将自己实现的特殊层内嵌到 ncnn 的前向计算过程中，
      组合出更自由的网络结构和更强大的特性。
      
      
* 恩，很强就是了，不怕被塞卷 QvQ

---

### Example project

https://github.com/Tencent/ncnn/tree/master/examples/squeezencnn

### 技术交流QQ群：637093648  答案：卷卷卷卷卷

---

### License

BSD 3 Clause

