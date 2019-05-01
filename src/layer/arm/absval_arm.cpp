// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "absval_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

DEFINE_LAYER_CREATOR(AbsVal_arm)
    
//  arm 内联汇编
// asm(
// 代码列表
// : 输出运算符列表        "r" 表示同用寄存器  "m" 表示内存地址 "I" 立即数 
// : 输入运算符列表        "=r" 修饰符 = 表示只写，无修饰符表示只读，+修饰符表示可读可写，&修饰符表示只作为输出
// : 被更改资源列表
// );
// __asm__　__volatile__(); 
// __volatile__或volatile 是可选的，假如用了它，则是向GCC 声明不答应对该内联汇编优化，
// 否则当 使用了优化选项(-O)进行编译时，GCC 将会根据自己的判定决定是否将这个内联汇编表达式中的指令优化掉。

// 换行符和制表符的使用可以使得指令列表看起来变得美观。
int AbsVal_arm::forward_inplace(Mat& bottom_top_blob) const
{
    int w = bottom_top_blob.w;// 输入特征图宽度
    int h = bottom_top_blob.h;// 输入特征图高度
    int channels = bottom_top_blob.c;// 输入特征图通道数
    int size = w * h;// 一个通道的元素数量

    #pragma omp parallel for // omp并行
    // #pragma omp parallel for num_threads(opt.num_threads)
    for (int q=0; q<channels; q++)//遍历每一个特征通道
    {
        float* ptr = bottom_top_blob.channel(q);// 当前特征通道数据的起始地址指针

// 如果支持ARM_NEON 则使用NEOB进行优化
#if __ARM_NEON
        int nn = size >> 2;// 128位的寄存器，一次可以操作 4个float32位,剩余不够4个的，最后面直接c语言执行
                           // 左移两位相当于除以4
        int remain = size - (nn << 2);// 4*32 =128字节对其后 剩余的 float32个数, 剩余不够4个的数量
        
#else
        int remain = size; // 若不支持优化，则全部使用不同C语言版本进行计算
#endif // __ARM_NEON
        
/*
从内存中载入:
v7:
   带了前缀v的就是v7 32bit指令的标志；
   ld1表示是顺序读取，还可以取ld2就是跳一个读取，ld3、ld4就是跳3、4个位置读取，这在RGB分解的时候贼方便；
   后缀是f32表示单精度浮点，还可以是s32、s16表示有符号的32、16位整型值。
   这里Q寄存器是用q表示，q5对应d10、d11可以分开单独访问（注：v8就没这么方便了。）
   大括号里面最多只有两个Q寄存器。

     "vld1.f32   {q10}, [%3]!        \n"
     "vld1.s16 {q0, q1}, [%2]!       \n" 


v8:
  ARMV8（64位cpu） NEON寄存器 用 v来表示 v1.8b v2.8h  v3.4s v4.2d
  后缀为8b/16b/4h/8h/2s/4s/2d）
  大括号内最多支持4个V寄存器；

  "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"   // 4s表示float32
  "ld1    {v0.8h, v1.8h}, [%2], #32     \n"
  "ld1    {v0.4h, v1.4h}, [%2], #32     \n"             // 4h 表示int16

*/
        
// 优化过程
#if __ARM_NEON
// arm_v8================================
#if __aarch64__ // ARMv8-A 是首款64 位架构的ARM 处理器，是移动手机端使用的CPU
        if (nn > 0)// 这里的循环次数已经是 除以4之后的了
        {
        asm volatile(
            "0:                               \n" // 0: 作为标志，局部标签
            "prfm       pldl1keep, [%1, #128] \n" // %1处为ptr标识为1标识,即数据地址，预取 128个字节 4*32 = 128
            "ld1        {v0.4s}, [%1]         \n" // 载入 ptr 指针对应的值，连续4个
            "fabs       v0.4s, v0.4s          \n" // ptr 指针对应的值 连续4个，使用fabs函数 进行绝对值操作 4s表示浮点数
            "subs       %w0, %w0, #1          \n" // %0 引用 参数 nn 操作次数每次 -1  #1表示1
            "st1        {v0.4s}, [%1], #16    \n" // %1 引用 参数 ptr 指针 向前移动 4*4=16字节
            "bne        0b                    \n" // 如果非0，则向后跳转到 0标志处执行
            
            // 每个操作数的寄存器行为 “=”，表示此操作数类型是只写，即输出寄存器。
            : "=r"(nn),     // %0 操作次数 nn  循环变量
              "=r"(ptr)     // %1 引用参数 ptr 数据内存地址指针
            
             // 数据 标签标识 nn 标识为0  ptr标识为1
            : "0"(nn),  
              "1"(ptr)
            // 寄存器变化表　list of clobbered registers  
            : "cc", "memory", "v0" // v0寄存器，内存memory，cc??可能会变化
        );
        }
#else
        
// arm_v7===========================
        if (nn > 0)
        {
        asm volatile(
            "0:                             \n" // 0: 作为标志，局部标签
            "vld1.f32   {d0-d1}, [%1]       \n" // %1处为ptr标识为1标识,即数据地址，
            "vabs.f32   q0, q0              \n" // q0寄存器 = [d1 d0]，128位寄存器，取出四个 float 单精度浮点数
            "subs       %0, #1              \n" // %0为 循环变量nn标识，标识循环次数-1  #1表示1
            "vst1.f32   {d0-d1}, [%1]!      \n" // ??????
            "bne        0b                  \n" // 如果非0，则向后跳转到 0标志处执行
            // 每个操作数的寄存器行为 “=”，表示此操作数类型是只写，即输出寄存器。
            : "=r"(nn),     // %0
              "=r"(ptr)     // %1
            // 数据 标签标识 nn 标识为0  ptr标识为1
            : "0"(nn),
              "1"(ptr)
            // 寄存器变化表　list of clobbered registers  
            : "cc", "memory", "q0"// q0寄存器，内存memory，cc??可能会变化
        );
        }
#endif // __aarch64__
#endif // __ARM_NEON
        
        // 剩余不够4个的直接c语言执行=====
        for (; remain>0; remain--)// 循环次数-1
        {
            *ptr = *ptr > 0 ? *ptr : - *ptr;
            ptr++;// 指针+1
        }
    }

    return 0;
}

} // namespace ncnn
