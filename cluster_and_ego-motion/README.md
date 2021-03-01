## 1、EgoMotion

### （1）成员函数

#### 构造函数

```c++
EgoMotion(int erode_kernal_size, int W, int H);
```

参数：腐蚀核大小（设成3最佳）、图片宽、图片高。

#### 自运动补偿

```C++
void ego_motion(cv::Mat events, std::vector<cv::Vec3d> omegas);
```

参数：事件（3×N的图片）、imu值。

### （2）成员变量

```c++
cv::Mat Count_image, Time_image, Normalized_time_image, Untreshold_normalized_time_image;
```

```C++
std::vector<cv::Vec3d> events;
```

补偿后的事件以及图片。

## 2、Cluster

### 成员函数

#### 构造函数

```C++
Cluster(float threshold, int MinPts);
```

参数：聚类阈值、最少点数（设成200.0、5最佳）

#### 聚类

```C++
std::vector<std::vector<cv::Point> > cluster(cv::Mat &Normalized_time_image, cv::Mat& Time_image);
```

参数：自运动补偿得到的归一化时间图与未归一化时间图

返回：不同类别的像素点

## 3、实例main

```C++
int main()
{
    EgoMotion egomotion = EgoMotion(3, 346, 260)
    Cluster cluster = Cluster(100, 0.8, 0.1, 0.2, 20);
    for(int imgnum = 1;imgnum<=130;imgnum++)
    {
        cv::Mat events = readevent("D://events/events/"+std::to_string(imgnum)+".txt");
        std::vector<cv::Vec3d> omegas = readomega("D://events/imu/"+std::to_string(imgnum)+".txt");

        clock_t start, finish;
        double  duration;
        std::vector<std::vector<cv::Point> > cluster_point;

        start = clock();
        egomotion.ego_motion(events, omegas);
        cluster_point = cluster.cluster(egomotion.Normalized_time_image, egomotion.Untreshold_normalized_time_image);
        finish = clock();

        duration = (double)(finish - start) / CLOCKS_PER_SEC;
        std::cout<<duration<<'\t';

    }

    return 0;
}
```