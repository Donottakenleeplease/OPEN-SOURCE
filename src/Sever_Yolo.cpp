#include <stdio.h>
#include <string.h>
#include <time.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <fstream>
#include <iostream>
 
#include "mongoose.h"
#include "base64.h"
#include "improcess.h"
#include "darknet.h"

 
using namespace std;
using namespace cv;

int env_handler(struct mg_connection *conn);

float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };
 
float get_color(int c, int x, int max){
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    return r;
}
 
int main(int argc, char *argv[])
{
  struct mg_server* server;
  server = mg_create_server(NULL);                  // 初始化一个mongoose server
  mg_set_option(server, "listening_port", "8003");  // 设置端口号为8003
  mg_add_uri_handler(server, "/", env_handler);     // 设置回调函数
  printf("Starting on port %s ...\n", mg_get_option(server, "listening_port"));  
  while (1) {
    mg_poll_server(server, 100);  // 超时时间（ms）
  }
  mg_destroy_server(&server);
  
  return 0;
}
 
int env_handler(struct mg_connection *conn) 
{
  static int counter = 0;
  counter++;

  const char* encoded_data = conn->content;  // 服务端收到的消息
  int encoded_len = conn -> content_len;
  string str_encoded(encoded_data,encoded_len);

  vector<BYTE> str_decoded_byte = base64_decode(str_encoded);
  Mat mat = imdecode(str_decoded_byte,CV_LOAD_IMAGE_COLOR);

  timeval start, end;
  gettimeofday(&start, NULL);
  gettimeofday(&end, NULL);

/***** 模型调用 ******/
  if(counter == 1)
  {
    string cfgfile = "/home/likuilin/Yolo/darknet/cfg/yolov3.cfg";//读取模型文件，请自行修改相应路径
    string weightfile = "/home/likuilin/Yolo/darknet/yolov3.weights";
    float thresh=0.5;//参数设置
    float nms=0.35;
    int classes=80;

    network *net=load_network((char*)cfgfile.c_str(),(char*)weightfile.c_str(),0);//加载网络模型
    set_batch_network(net, 1);

    Mat rgbImg;

    vector<string> classNamesVec;
    ifstream classNamesFile("/home/likuilin/Yolo/darknet/data/coco.names");//标签文件coco有80类

    if (classNamesFile.is_open())
    {
        string className = "";
        while (getline(classNamesFile, className))
            classNamesVec.push_back(className);
    }    
  }

  /******* 开始网络推理 ********/

  cvtColor(mat, rgbImg, cv::COLOR_BGR2RGB);

  float* srcImg;
  size_t srcSize=rgbImg.rows*rgbImg.cols*3*sizeof(float);
  srcImg=(float*)malloc(srcSize);

  imgConvert(rgbImg,srcImg);//将图像转为yolo形式

  float* resizeImg;
  size_t resizeSize=net->w*net->h*3*sizeof(float);
  resizeImg=(float*)malloc(resizeSize);
  imgResize(srcImg,resizeImg,mat.cols,mat.rows,net->w,net->h);//缩放图像

  network_predict(net,resizeImg);//网络推理
  int nboxes=0;
  detection *dets=get_network_boxes(net,rgbImg.cols,rgbImg.rows,thresh,0.5,0,1,&nboxes);

  if(nms)
  {
    do_nms_sort(dets,nboxes,classes,nms);
  }
  vector<cv::Rect>boxes;
  boxes.clear();
  vector<int>classNames;

  string detect_result = ""; //构造传输
  
  for (int i = 0; i < nboxes; i++) //检测classname与像素坐标
  {
    bool flag=0;
    int className;

    if(!detect_result.empty())
    {
      detect_result += " ";
    }//构造传输的string数据

    for(int j=0;j<classes;j++)
    {
      if(dets[i].prob[j]>thresh)
      {
        if(!flag)
        {
          flag=1;
          className=j;//只检测人则改为0
        }
      }
    }
    if(flag)
    {
      int left = (dets[i].bbox.x - dets[i].bbox.w / 2.)*mat.cols;
      int right = (dets[i].bbox.x + dets[i].bbox.w / 2.)*mat.cols;
      int top = (dets[i].bbox.y - dets[i].bbox.h / 2.)*mat.rows;
      int bot = (dets[i].bbox.y + dets[i].bbox.h / 2.)*mat.rows;

      if (left < 0)
      {
        left = 0;
      }
      if (right > mat.cols - 1)
      {
        right = mat.cols - 1;
      }
    
      if (top < 0)
      {
        top = 0;
      }
      if (bot > mat.rows - 1)
          bot = mat.rows - 1;

      Rect box(left, top, fabs(left - right), fabs(top - bot));
      boxes.push_back(box);
      classNames.push_back(className);

      char ptr_result[30]; // 传输数据构建
      sprintf(ptr_result,"%d %d %d %d",(int)left,(int)top,(int)right,(int)bot);
      string str_result(ptr_result);
      detect_result += str_result;
    }
  }

  free_detections(dets, nboxes); 
  
  for(int i=0;i<boxes.size();i++)
  {

    int offset = classNames[i]*123457 % 80;
    float red = 255*get_color(2,offset,80);
    float green = 255*get_color(1,offset,80);
    float blue = 255*get_color(0,offset,80);

    rectangle(mat,boxes[i],Scalar(blue,green,red),2);

    String label = String(classNamesVec[classNames[i]]);
    int baseLine = 0;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    putText(mat, label, Point(boxes[i].x, boxes[i].y + labelSize.height),
            FONT_HERSHEY_SIMPLEX, 1, Scalar(red, blue, green),2);

  }
  imshow("video",mat);

  free(srcImg);
  free(resizeImg);
  free_network(net);

  printf("Counter: %3d, BBOX: %s, Time of Detect: %f\n", counter, 
                                                         detect_result.empty() ? "Null" : detect_result.c_str(), 
                                                         (double)((end.tv_sec - start.tv_sec)*1000.0 + (end.tv_usec - start.tv_usec)/1000.0));
  mg_printf(conn, "%s", detect_result.c_str());

  return 0;
}
