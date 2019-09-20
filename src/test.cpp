#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <iostream>
 
#include "mongoose.h"
#include "base64.h"
 
using namespace std;
using namespace cv;
using namespace dlib;
 
int env_handler(struct mg_connection *conn);
 
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
  static dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
  static int counter = 0;
  counter++;
 
  const char * encoded_data = conn->content;  // 服务端收到的消息
  int encoded_len = conn->content_len;        // 服务端收到的消息长度
  string str_encoded(encoded_data, encoded_len);
 
  std::vector<BYTE> str_decoded_byte = base64_decode(str_encoded);
  Mat mat = imdecode(str_decoded_byte, CV_LOAD_IMAGE_COLOR);
 
  // 开始人脸检测算法
  dlib::array2d<bgr_pixel> img;  
  dlib::assign_image(img, dlib::cv_image<bgr_pixel>(mat));
 
  timeval start, end;
  gettimeofday(&start, NULL);
  std::vector<dlib::rectangle> dets = detector(img);
  gettimeofday(&end, NULL);
 
  std::string detect_result = "";
  for (int i = 0; i < dets.size(); i++)
  {
    if (!detect_result.empty()) detect_result += " ";
 
    char ptr_result[30];
    sprintf(ptr_result, "%d %d %d %d", (int)dets[i].left(), (int)dets[i].top(), (int)dets[i].right(), (int)dets[i].bottom());
    string str_result(ptr_result);
    detect_result += str_result;
  }
 
  printf("Counter: %3d, BBOX: %s, Time of Detect: %f\n", counter, 
                                                         detect_result.empty() ? "Null" : detect_result.c_str(), 
                                                         (double)((end.tv_sec - start.tv_sec)*1000.0 + (end.tv_usec - start.tv_usec)/1000.0));
  mg_printf(conn, "%s", detect_result.c_str());
  
  return 0;
}