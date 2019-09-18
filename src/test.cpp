#include <stdio.h>
#include <string.h>
#include <time.h>
#include <iostream>
 
#include "mongoose.h"
 
using namespace std;
 
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
  static int counter = 0;
  counter++;
  printf("counter: %d\n", counter);
 
  const char * str_data = conn->content;  // 服务端收到的消息
  int str_len = conn->content_len;        // 服务端收到的消息长度
  string str(str_data, str_len);
  mg_printf(conn, "Received: %s, %d", str.c_str(), counter);
  
  return 0;
}
